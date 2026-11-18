"""Cut photos of handwritten prompt pages into one image per formula.

Each page photo holds up to 10 formulas from prompts/formulas.txt, written in
order, one per line. Photos are matched to pages by sorted filename (first
photo = page 1) unless --first-page says otherwise.

How a page is split:
1. Even out the lighting, then mark dark pixels as ink.
2. Count ink per row. Rows with almost no ink are gaps.
3. The page has N formulas, so cut at the N-1 widest gaps. Small gaps inside
   a formula (like above a fraction bar) are narrower and stay uncut.
4. Trim each line to its ink, dropping small marks far from the formula
   (such as the page number in the corner).

Writes data/team/images/*.png, data/team.jsonl (the manifest format the
rest of the code reads) and data/team/check.html to review every crop.

Usage:
  python scripts/crop_pages.py --photos data/raw/team/aaditya --writer team01
"""
import argparse
import base64
import html
import io
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

PER_PAGE = 10
PAPER_CUTOFF = 1.0  # Share of the automatic brightness cutoff that counts as paper.


def load_photo(path):
    """Open a photo upright and in grayscale. iPhone HEIC files go through sips."""
    if path.suffix.lower() in {".heic", ".heif"}:
        out = Path(tempfile.mkdtemp()) / (path.stem + ".jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(out)],
                       check=True, capture_output=True)
        path = out
    image = ImageOps.exif_transpose(Image.open(path)).convert("L")
    if max(image.size) > 2000:  # Speed only; formulas stay well above model size.
        image.thumbnail((2000, 2000))
    return image


def paper_region(gray, cutoff=None):
    """True inside the sheet of paper. The paper is the largest bright area;
    carpet, desk and the notebook spine are darker or textured."""
    import cv2
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    level, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # A bit below the automatic cutoff, so shadowed paper still counts.
    bright = np.uint8(blur > (cutoff or PAPER_CUTOFF) * level) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    paper = np.uint8(labels == largest) * 255
    # Fill shadows and ink inside the sheet, then pull in from the edges.
    hull = cv2.convexHull(cv2.findNonZero(paper))
    filled = np.zeros_like(paper)
    cv2.fillPoly(filled, [hull], 255)
    margin = int(0.03 * min(gray.shape))
    return cv2.erode(filled, np.ones((margin, margin), np.uint8)) > 0


def ink_mask(image, cutoff=None):
    """True where there is pen ink on the paper."""
    import cv2
    gray = np.asarray(image, dtype=np.uint8)
    # Dividing by a blurred copy evens out shadows across the page.
    background = cv2.GaussianBlur(gray, (0, 0), 25).astype(float) + 1
    paper = paper_region(gray, cutoff)
    mask = (gray / background < 0.65) & paper
    # Carpet or fabric shows up as dense texture over a large area; pen
    # strokes never fill that much of their surroundings. Remove such areas.
    density = cv2.blur(mask.astype(np.float32), (61, 61))
    texture = cv2.dilate(np.uint8(density > 0.25), np.ones((31, 31), np.uint8)) > 0
    mask &= ~texture
    # Keep stroke-like pieces. Drop specks and long thin lines (grid lines,
    # page edges), which handwriting never forms on its own.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask))
    # Pieces touching the edge of the paper are carpet or page curl; the
    # handwriting never reaches the edge.
    rim = paper & ~cv2.erode(np.uint8(paper), np.ones((15, 15), np.uint8)).astype(bool)
    on_rim = set(np.unique(labels[rim & mask]))
    keep = np.zeros(count, bool)
    height, width = mask.shape
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        # Only long thin pieces count as lines, so strokes like f, 1 or a prime stay.
        thin_line = (w <= 4 and h > 8 * max(w, 1) and h > 0.03 * height) or \
                    (h <= 4 and w > 8 * max(h, 1) and w > 0.2 * width)
        keep[i] = area >= 8 and not thin_line and h < 0.15 * height and i not in on_rim
    return keep[labels]


def gap_runs(empty):
    """(start, end) of each run of True values."""
    runs, start = [], None
    for i, value in enumerate(empty):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(empty)))
    return runs


def split_lines(mask, count):
    """Row ranges of `count` formula lines.

    Finds bands of rows with ink, ignores faint bands (specks, a dotted page
    edge), joins the strong bands into `count` lines at the widest gaps, and
    cuts halfway between neighbouring lines so small marks like dots and
    primes stay with their formula.
    """
    rows = mask.sum(axis=1)
    empty = rows <= max(2, 0.002 * mask.shape[1])
    bands = gap_runs(~empty)
    if not bands:
        raise ValueError("no ink found")
    masses = [rows[s:e].sum() for s, e in bands]
    typical = float(np.median(sorted(masses, reverse=True)[:count]))
    # A real line has some ink and some height; specks and dotted page
    # edges are faint or very flat.
    strong = [(s, e) for (s, e), m in zip(bands, masses)
              if m >= 0.1 * typical and e - s >= 0.0095 * mask.shape[0]]
    if len(strong) < count:
        raise ValueError(f"found {len(strong)} lines, expected {count}")
    order = sorted(range(len(strong) - 1), key=lambda i: strong[i + 1][0] - strong[i][1], reverse=True)
    cuts = sorted(order[:count - 1])
    groups, first = [], 0
    for cut in cuts + [len(strong) - 1]:
        groups.append([strong[first][0], strong[cut][1]])
        first = cut + 1
    # Widen each line to the midpoint of the gap on each side.
    reach = int(0.02 * mask.shape[0])
    edges = [max(0, groups[0][0] - reach)]
    edges += [(groups[i][1] + groups[i + 1][0]) // 2 for i in range(len(groups) - 1)]
    edges += [min(mask.shape[0], groups[-1][1] + reach)]
    return list(zip(edges[:-1], edges[1:]))


def trim(mask, y0, y1):
    """Ink box of one line, keeping only the main group of columns."""
    band = mask[y0:y1]
    cols = band.sum(axis=0) > 0
    runs = gap_runs(cols)  # Runs of columns that have ink.
    if not runs:
        return None
    # Group column runs closer than 8% of the page width; keep the heaviest group.
    groups, gap = [[runs[0]]], 0.08 * mask.shape[1]
    for run in runs[1:]:
        if run[0] - groups[-1][-1][1] > gap:
            groups.append([])
        groups[-1].append(run)
    best = max(groups, key=lambda g: sum(band[:, s:e].sum() for s, e in g))
    x0, x1 = best[0][0], best[-1][1]
    ys = np.flatnonzero(band[:, x0:x1].sum(axis=1))
    return x0, y0 + ys[0], x1, y0 + ys[-1] + 1


def clean_crop(image, mask, box, scale):
    """Dark ink on pure white, like the training images: grid lines, paper
    texture and shadows are removed, ink keeps its shading."""
    x0, y0, x1, y1 = box
    gray = np.asarray(image, dtype=float)[y0:y1, x0:x1]
    ink = mask[y0:y1, x0:x1]
    # Slightly widen the ink area so stroke edges keep their anti-aliasing.
    import cv2
    ink = cv2.dilate(np.uint8(ink), np.ones((3, 3), np.uint8)) > 0
    dark = np.percentile(gray[ink], 5) if ink.any() else 0
    paper = np.median(gray[~ink]) if (~ink).any() else 255
    stretched = np.clip((gray - dark) / max(1.0, paper - dark), 0, 1) * 255
    clean = np.where(ink, stretched, 255).astype(np.uint8)
    result = Image.fromarray(clean)
    if scale != 1:
        size = (max(1, round(result.width * scale)), max(1, round(result.height * scale)))
        result = result.resize(size, Image.Resampling.LANCZOS)
    return result


def thumbnail(image):
    copy = image.copy()
    copy.thumbnail((600, 90))
    buffer = io.BytesIO()
    copy.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", required=True, help="Folder of page photos")
    parser.add_argument("--writer", required=True, help="Writer ID, such as team01")
    parser.add_argument("--first-page", type=int, default=1, help="Page number of the first photo")
    parser.add_argument("--output", default="data/team")
    parser.add_argument("--session", default="session01")
    parser.add_argument("--shadow-pages", default="",
                        help="Pages with a shadow on the paper, e.g. 6,10. These use a lower "
                             "brightness cutoff so shaded paper still counts as paper.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Resize factor for the saved crops")
    parser.add_argument("--corrections",
                        help="TSV of number<TAB>label<TAB>reason for formulas written "
                             "differently from the prompt; the label must match the page")
    parser.add_argument("--skip", action="append", default=[], metavar="PAGE:LINE",
                        help="A crossed-out line to ignore, e.g. 11:10 = 10th line on page 11")
    args = parser.parse_args()

    formulas = [l.strip() for l in Path("prompts/formulas.txt").read_text().splitlines() if l.strip()]
    if args.corrections:
        for line in Path(args.corrections).read_text().splitlines()[1:]:
            number, label, _reason = line.split("\t")
            formulas[int(number) - 1] = label
    photos = sorted(p for p in Path(args.photos).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif"})
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, cards, problems = [], [], []

    skips = {}
    for item in args.skip:
        page, line = map(int, item.split(":"))
        skips.setdefault(page, set()).add(line - 1)

    for page, photo in enumerate(photos, args.first_page):
        first = (page - 1) * PER_PAGE
        labels = formulas[first:first + PER_PAGE]
        if not labels:
            problems.append(f"{photo.name}: page {page} has no formulas")
            continue
        image = load_photo(photo)
        shadow = str(page) in args.shadow_pages.split(",")
        mask = ink_mask(image, 0.85 if shadow else None)
        try:
            lines = split_lines(mask, len(labels) + len(skips.get(page, ())))
        except ValueError as error:
            problems.append(f"{photo.name} (page {page}): {error}")
            continue
        lines = [line for i, line in enumerate(lines) if i not in skips.get(page, ())]
        for offset, ((y0, y1), latex) in enumerate(zip(lines, labels)):
            number = first + offset + 1
            box = trim(mask, y0, y1)
            if box is None:
                problems.append(f"formula {number}: empty line")
                continue
            pad = 12
            x0, ya, x1, yb = box
            box = (max(0, x0 - pad), max(0, ya - pad),
                   min(image.width, x1 + pad), min(image.height, yb + pad))
            crop = clean_crop(image, mask, box, args.scale)
            name = f"{args.writer}-{number:03d}"
            crop.save(out / "images" / f"{name}.png")
            rows.append({"id": name, "image": str((out / "images" / f"{name}.png").resolve()),
                         "latex": latex, "writer_id": args.writer, "dataset": "team",
                         "official_split": "new", "polarity": "dark_on_light",
                         "capture_session": args.session, "expression_id": f"formula{number:03d}"})
            cards.append(f'<tr><td>{number}</td><td><img src="data:image/png;base64,{thumbnail(crop)}"></td>'
                         f'<td class="f" data-tex="{html.escape(latex)}"></td></tr>')

    manifest = out.with_suffix(".jsonl")  # data/team -> data/team.jsonl
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "check.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Crop check</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<style>body{{font-family:sans-serif}} td{{padding:6px 12px;border-bottom:1px solid #ccc}}
img{{background:#fff}}</style></head><body>
<h1>{len(rows)} crops for {html.escape(args.writer)}</h1>
<p>Check that each image shows the formula next to it.</p>
<table>{''.join(cards)}</table>
<script>document.querySelectorAll("td.f").forEach(td=>katex.render(td.dataset.tex,td,{{throwOnError:false}}));</script>
</body></html>""")
    print(f"{len(rows)} crops from {len(photos)} photos -> {manifest}, {out / 'check.html'}")
    for problem in problems:
        print("PROBLEM:", problem)


if __name__ == "__main__":
    main()
