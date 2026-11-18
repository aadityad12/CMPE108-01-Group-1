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


def ink_mask(image):
    """True where there is ink. Dividing by a blurred copy removes shadows."""
    background = np.asarray(image.filter(ImageFilter.GaussianBlur(25)), dtype=float) + 1
    ratio = np.asarray(image, dtype=float) / background
    mask = ratio < 0.75
    # Ignore a thin border, where page edges and table shadows show up.
    edge = int(0.02 * min(mask.shape))
    mask[:edge, :] = mask[-edge:, :] = mask[:, :edge] = mask[:, -edge:] = False
    return mask


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
    """Row ranges of `count` formula lines, cut at the widest gaps."""
    rows = mask.sum(axis=1)
    empty = rows <= max(2, 0.002 * mask.shape[1])
    ink_rows = np.flatnonzero(~empty)
    if len(ink_rows) == 0:
        raise ValueError("no ink found")
    top, bottom = ink_rows[0], ink_rows[-1] + 1
    inner = [(s, e) for s, e in gap_runs(empty[top:bottom])]
    if len(inner) < count - 1:
        raise ValueError(f"found {len(inner) + 1} lines, expected {count}")
    widest = sorted(inner, key=lambda r: r[1] - r[0], reverse=True)[:count - 1]
    cuts = sorted(top + (s + e) // 2 for s, e in widest)
    edges = [top] + cuts + [bottom]
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
    args = parser.parse_args()

    formulas = [l.strip() for l in Path("prompts/formulas.txt").read_text().splitlines() if l.strip()]
    photos = sorted(p for p in Path(args.photos).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif"})
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, cards, problems = [], [], []

    for page, photo in enumerate(photos, args.first_page):
        first = (page - 1) * PER_PAGE
        labels = formulas[first:first + PER_PAGE]
        if not labels:
            problems.append(f"{photo.name}: page {page} has no formulas")
            continue
        image = load_photo(photo)
        mask = ink_mask(image)
        try:
            lines = split_lines(mask, len(labels))
        except ValueError as error:
            problems.append(f"{photo.name} (page {page}): {error}")
            continue
        for offset, ((y0, y1), latex) in enumerate(zip(lines, labels)):
            number = first + offset + 1
            box = trim(mask, y0, y1)
            if box is None:
                problems.append(f"formula {number}: empty line")
                continue
            pad = 12
            x0, ya, x1, yb = box
            crop = image.crop((max(0, x0 - pad), max(0, ya - pad),
                               min(image.width, x1 + pad), min(image.height, yb + pad)))
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
