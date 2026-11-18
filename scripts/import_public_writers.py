"""Build a manifest of public CROHME writers that have enough samples.

Two sources, both from CROHME23.zip (https://zenodo.org/records/8428035):

1. CROHME 2023 training writers J021-J025 (58-70 distinct formulas each).
   They were collected in 2023, after the CoMER checkpoint was trained, so the
   base model has never seen them. These are evaluation writers.
   Their labels are raw LaTeX, so they are converted to the model's token
   format. Their images are paper scans, so they are cropped to the ink,
   cleaned up and scaled to match the training images.

2. CROHME 2023 training writers J011, J012 and J015 (about 35 formulas each),
   also unseen by the base model. These are development writers: they are only
   used to tune learning rates and step counts, never for reported results.

CROHME 2019 writer User002 (199 formulas) was tried as a development writer,
but the base model already reads its test set perfectly (100% exact; it was
in CoMER's training data), so it cannot show whether tuning helps.

Usage (after extracting the archive to data/raw/crohme23):
  PYTHONPATH=. python scripts/import_public_writers.py
Writes data/public/images/*.png and data/public.jsonl.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from wlm.data import tokens

ROOT = Path("data/raw/crohme23/TC11_CROHME23")
OUT = Path("data/public")
EVAL_WRITERS = {"J021", "J022", "J023", "J024", "J025"}
DEV_WRITERS = {"J011", "J012", "J015"}
VOCAB = set(Path("third_party/CoMER/comer/datamodule/dictionary.txt").read_text().split())
ANNOTATION = re.compile(r'<annotation type="([^"]+)"[^>]*>([^<]*)<')

ALIASES = {"\\to": "\\rightarrow", "\\ge": "\\geq", "\\le": "\\leq", "\\ne": "\\neq",
           "\\dots": "\\ldots", "\\lt": "<", "\\gt": ">", "\\lbrace": "\\{",
           "\\rbrace": "\\}", "\\vert": "|", "\\mid": "|"}
DROP = {"\\left", "\\right", "\\,", "\\;", "\\!", "\\ ", "\\quad", "\\displaystyle", "\\limits"}


def annotations(path):
    """File-level annotations. Symbol-level ones come later in the file and
    reuse the name "truth", so only the first value of each type is kept."""
    found = {}
    for key, value in ANNOTATION.findall(path.read_text(errors="replace")):
        found.setdefault(key, value)
    return found


# ---------- LaTeX: raw CROHME 2023 string -> model token string ----------

def normalize(latex):
    """Convert raw LaTeX to the model's format, e.g. x^2 -> x ^ { 2 }.

    Script and fraction arguments always get braces, \\left/\\right and spacing
    commands are dropped, and \\sum / \\lim get \\limits before their scripts,
    as in the CROHME training labels. Returns None if anything is left that the
    model cannot output.
    """
    items = [ALIASES.get(t, t) for t in tokens(latex) if t not in DROP]
    items = [t for t in items for t in (["^", "{", "\\prime", "}"] if t == "'" else [t])]
    try:
        out, end = parse(items, 0, stop=None)
    except (IndexError, ValueError):
        return None
    if end != len(items):
        return None
    if any(t not in VOCAB for t in out) or not out or len(out) > 198:
        return None
    return " ".join(out)


def parse(items, i, stop):
    """Normalize tokens from i until `stop` ('}' or None). Returns (tokens, next index)."""
    out = []
    while i < len(items) and items[i] != stop:
        token = items[i]
        if token in ("^", "_"):
            if out and out[-1] in ("\\sum", "\\lim"):
                out.append("\\limits")
            arg, i = argument(items, i + 1)
            out += [token, "{", *arg, "}"]
        elif token == "\\frac":
            top, i = argument(items, i + 1)
            bottom, i = argument(items, i)
            out += ["\\frac", "{", *top, "}", "{", *bottom, "}"]
        elif token == "\\sqrt":
            i += 1
            if i < len(items) and items[i] == "[":
                index, i = parse(items, i + 1, stop="]")
                out += ["\\sqrt", "[", *index, "]"]
                i += 1
            else:
                out.append("\\sqrt")
            arg, i = argument(items, i)
            out += ["{", *arg, "}"]
        elif token == "{":  # A bare group adds nothing; keep its contents.
            inner, i = parse(items, i + 1, stop="}")
            out += inner
            i += 1
        elif token == "}":
            raise ValueError("unbalanced braces")
        else:
            out.append(token)
            i += 1
    if stop is not None and i >= len(items):
        raise ValueError(f"missing {stop}")
    return out, i


def argument(items, i):
    """One argument: a {group} or a single token."""
    if items[i] == "{":
        inner, i = parse(items, i + 1, stop="}")
        return inner, i + 1
    if items[i] in ("\\frac", "\\sqrt"):  # Rare and ambiguous; skip the label.
        raise ValueError("nested command without braces")
    return [items[i]], i + 1


# ---------- Images: paper scan -> clean crop like the training images ----------

def keep_main(profile, mass_share=0.03, gap_share=0.03):
    """First/last index of the ink, ignoring small specks away from the formula."""
    runs, start = [], None
    for i, value in enumerate(profile > 0):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(profile)])
    groups = [[runs[0]]]
    for run in runs[1:]:
        if run[0] - groups[-1][-1][1] > gap_share * len(profile):
            groups.append([])
        groups[-1].append(run)
    total = profile.sum()
    kept = [g for g in groups if sum(profile[s:e].sum() for s, e in g) >= mass_share * total]
    return kept[0][0][0], kept[-1][-1][1]


def clean_scan(path, scale):
    gray = np.asarray(Image.open(path).convert("L"), dtype=float)
    background = np.median(gray)
    ink = gray < background - 40
    x0, x1 = keep_main(ink.sum(axis=0))
    y0, y1 = keep_main(ink[:, x0:x1].sum(axis=1))
    pad = 10
    crop = gray[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    darkest = np.percentile(crop, 1)
    # Stretch contrast so ink is black and paper is white, as in training images.
    clean = np.clip((crop - darkest) / max(1.0, background - 40 - darkest), 0, 1)
    image = Image.fromarray(np.uint8(255 * clean))
    # Pen strokes in these scans are thin; thicken them slightly (MinFilter
    # spreads dark ink) so they look like the training images after scaling.
    image = image.filter(ImageFilter.MinFilter(5))
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def main():
    rows, skipped = [], Counter()

    # 1. CROHME 2023 evaluation writers.
    ink_dir = ROOT / "INKML/train/CROHME2023_train"
    img_dir = ROOT / "IMG/train/CROHME2013_train"  # The archive's name for the 2023 train images.
    # Scale chosen with the base model on 60 images from OTHER 2023 writers
    # (not J021-J025), so the choice cannot favor the evaluation writers:
    # exact match 35.0% at 0.25, 56.7% at 0.4, 68.3% at 0.6, 60.0% at 0.8, 30.0% at 1.0.
    scale = 0.6
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for path in sorted(ink_dir.glob("*.inkml")):
        info = annotations(path)
        writer = info.get("writer")
        if writer not in EVAL_WRITERS | DEV_WRITERS:
            continue
        latex = normalize(info.get("truth", ""))
        if latex is None:
            skipped[f"{writer}: label outside vocabulary"] += 1
            continue
        image = clean_scan(img_dir / f"{path.stem}.png", scale)
        target = OUT / "images" / f"{path.stem}.png"
        image.save(target)
        rows.append({"id": f"CROHME23-{path.stem}", "image": str(target.resolve()), "latex": latex,
                     "writer_id": f"CROHME23:{writer}", "dataset": "CROHME23_train",
                     "official_split": "train", "polarity": "dark_on_light",
                     "role": "evaluation" if writer in EVAL_WRITERS else "development",
                     "raw_latex": info.get("truth")})

    Path("data/public.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = Counter(r["writer_id"] for r in rows)
    distinct = defaultdict(set)
    for r in rows:
        distinct[r["writer_id"]].add(r["latex"])
    for writer in sorted(counts):
        print(f"{writer}: {counts[writer]} images, {len(distinct[writer])} distinct formulas")
    for reason, n in skipped.items():
        print(f"skipped {n}: {reason}")


if __name__ == "__main__":
    main()
