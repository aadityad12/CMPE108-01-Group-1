"""Pair base and adapted predictions, classify token errors, and build a review sheet.

With only --base, describes one model's errors. Error types come from a token
alignment and are suggestions for the manual review, not a structural evaluator.
"""
import argparse
import base64
from collections import Counter
import csv
import html
import io
import json
import math
from pathlib import Path
import random
from .data import tokens

LAYOUT = {"^", "_", "\\frac", "\\sqrt", "\\limits"}
BRACES = {"{", "}"}


def align(a, b):
    """Minimum-edit token alignment as (op, left, right); op is =, sub, del or ins."""
    rows, cols = len(a) + 1, len(b) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            table[i][j] = min(table[i-1][j] + 1, table[i][j-1] + 1,
                              table[i-1][j-1] + (a[i-1] != b[j-1]))
    ops, i, j = [], len(a), len(b)
    while i or j:
        if i and j and table[i][j] == table[i-1][j-1] + (a[i-1] != b[j-1]):
            ops.append(("=" if a[i-1] == b[j-1] else "sub", a[i-1], b[j-1]))
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i-1][j] + 1:
            ops.append(("del", a[i-1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j-1]))
            j -= 1
    return ops[::-1]


def error_type(op, left, right):
    touched = {left, right} - {None}
    if touched & LAYOUT:
        return "layout"
    if touched <= BRACES:
        return "braces"
    if op == "sub":
        return "symbol_substitution" if not touched & BRACES else "braces"
    return "missing_symbol" if op == "del" else "extra_symbol"


def describe(truth, prediction):
    errors = [(op, l, r) for op, l, r in align(tokens(truth), tokens(prediction)) if op != "="]
    types = Counter(error_type(*e) for e in errors)
    confusions = Counter(f"{l} -> {r}" for op, l, r in errors
                         if op == "sub" and error_type(op, l, r) == "symbol_substitution")
    return errors, types, confusions


def mcnemar(improved, regressed):
    """Exact two-sided McNemar p-value from discordant pair counts."""
    n = improved + regressed
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(improved, regressed) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate ids in {path}")
    return by_id


def status(base_ed, adapted_ed):
    if adapted_ed is None:
        return "correct" if base_ed == 0 else "wrong"
    if adapted_ed < base_ed:
        return "improved"
    if adapted_ed > base_ed:
        return "regressed"
    return "unchanged_correct" if base_ed == 0 else "unchanged_wrong"


def thumbnail(path, height=72):
    from PIL import Image, ImageOps
    try:
        with Image.open(path) as raw:
            image = raw.convert("L")
    except OSError:
        return ""
    if image.getpixel((0, 0)) < 128:
        image = ImageOps.invert(image)  # Show every image as dark ink on paper.
    scale = height / image.height
    image = image.resize((max(1, round(image.width * scale)), height))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def marked(truth, prediction):
    # Highlight prediction tokens that differ from the aligned ground truth.
    out = []
    for op, left, right in align(tokens(truth), tokens(prediction)):
        if op == "=":
            out.append(html.escape(right))
        elif op == "del":
            out.append(f'<del title="missing">{html.escape(left)}</del>')
        else:
            out.append(f'<mark title="{op}">{html.escape(right)}</mark>')
    return " ".join(out)


def write_html(path, rows, title, adapted_label):
    cards = []
    for row in rows:
        image = thumbnail(row["image"])
        preds = [("base", row["base_prediction"], row["base_ed"])]
        if row["adapted_prediction"] is not None:
            preds.append((adapted_label, row["adapted_prediction"], row["adapted_ed"]))
        lines = "".join(f'<tr><th>{name} (ED {ed})</th><td><code>{marked(row["truth"], text)}</code>'
                        f'<div class="math">\\({html.escape(text)}\\)</div></td></tr>'
                        for name, text, ed in preds)
        cards.append(f'<section class="{row["status"]}"><h3>{html.escape(row["id"])} · {row["status"]}'
                     f' · suggested: {row["suggested_type"] or "none"}</h3>'
                     f'{f"<img src={chr(34)}{image}{chr(34)}>" if image else ""}<table>'
                     f'<tr><th>truth</th><td><code>{html.escape(row["truth"])}</code>'
                     f'<div class="math">\\({html.escape(row["truth"])}\\)</div></td></tr>{lines}</table>'
                     '<p>Manual tag: ____________________</p></section>')
    Path(path).write_text(f"""<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']]}}}}</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>body{{font:15px system-ui;max-width:1100px;margin:2em auto;color:#182934}}
section{{border:1px solid #ccd;border-left:6px solid #999;border-radius:6px;padding:10px 14px;margin:14px 0;break-inside:avoid}}
.improved{{border-left-color:#2a8}}.regressed{{border-left-color:#c33}}h3{{font-size:14px;margin:0 0 8px}}
img{{max-width:100%;border:1px solid #eee}}th{{text-align:left;width:150px;vertical-align:top}}
mark{{background:#fd8}}del{{color:#c33}}.math{{color:#456;margin:4px 0}}p{{color:#789}}</style>
<h1>{html.escape(title)}</h1><p>Highlighted tokens differ from the aligned truth; struck tokens are missing.
Suggested types are automatic; confirm each by eye.</p>{"".join(cards)}""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="predictions.jsonl of the shared base")
    parser.add_argument("--adapted", help="predictions.jsonl of the adapted model")
    parser.add_argument("--adapted-label", default="adapted")
    parser.add_argument("--output", required=True)
    parser.add_argument("--review", type=int, default=30, help="Cases in the HTML sheet")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    base = load(args.base)
    adapted = load(args.adapted) if args.adapted else None
    if adapted is not None and set(base) != set(adapted):
        raise ValueError("Prediction files cover different ids; evaluate the identical manifest.")
    rows, totals = [], {"base": Counter(), "adapted": Counter()}
    confusions = {"base": Counter(), "adapted": Counter()}
    for key in sorted(base):
        left = base[key]
        right = adapted[key] if adapted else None
        if right is not None and right["truth"] != left["truth"]:
            raise ValueError(f"Ground truth differs for {key}")
        _, base_types, base_conf = describe(left["truth"], left["prediction"])
        totals["base"].update(base_types)
        confusions["base"].update(base_conf)
        adapted_ed = None
        changed_types = base_types
        if right is not None:
            adapted_ed = right["ed"]
            _, adapted_types, adapted_conf = describe(right["truth"], right["prediction"])
            totals["adapted"].update(adapted_types)
            confusions["adapted"].update(adapted_conf)
            # For a regression the new errors matter; otherwise, the ones that were fixed.
            changed_types = adapted_types if adapted_ed > left["ed"] else base_types
        rows.append({"id": key, "image": left["image"], "writer_id": left.get("writer_id"),
                     "truth": left["truth"], "base_prediction": left["prediction"],
                     "adapted_prediction": right["prediction"] if right else None,
                     "base_ed": left["ed"], "adapted_ed": adapted_ed,
                     "delta_ed": None if adapted_ed is None else adapted_ed - left["ed"],
                     "status": status(left["ed"], adapted_ed),
                     "suggested_type": changed_types.most_common(1)[0][0] if changed_types else "",
                     "fraction": left["fraction"], "superscript": left["superscript"],
                     "radical": left["radical"], "long": left["long"], "manual_tag": ""})
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "pairs.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in rows[0] if k != "image"])
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k != "image"} for row in rows)
    counts = Counter(row["status"] for row in rows)
    summary = {"n": len(rows), "status": dict(counts),
               "error_types": {name: dict(c) for name, c in totals.items() if c},
               "top_symbol_confusions": {name: c.most_common(15) for name, c in confusions.items() if c}}
    if adapted is not None:
        improved_exact = sum(r["base_ed"] > 0 and r["adapted_ed"] == 0 for r in rows)
        regressed_exact = sum(r["base_ed"] == 0 and r["adapted_ed"] > 0 for r in rows)
        summary["exact_match_discordant"] = {"fixed": improved_exact, "broken": regressed_exact,
                                             "mcnemar_exact_p": mcnemar(improved_exact, regressed_exact)}
        pool = {name: [r for r in rows if r["status"] == name] for name in ("improved", "regressed")}
    else:
        pool = {"wrong": [r for r in rows if r["status"] == "wrong"]}
    # Balanced, seeded sample so the review includes both improvements and regressions.
    rng, review = random.Random(args.seed), []
    share = max(1, args.review // len(pool))
    for name, items in pool.items():
        review += rng.sample(items, min(share, len(items)))
    write_html(folder / "review.html", review, f"Error review: {Path(args.base).parent.name}"
               + (f" vs {args.adapted_label}" if adapted else ""), args.adapted_label)
    summary["reviewed_ids"] = [r["id"] for r in review]
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "reviewed_ids"}, indent=2))


if __name__ == "__main__":
    main()
