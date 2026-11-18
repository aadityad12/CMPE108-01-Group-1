"""Collect learning-rate tuning results into one table.

Works with both layouts:
- HPC (slurm/tune.sbatch): OUT/<method>_lr<lr>_s<steps>/heldout/metrics.json
- Local (scripts/tune_local.py): OUT/<method>_lr<lr>_s<steps>/<writer>/heldout/metrics.json
  plus OUT/base/<writer>/heldout/metrics.json for the untuned base model.

Scores are averaged over writers (each writer counts equally). Prints a table
sorted by held-out exact match within each method and writes OUT/summary.csv.
Pick the setting with the best held-out score per method.

Usage: python scripts/tune_summary.py runs/tune/NAME
"""
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean

root = Path(sys.argv[1])


def scores(folder):
    """Held-out metrics for one setting, one entry per writer."""
    files = sorted(folder.glob("heldout/metrics.json")) or sorted(folder.glob("*/heldout/metrics.json"))
    return [json.loads(f.read_text())["micro"] for f in files]


rows = []
for run in sorted(root.iterdir()):
    match = re.fullmatch(r"(\w+?)_lr([\d.e-]+)_s(\d+)", run.name)
    if run.name == "base":
        method, lr, steps = "base", "-", 0
    elif match:
        method, lr, steps = match[1], match[2], int(match[3])
    else:
        continue
    found = scores(run)
    if not found:
        continue
    rows.append({"method": method, "lr": lr, "steps": steps, "writers": len(found),
                 "heldout_exact": round(100 * mean(s["exact"] for s in found), 1),
                 "heldout_within_1": round(100 * mean(s["within_1_token"] for s in found), 1),
                 "heldout_mean_ed": round(mean(s["mean_ed"] for s in found), 2)})

rows.sort(key=lambda r: (r["method"] != "base", r["method"], -r["heldout_exact"], r["heldout_mean_ed"]))
with open(root / "summary.csv", "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{'method':8} {'lr':>6} {'steps':>5} {'writers':>7} {'exact%':>7} {'within1%':>8} {'mean ED':>7}")
for r in rows:
    print(f"{r['method']:8} {r['lr']:>6} {r['steps']:>5} {r['writers']:>7} {r['heldout_exact']:>7} "
          f"{r['heldout_within_1']:>8} {r['heldout_mean_ed']:>7}")
print(f"Table saved to {root / 'summary.csv'}")
