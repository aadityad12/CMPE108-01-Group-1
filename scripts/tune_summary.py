"""Collect learning-rate tuning results into one table.

Reads every OUT/<method>_lr<lr>_s<steps>/{support,heldout}/metrics.json
written by slurm/tune.sbatch and prints a table sorted by held-out exact match.
Also writes OUT/summary.csv.

Pick the setting with the best held-out score per method. A good setting
learns the support set without hurting held-out images.

Usage: python scripts/tune_summary.py runs/tune/NAME
"""
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for run in sorted(root.iterdir()):
    match = re.fullmatch(r"(\w+)_lr([\d.e-]+)_s(\d+)", run.name)
    if not match or not (run / "heldout" / "metrics.json").exists():
        continue
    scores = {split: json.loads((run / split / "metrics.json").read_text())["micro"]
              for split in ("support", "heldout")}
    rows.append({"method": match[1], "lr": match[2], "steps": int(match[3]),
                 "support_exact": round(100 * scores["support"]["exact"], 1),
                 "heldout_exact": round(100 * scores["heldout"]["exact"], 1),
                 "heldout_mean_ed": round(scores["heldout"]["mean_ed"], 3)})

rows.sort(key=lambda r: (r["method"], -r["heldout_exact"], r["heldout_mean_ed"]))
with open(root / "summary.csv", "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{'method':8} {'lr':>6} {'steps':>5} {'support%':>8} {'heldout%':>8} {'heldout ED':>10}")
for r in rows:
    print(f"{r['method']:8} {r['lr']:>6} {r['steps']:>5} {r['support_exact']:>8} "
          f"{r['heldout_exact']:>8} {r['heldout_mean_ed']:>10}")
print(f"{len(rows)} runs; table saved to {root / 'summary.csv'}")
