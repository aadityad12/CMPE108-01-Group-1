"""Learning-rate and step tuning on development writers, on one machine.

Same grid as slurm/tune.sbatch, for when the HPC is not available. For each
development writer folder (adapt_10.jsonl + test.jsonl), each method, learning
rate and step budget: train on the 10 support images, then score the writer's
held-out images. Finished runs are skipped, so it can be stopped and restarted.

Usage:
  python scripts/tune_local.py --writers data/writers/dev --output runs/tune/dev --device cpu
  python scripts/tune_summary.py runs/tune/dev    # table averaged over writers
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

GRID = {
    "lora": [1e-4, 3e-4, 1e-3],
    "decoder": [1e-5, 2e-5, 5e-5],
    "full": [2e-6, 5e-6, 1e-5],
}


def run(command, log):
    with open(log, "w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--writers", required=True, help="Folder of development writer splits")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", default="50,200", help="Comma-separated step budgets")
    parser.add_argument("--methods", default="lora,decoder,full")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    writers = sorted(p for p in Path(args.writers).iterdir() if (p / "adapt_10.jsonl").exists())
    settings = [(m, lr, int(s)) for m in args.methods.split(",") for lr in GRID[m]
                for s in args.steps.split(",")]
    total, done = len(writers) * len(settings), 0
    start = time.time()
    for method, lr, steps in settings:
        for writer in writers:
            run_dir = Path(args.output) / f"{method}_lr{lr:g}_s{steps}" / writer.name
            done += 1
            if (run_dir / "heldout" / "metrics.json").exists():
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            run([sys.executable, "-m", "wlm.train", "--base", "checkpoints/base.pt",
                 "--manifest", str(writer / "adapt_10.jsonl"), "--output", str(run_dir),
                 "--method", method, "--lr", str(lr), "--steps", str(steps),
                 "--batch-size", "4", "--seed", "0", "--device", args.device],
                run_dir / "train.log")
            run([sys.executable, "-m", "wlm.predict", "evaluate", "--checkpoint",
                 str(run_dir / "model.pt"), "--manifest", str(writer / "test.jsonl"),
                 "--output", str(run_dir / "heldout"), "--kind", "writer", "--device", args.device],
                run_dir / "eval.log")
            # Keep disk use small: the tuning checkpoints are not needed afterwards.
            (run_dir / "model.pt").unlink(missing_ok=True)
            elapsed = time.time() - start
            print(f"[{done}/{total}] {method} lr={lr:g} steps={steps} {writer.name} "
                  f"({elapsed / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
