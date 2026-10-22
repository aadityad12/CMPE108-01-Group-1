"""Create a reviewable experiment list, then run one entry per Slurm array task."""
import argparse
import json
from pathlib import Path
from .data import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--splits", required=True, help="Directory containing split_report.json")
    p.add_argument("--base", required=True)
    p.add_argument("--output", default="runs/personal")
    p.add_argument("--jobs", default="jobs.jsonl")
    p.add_argument("--methods", default="full,decoder,lora")
    p.add_argument("--budgets", default="10,25,50,100")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--general", help="Optional fixed, held-out general test manifest")
    p.add_argument("--augment", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--jobs", default="jobs.jsonl")
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "plan":
        report = json.loads((Path(args.splits) / "split_report.json").read_text())
        jobs = []
        for row in report:
            if row["status"] != "ready":
                continue
            folder = Path(row["folder"])
            for method in args.methods.split(","):
                if method not in ("full", "decoder", "lora"):
                    raise ValueError(method)
                for k in map(int, args.budgets.split(",")):
                    for seed in map(int, args.seeds.split(",")):
                        support = folder / f"adapt_{k}.jsonl"
                        test = folder / "test.jsonl"
                        jobs.append({"base": str(Path(args.base).resolve()),
                                     "base_sha256": file_sha256(args.base),
                                     "manifest": str(support), "manifest_sha256": file_sha256(support),
                                     "test": str(test), "test_sha256": file_sha256(test),
                                     "output": str((Path(args.output) / folder.name / method /
                                                    f"k{k}_s{seed}").resolve()),
                                     "method": method, "seed": seed, "steps": args.steps,
                                     "batch_size": args.batch_size, "augment": args.augment,
                                     "general": str(Path(args.general).resolve()) if args.general else None,
                                     "general_sha256": file_sha256(args.general) if args.general else None})
        Path(args.jobs).write_text("".join(json.dumps(job) + "\n" for job in jobs))
        print(f"Wrote {len(jobs)} experiments to {args.jobs}. Array indices: 0..{len(jobs)-1}")
    else:
        from .train import run_training
        from .predict import evaluate
        jobs = [json.loads(line) for line in Path(args.jobs).read_text().splitlines()]
        if not 0 <= args.index < len(jobs):
            raise ValueError("Job index outside plan")
        job = jobs[args.index]
        for key in ("base", "manifest", "test", "general"):
            if job[key] and file_sha256(job[key]) != job[key + "_sha256"]:
                raise ValueError(f"Experiment input changed after planning: {key}")
        folder = Path(job["output"])
        current = folder / "model.pt"
        checkpoint = run_training(**{k: job[k] for k in ("base", "manifest", "output", "method",
                                                         "seed", "steps", "batch_size", "augment")},
                                  device=args.device, resume=str(current) if current.exists() else None)
        evaluate(checkpoint, job["test"], folder / "writer", device=args.device)
        if job["general"]:
            evaluate(checkpoint, job["general"], folder / "general", device=args.device, kind="general")


if __name__ == "__main__":
    main()
