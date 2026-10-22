"""Writer-level summaries, paired gains, bootstrap intervals and learning curves."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics


def mean(values):
    return statistics.mean(values)


def interval(values, iterations=2000):
    # Resample independent writers, after averaging a writer's seeds.
    rng = random.Random(42)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(iterations))
    return samples[int(.025 * iterations)], samples[min(iterations - 1, int(.975 * iterations))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    cells, baselines, raw_rows = defaultdict(list), {}, []
    general_base, general_rows = {}, []
    for path in sorted(Path(args.root).rglob("metrics.json")):
        report = json.loads(path.read_text())
        meta = report["meta"]
        if report.get("limit") is not None:
            continue  # Smoke-test metrics must never enter paper tables.
        method, k = meta["method"], meta.get("k")
        if report["kind"] == "general":
            if method == "base":
                general_base[report["manifest_sha256"]] = report["micro"]["exact"]
            elif k is not None:
                general_rows.append((report, path))
            continue
        if k is None:
            continue
        for writer, score in report["by_writer"].items():
            if writer == "UNKNOWN":
                continue
            baseline_key = (writer, report["manifest_sha256"])
            if method == "base":
                baselines[baseline_key] = score["exact"]
            row = {"writer_id": writer, "method": method, "k": k,
                   "seed": meta.get("seed", -1), "exact": score["exact"],
                   "within_1_token": score["within_1_token"], "within_2_tokens": score["within_2_tokens"],
                   "mean_ed": score["mean_ed"], "mean_ned": score["mean_ned"],
                   "test_sha256": report["manifest_sha256"],
                   "base_sha256": meta.get("base_sha256", report["checkpoint_sha256"]),
                   "trainable_parameters": meta.get("trainable_parameters", 0),
                   "training_seconds": meta.get("training_seconds", 0),
                   "peak_cuda_bytes": meta.get("peak_cuda_bytes"), "source": str(path)}
            raw_rows.append(row)
    if not raw_rows:
        raise SystemExit("No final writer evaluations found. Run baseline and personalization evaluation first.")
    write_csv(folder / "per_run.csv", raw_rows)
    # Do not silently combine results from different bases or target test sets.
    base_hashes = {r["base_sha256"] for r in raw_rows}
    if len(base_hashes) > 1:
        raise ValueError("Multiple shared bases found. Analyze each experiment root separately.")
    test_versions = defaultdict(set)
    for r in raw_rows:
        test_versions[r["writer_id"]].add(r["test_sha256"])
    if any(len(values) > 1 for values in test_versions.values()):
        raise ValueError("A writer has multiple test manifests. Analyze separate experiment roots.")
    for row in raw_rows:
        if row["method"] != "base":
            cells[(row["method"], row["k"], row["writer_id"])].append(row)
    for key, runs in cells.items():
        seeds = [r["seed"] for r in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate evaluations for writer/method/budget/seed: {key}")
    seed_sets = {tuple(sorted(r["seed"] for r in runs)) for runs in cells.values()}
    if len(seed_sets) > 1:
        raise ValueError("Methods/budgets/writers have different completed seed sets. Finish missing jobs first.")
    aggregated = defaultdict(list)
    for (method, k, writer), runs in cells.items():
        baseline = baselines.get((writer, runs[0]["test_sha256"]))
        if baseline is None:
            raise ValueError(f"Missing matched base evaluation for {writer}")
        aggregated[(method, k)].append({"writer": writer, "exact": mean(r["exact"] for r in runs),
                                        "gain": mean(r["exact"] for r in runs) - baseline,
                                        "ed": mean(r["mean_ed"] for r in runs),
                                        "within1": mean(r["within_1_token"] for r in runs),
                                        "within2": mean(r["within_2_tokens"] for r in runs),
                                        "seeds": len({r["seed"] for r in runs})})
    cohorts = {tuple(sorted(r["writer"] for r in values)) for values in aggregated.values()}
    if len(cohorts) > 1:
        raise ValueError("Methods/budgets have different completed writer cohorts. Finish missing jobs first.")
    summary = []
    for (method, k), values in sorted(aggregated.items()):
        low, high = interval([r["exact"] for r in values])
        gain_low, gain_high = interval([r["gain"] for r in values])
        summary.append({"method": method, "k": k, "writers": len(values),
                        "min_seeds_per_writer": min(r["seeds"] for r in values),
                        "macro_exact": mean(r["exact"] for r in values),
                        "ci_low": low, "ci_high": high,
                        "paired_gain_pp": 100 * mean(r["gain"] for r in values),
                        "gain_ci_low_pp": 100 * gain_low, "gain_ci_high_pp": 100 * gain_high,
                        "within_1_token": mean(r["within1"] for r in values),
                        "within_2_tokens": mean(r["within2"] for r in values),
                        "mean_ed": mean(r["ed"] for r in values)})
    write_csv(folder / "summary.csv", summary)
    forgetting = []
    for report, path in general_rows:
        baseline = general_base.get(report["manifest_sha256"])
        if baseline is None:
            raise ValueError("General evaluations need a matched base evaluation on the identical manifest.")
        meta = report["meta"]
        forgetting.append({"writer_id": meta.get("writer_id"), "method": meta["method"],
                           "k": meta["k"], "seed": meta["seed"],
                           "general_exact": report["micro"]["exact"],
                           "forgetting_pp": 100 * (baseline - report["micro"]["exact"]),
                           "source": str(path)})
    if forgetting:
        write_csv(folder / "forgetting.csv", forgetting)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in sorted({r["method"] for r in summary}):
            rows = sorted((r for r in summary if r["method"] == method), key=lambda r: r["k"])
            ax.plot([r["k"] for r in rows], [100*r["macro_exact"] for r in rows], marker="o", label=method)
            ax.fill_between([r["k"] for r in rows], [100*r["ci_low"] for r in rows],
                            [100*r["ci_high"] for r in rows], alpha=.12)
        if baselines:
            baseline = mean(baselines.values()) * 100
            ax.axhline(baseline, color="black", linestyle="--", label="shared base")
        ax.set(xlabel="Labeled adaptation examples per writer", ylabel="Writer-macro exact match (%)",
               title="Personalization on fixed held-out writer images")
        ax.set_xticks([10, 25, 50, 100])
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(folder / "learning_curve.png", dpi=200)
        plt.close(fig)
    except ImportError:
        print("CSV files saved. Install requirements-analysis.txt to produce the plot.")
    print(f"Saved real-data summaries to {folder}; no synthetic performance values were inserted.")


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
