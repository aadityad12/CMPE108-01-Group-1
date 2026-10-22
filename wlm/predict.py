"""Run real CoMER bidirectional beam search and evaluate held-out images."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import torch
from .data import file_sha256, make_batch, read_rows, write_rows
from .metrics import score_example, summarize
from .model import choose_device, load_model


@torch.no_grad()
def infer(model, raw, rows, device, beam_size=None):
    from comer.datamodule import vocab
    model.eval()
    images, mask, _ = make_batch(rows, device, labels=False)
    decode = dict(raw["decode"])
    if beam_size is not None:
        decode["beam_size"] = beam_size
    hypotheses = model.beam_search(images, mask, **decode)
    return [vocab.indices2label(hypothesis.seq) for hypothesis in hypotheses]


def evaluate(checkpoint, manifest, output, device="auto", batch_size=1,
             beam_size=None, limit=None, base_override=None, kind="writer", as_base=False):
    device = choose_device(str(device))
    model, raw = load_model(checkpoint, device, base_override)
    rows = read_rows(manifest)
    if limit is not None:
        rows = rows[:limit]
    records = []
    started = time.monotonic()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        predictions = infer(model, raw, batch, device, beam_size)
        for row, prediction in zip(batch, predictions):
            records.append({"id": row["id"], "image": row["image"], "writer_id": row["writer_id"],
                            "truth": row["latex"], "prediction": prediction,
                            **score_example(row["latex"], prediction)})
        print(f"Decoded {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["writer_id"] or "UNKNOWN"].append(record)
    by_writer = {writer: summarize(examples) for writer, examples in grouped.items()}
    metadata = dict(raw["meta"])
    if as_base:
        metadata.update(method="base", k=0, base_sha256=file_sha256(checkpoint))
    summary = {"kind": kind, "checkpoint": str(Path(checkpoint).resolve()),
               "checkpoint_sha256": file_sha256(checkpoint), "manifest_sha256": file_sha256(manifest),
               "meta": metadata, "micro": summarize(records), "by_writer": by_writer,
               "writer_macro_exact": sum(s["exact"] for s in by_writer.values()) / len(by_writer),
               "seconds": time.monotonic() - started, "device": str(device),
               "beam_size": beam_size or raw["decode"]["beam_size"], "limit": limit,
               "categories": {category: summarize([e for e in records if e[category]])
                              for category in ("fraction", "superscript", "radical", "long")
                              if any(e[category] for e in records)}}
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (folder / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["micro"], indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("image")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--polarity", choices=["dark_on_light", "light_on_dark"], required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--beam-size", type=int)
    p.add_argument("--base-override")
    p = sub.add_parser("evaluate")
    for name in ("checkpoint", "manifest", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--beam-size", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--base-override")
    p.add_argument("--kind", choices=["writer", "general"], default="writer")
    p.add_argument("--as-base", action="store_true", help="Label the shared starting model as a baseline.")
    args = parser.parse_args()
    if args.command == "evaluate":
        params = vars(args).copy()
        params.pop("command")
        evaluate(**params)
    else:
        device = choose_device(args.device)
        model, raw = load_model(args.checkpoint, device, args.base_override)
        row = {"image": args.image, "polarity": args.polarity}
        print(infer(model, raw, [row], device, args.beam_size)[0])


if __name__ == "__main__":
    main()
