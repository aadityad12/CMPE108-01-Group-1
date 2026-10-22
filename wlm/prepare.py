"""Import real dataset labels, audit them, and create reproducible writer splits."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET
import zipfile
from .data import canonical, file_sha256, read_rows, tokens, write_rows


def import_upstream(args):
    archive = Path("third_party/CoMER/data.zip")
    target = Path(args.images).resolve()
    rows = []
    with zipfile.ZipFile(archive) as z:
        lines = z.read(f"data/{args.split}/caption.txt").decode().splitlines()
        for line in lines:
            name, latex = line.split(maxsplit=1)
            path = target / args.split / f"{name}.bmp"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(f"data/{args.split}/img/{name}.bmp"))
            rows.append({"id": f"upstream-{args.split}-{name}", "image": str(path), "latex": latex,
                         "writer_id": None, "dataset": "CoMER-bundled-CROHME",
                         "official_split": args.split, "polarity": "light_on_dark"})
    write_rows(args.output, rows)
    print(f"Imported {len(rows)} examples. Writer IDs are unknown; use for general evaluation only.")


def import_hme(args):
    rows = []
    for line_number, line in enumerate(Path(args.labels).read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Bad label line {line_number}: expected filename then LaTeX")
        name, latex = parts
        rows.append({"id": f"HME100K-{args.split}-{name}",
                     "image": str((Path(args.images) / name).resolve()), "latex": latex,
                     "writer_id": None, "dataset": "HME100K", "official_split": args.split,
                     "polarity": "dark_on_light"})
    write_rows(args.output, rows)
    print(f"Imported {len(rows)} HME100K labels; no writer IDs inferred.")


def import_crohme(args):
    ink_root, image_root = Path(args.inkml), Path(args.images)
    images = defaultdict(list)
    for path in image_root.rglob("*"):
        if path.suffix.lower() in (".png", ".bmp", ".jpg", ".jpeg"):
            images[path.stem].append(path)
    writers = {}
    if args.writer_map:
        with open(args.writer_map, newline="") as stream:
            for row in csv.DictReader(stream):
                if row["id"] in writers:
                    raise ValueError(f"Duplicate writer-map id: {row['id']}")
                writers[row["id"]] = row["writer_id"]
    rows, rejected = [], []
    for path in sorted(ink_root.rglob("*.inkml")):
        annotations = {}
        root = ET.parse(path).getroot()
        for child in root:
            if child.tag.rsplit("}", 1)[-1] == "annotation":
                key = re.sub(r"[^a-z]", "", child.attrib.get("type", "").lower())
                annotations[key] = "".join(child.itertext()).strip()
        latex = annotations.get("truth") or annotations.get("latex")
        matches = images[path.stem]
        if not latex or len(matches) != 1:
            rejected.append({"inkml": str(path), "reason": "missing truth or non-unique image stem"})
            continue
        relative_id = path.relative_to(ink_root).as_posix()
        writer = writers.get(relative_id) or writers.get(path.stem)
        writer = writer or annotations.get("writerid") or annotations.get("writer")
        rows.append({"id": f"{args.dataset}-{relative_id}", "image": str(matches[0].resolve()),
                     "latex": latex, "writer_id": f"{args.dataset}:{writer}" if writer else None,
                     "dataset": args.dataset, "official_split": args.split, "polarity": args.polarity,
                     "inkml": str(path.resolve()), "annotations": annotations})
    write_rows(args.output, rows)
    Path(str(args.output) + ".rejected.json").write_text(json.dumps(rejected, indent=2))
    print(f"Imported {len(rows)}; rejected {len(rejected)}; "
          f"missing writer IDs: {sum(not r['writer_id'] for r in rows)}.")


def audit(args):
    dictionary = Path("third_party/CoMER/comer/datamodule/dictionary.txt")
    known = set(dictionary.read_text().splitlines())
    rows, seen_ids, seen_images, accepted = read_rows(args.manifest), set(), {}, []
    writers, unknown, problems = Counter(), Counter(), []
    for row in rows:
        reasons = []
        if row["id"] in seen_ids:
            reasons.append("duplicate id")
        seen_ids.add(row["id"])
        path = Path(row["image"])
        if not path.is_file():
            reasons.append("missing image")
        else:
            try:
                from PIL import Image
                with Image.open(path) as im:
                    im.verify()
            except Exception as error:
                reasons.append(f"unreadable image: {error}")
            digest = file_sha256(path)
            if digest in seen_images:
                reasons.append(f"duplicate image bytes: {seen_images[digest]}")
            seen_images[digest] = row["id"]
        sequence = tokens(row["latex"])
        oov = [t for t in sequence if t not in known]
        unknown.update(oov)
        if oov:
            reasons.append("out-of-vocabulary tokens")
        if not sequence or len(sequence) > 198:
            reasons.append("empty label or more than 198 tokens")
        writers[row["writer_id"] or "UNKNOWN"] += 1
        if reasons:
            problems.append({"id": row["id"], "reasons": reasons, "oov": oov})
        else:
            accepted.append(row)
    report = {"n": len(rows), "eligible_training_rows": len(accepted),
              "writer_counts": dict(writers), "unknown_tokens": dict(unknown), "problems": problems}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    if args.filtered:
        write_rows(args.filtered, accepted)
    print(json.dumps({k: v for k, v in report.items() if k != "problems"}, indent=2))
    print("Filtering is for training. Preserve every test row and report OOV coverage separately.")


def writer_splits(args):
    rows = read_rows(args.manifest)
    requested = set(args.writers.split(",")) if args.writers else None
    grouped = defaultdict(list)
    for row in rows:
        writer = row["writer_id"]
        if writer and (requested is None or writer in requested):
            grouped[writer].append(row)
    if requested and not requested.issubset(grouped):
        raise ValueError(f"Writer IDs not found: {requested - set(grouped)}")
    summary = []
    for writer, items in sorted(grouped.items()):
        # Keep one acquisition per distinct normalized formula per writer. This
        # prevents a repeated formula/photo variant entering support and test.
        unique = {}
        for row in sorted(items, key=lambda r: r["id"]):
            unique.setdefault(canonical(row["latex"]), row)
        items = list(unique.values())
        if len(items) < 100 + args.test_size:
            summary.append({"writer_id": writer, "status": "insufficient",
                            "unique_formulas": len(items), "needed": 100 + args.test_size})
            continue
        random.Random(f"{args.seed}:{writer}").shuffle(items)
        test, pool = items[:args.test_size], items[args.test_size:]
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", writer)
        # Hash suffix avoids collisions between sanitized writer IDs.
        import hashlib
        slug += "_" + hashlib.sha256(writer.encode()).hexdigest()[:8]
        folder = Path(args.output) / slug
        write_rows(folder / "test.jsonl", test)
        for k in (10, 25, 50, 100):
            write_rows(folder / f"adapt_{k}.jsonl", pool[:k])
        summary.append({"writer_id": writer, "folder": str(folder.resolve()), "status": "ready",
                        "unique_formulas": len(items), "test_size": len(test),
                        "unused": len(pool) - 100})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def domain_split(args):
    # Use only the official training partition for this split.
    rows = read_rows(args.manifest)
    if any(row.get("official_split") != "train" for row in rows):
        raise ValueError("Domain splits require official_split=train for every row.")
    groups = defaultdict(list)
    for row in rows:
        groups[canonical(row["latex"])].append(row)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    n = max(1, round(len(keys) * args.val_fraction))
    if n >= len(keys):
        raise ValueError("Need at least two distinct formulas for train/validation.")
    val_keys = set(keys[:n])
    train = [r for key, values in groups.items() if key not in val_keys for r in values]
    val = [r for key, values in groups.items() if key in val_keys for r in values]
    write_rows(Path(args.output) / "train.jsonl", train)
    write_rows(Path(args.output) / "val.jsonl", val)
    print(f"Domain train={len(train)} val={len(val)}; formula groups kept together.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("upstream")
    p.add_argument("--split", choices=["train", "2014", "2016", "2019"], required=True)
    p.add_argument("--images", default="data/upstream_images")
    p.add_argument("--output", required=True)
    p.set_defaults(func=import_upstream)
    p = sub.add_parser("hme100k")
    for flag in ("labels", "images", "output"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.set_defaults(func=import_hme)
    p = sub.add_parser("crohme")
    for flag in ("inkml", "images", "output", "dataset", "split"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--writer-map")
    p.add_argument("--polarity", choices=["dark_on_light", "light_on_dark"], required=True)
    p.set_defaults(func=import_crohme)
    p = sub.add_parser("audit")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--filtered")
    p.set_defaults(func=audit)
    p = sub.add_parser("writer-splits")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="data/writers")
    p.add_argument("--writers", help="Comma-separated exact, namespaced writer IDs")
    p.add_argument("--test-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=writer_splits)
    p = sub.add_parser("domain-split")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="data/domain")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=domain_split)
    args = parser.parse_args()
    if args.command == "domain-split" and not 0 < args.val_fraction < 1:
        parser.error("val-fraction must be between zero and one")
    if args.command == "writer-splits" and args.test_size < 1:
        parser.error("test-size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
