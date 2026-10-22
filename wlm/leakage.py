"""Check two manifests before running experiments; missing IDs cannot prove isolation."""
import argparse
from collections import defaultdict
import json
from .data import canonical, file_sha256, read_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--require-writer-disjoint", action="store_true")
    p.add_argument("--require-formula-disjoint", action="store_true")
    args = p.parse_args()
    left, right = read_rows(args.left), read_rows(args.right)
    left_writers = {r["writer_id"] for r in left if r["writer_id"]}
    right_writers = {r["writer_id"] for r in right if r["writer_id"]}
    hashes = {file_sha256(r["image"]): r["id"] for r in left}
    duplicates = [(hashes[file_sha256(r["image"])], r["id"]) for r in right if file_sha256(r["image"]) in hashes]
    left_formulas = {canonical(r["latex"]) for r in left}
    right_formulas = {canonical(r["latex"]) for r in right}
    report = {"shared_ids": sorted({r['id'] for r in left} & {r['id'] for r in right}),
              "identical_image_bytes": duplicates, "shared_writers": sorted(left_writers & right_writers),
              "unknown_writer_rows": sum(not r["writer_id"] for r in left + right),
              "shared_formulas": len(left_formulas & right_formulas)}
    print(json.dumps(report, indent=2))
    if report["shared_ids"] or duplicates:
        raise SystemExit("Leakage: repeated IDs or image bytes across these manifests.")
    if args.require_writer_disjoint and (report["shared_writers"] or report["unknown_writer_rows"]):
        raise SystemExit("Writer disjointness is not established.")
    if args.require_formula_disjoint and report["shared_formulas"]:
        raise SystemExit("Repeated normalized formulas across these manifests.")


if __name__ == "__main__":
    main()
