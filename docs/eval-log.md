# Evaluation log

Running record of evaluations: what was run, where, and what came out.
Raw outputs live in `runs/` (git-ignored); this file keeps the numbers.

Metrics are token-level (see `wlm/metrics.py`), not official CROHME symbol-graph scores.

## 2026-11-03 · Error analysis of the base model on CROHME 2019

Command: `python -m wlm.compare --base runs/mac_reference/base_general_2019/predictions.jsonl --output results/error_analysis/base_2019 --review 30`.
Error types come from a token alignment (`wlm/compare.py`); they are suggestions that the
manual review confirms, not a structural evaluator.

| Token errors in the 448 wrong predictions | Count |
|---|---|
| Symbol substitution | 601 |
| Brace error | 381 |
| Missing symbol | 209 |
| Layout (`^`, `_`, `\frac`, `\sqrt`, `\limits`) | 178 |
| Extra symbol | 150 |

- 175 of 448 wrong predictions (39.1%) are wrong by **one symbol substitution only**.
- Most frequent substitutions: `C→c` 19, `\ldots→\cdots` 16, `X→x` 16, `|→1` 11, `9→q` 10,
  `P→p` 10, `S→5` 9, `r→\gamma` 8, `V→v` 8, `z→2` 7.
- Case-only confusions (same letter, other case) are 82 of 601 substitutions (13.6%).
- These are glyph-shape ambiguities that depend on how a person writes, which is what
  writer personalization should be able to fix. Watch these pairs in the writer study.
- 30-case review sheet: `results/error_analysis/base_2019/review.html` (manual tags to do).

## 2026-11-03 · Reproduction from a fresh clone

Rebuilt `.venv` from `env/mac-cpu-lock.txt`, reran `bootstrap.py`: identical `base.pt`
hash (`2bb8c70d…109f`). Reran the full base evaluation below: **identical metrics** on every
field. Decode time 1,565 s (≈1.3 s/image). The CPU pipeline is deterministic.

## 2026-10-21 · Base model on bundled CROHME 2019 (Mac CPU reference)

| Field | Value |
|---|---|
| Checkpoint | `checkpoints/base.pt` (sha256 `2bb8c70d…109f`) |
| Manifest | `data/general_2019.jsonl` from `wlm.prepare upstream --split 2019` |
| Device | Mac CPU (Apple Silicon), beam size 10, batch size 1 |
| Command | `python -m wlm.predict evaluate --checkpoint checkpoints/base.pt --manifest data/general_2019.jsonl --output runs/mac_reference/base_general_2019 --kind general --as-base --device cpu` |
| Decode time | 1,580 s for 1,199 images (≈1.3 s/image) |

| Metric | Value |
|---|---|
| n | 1,199 |
| Exact token match | 62.64% |
| Within 1 token | 80.07% |
| Within 2 tokens | 86.16% |
| Mean edit distance | 1.267 |
| Mean normalized edit distance | 0.066 |
| Token error rate | 7.57% |

By category (overlapping, descriptive):

| Category | n | Exact |
|---|---|---|
| Fraction | 352 | 65.6% |
| Superscript | 427 | 52.5% |
| Radical | 210 | 62.9% |
| Long (≥ 50 tokens) | 23 | 8.7% |

Edit-distance distribution: 751 exact, 209 at 1, 73 at 2, 37 at 3, 25 at 4, 104 at ≥ 5.

Notes:

- Exploratory reference only. Final base numbers get re-run on the HPC on the same
  device as the adapted models.
- Writer IDs are unknown for this set, so it measures general recognition and
  forgetting only, never personalization.
- Superscripts and long expressions are the weakest areas, which makes them good
  candidates for the error analysis.
- For HPC planning: decoding the full general set is not free (≈26 min on CPU).
  Time it on a GPU in the pilot before adding it to every sweep job.

## 2026-10-21 · Smoke test (20 images)

Same command with `--limit 20`: 75% exact, 4.6 s total. Smoke results are marked
with `limit` and excluded by `wlm.analyze`.
