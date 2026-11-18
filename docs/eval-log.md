# Evaluation log

Running record of evaluations: what was run, where, and what came out.
Raw outputs live in `runs/` (git-ignored); this file keeps the numbers.

Metrics are token-level (see `wlm/metrics.py`), not official CROHME symbol-graph scores.

## 2026-11-17 · Base model on the evaluation writers' test sets (Mac CPU)

Before any adaptation. Test sets are frozen (seed 42, see `docs/dataset/README.md`).
Beam 10. Command per writer:
`python -m wlm.predict evaluate --checkpoint checkpoints/base.pt --manifest data/writers/<group>/<writer>/test.jsonl --output runs/study/base/<writer> --kind writer --as-base --device cpu`

| Writer | Test n | Exact | Within 1 | Within 2 | Mean ED |
|---|---|---|---|---|---|
| team01 (own handwriting, phone photos) | 50 | 54.0% | 84.0% | 90.0% | 0.92 |
| CROHME23 J021 | 25 | 32.0% | 56.0% | 68.0% | 2.80 |
| CROHME23 J022 | 25 | 48.0% | 60.0% | 76.0% | 1.92 |
| CROHME23 J023 | 25 | 48.0% | 60.0% | 64.0% | 5.04 |
| CROHME23 J024 | 25 | 36.0% | 48.0% | 56.0% | 1.84 |
| CROHME23 J025 | 25 | 52.0% | 68.0% | 76.0% | 1.84 |
| CROHME23, all 5 pooled | 125 | 43.2% | 58.4% | 68.0% | 2.69 |

- team01: 30% of test formulas are wrong by exactly one token. That is the
  room writer personalization is meant to close.
- The 2023 writers score well below the CROHME 2019 test set (62.6%). They are
  new writers on paper scans, so this is a harder, more realistic setting.
- Development writers J011, J012, J015 (tuning only): 22.7% exact on average.

## 2026-11-17 · Demo dry run on the Mac (CPU)

`python app.py --device cpu`, driven through its two endpoints (adapt, recognize)
with the 10 pilot support images (inverted to dark ink on white, as the demo expects)
and one held-out image. Method LoRA, 200 steps.

| Check | Result |
|---|---|
| Base recognition of a new image | Correct, 1.4 s |
| Adapt to 10 labeled images | Finished in 28 s |
| Label line names a missing file | Rejected: "use filename, one TAB, then LaTeX" |
| Label uses a token outside the vocabulary | Rejected, names the token |
| Fewer than 10 images | Rejected: "Use 10 to 100 labeled expression images" |

The demo can run live on a laptop during the presentation. No HPC or tunnel needed.

## 2026-11-03 · CPU pilot: full, decoder and LoRA on 10 hard images

**Not a personalization result.** The support images come from many CROHME 2014 writers
(no IDs). The pilot checks that each method trains and decodes end to end, whether it can fit
a tiny support set (guide stage 2), what it does to images it never saw, and what it costs.

Setup:

- Screen: 60 random CROHME 2014 test images (`random.Random(7)`), base 63.3% exact.
- Support: the first 10 screened images the base got wrong with ED 1–4 and in-vocabulary
  labels → `data/pilot/support_10.jsonl` (writer ID set to `pilot-crohme2014`).
- Held-out: the other 50 screened 2014 images; general: 200 fixed CROHME 2019 images
  (`random.Random(0)`). Neither overlaps the support set.
- 200 steps, batch 4, seed 0, default LRs (full 1e-5, decoder 5e-5, LoRA 1e-3), beam 10,
  Mac M1 Pro CPU. Command: `scripts/cpu_pilot.sh`. LoRA at 50 and 100 steps: same command
  with `--steps 50|100 --output runs/pilot_cpu/lora_s50|lora_s100`.

Exact match % (mean token ED):

| Model | Support (10) | Held-out 2014 (50) | General 2019 (200) |
|---|---|---|---|
| Base | 0.0 (2.10) | 76.0 (0.82) | 67.5 (0.94) |
| Full, 200 steps | 100.0 (0) | 62.0 (0.88) | 63.5 (1.03) |
| Decoder, 200 steps | 100.0 (0) | 70.0 (0.92) | 61.5 (1.05) |
| LoRA, 200 steps | 100.0 (0) | 66.0 (0.84) | 62.0 (1.01) |
| LoRA, 100 steps | 100.0 (0) | 66.0 (0.88) | 63.5 (0.96) |
| LoRA, 50 steps | 100.0 (0) | 66.0 (0.80) | 62.0 (1.06) |

Exact-match discordant pairs vs. base, fixed / broken (exact McNemar p), from `wlm.compare`:

| Model | Held-out 2014 | General 2019 |
|---|---|---|
| Full | 1 / 8 (p = 0.039) | 10 / 18 (p = 0.18) |
| Decoder | 1 / 4 (p = 0.38) | 5 / 17 (p = 0.017) |
| LoRA 200 | 0 / 5 (p = 0.063) | 6 / 17 (p = 0.035) |
| LoRA 100 | 0 / 5 (p = 0.063) | 8 / 16 (p = 0.15) |
| LoRA 50 | 0 / 5 (p = 0.063) | 5 / 16 (p = 0.027) |

Cost (200 steps, CPU):

| Method | Trainable params | Train time | Checkpoint (incl. optimizer) |
|---|---|---|---|
| Full | 6,389,041 | 84.8 s | 74.1 MiB |
| Decoder | 3,232,417 | 28.4 s | 49.8 MiB |
| LoRA r=8 | 61,440 | 25.5 s (6.8 s at 50 steps) | 0.73 MiB |

Findings:

- Every method memorizes the 10 support images (0% → 100%) and all lose accuracy on unseen
  images (held-out −6 to −14 pp, general −4 to −6 pp). Sample sizes are small; only some
  drops reach p < 0.05.
- **The new errors mirror what the support set taught.** Support corrections included
  `M→m`, `s/5→S`, `.→,`, `3→B`; the regressions are dominated by `M→m`, `s→S`, `.→,`,
  `\beta→B`. Adaptation moves the decision boundary between ambiguous glyph pairs. For one
  writer that is the intended effect; for other writers it is interference. This supports
  per-writer, switchable adapters.
- LoRA at 50 steps already fits the support set and does the same damage, so the update
  count is not the main lever in this range. **The HPC development tuning should try
  lower learning rates** (e.g. LoRA 1e-4 / 3e-4) and smaller budgets (10–25 steps).
- LoRA trains about as fast as decoder tuning on CPU (the encoder forward pass dominates)
  but stores 68× less. Full tuning is 3× slower.
- On the Mac, a 10-image LoRA adaptation for the demo takes ≈26 s (200 steps) or ≈7 s (50 steps).

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
