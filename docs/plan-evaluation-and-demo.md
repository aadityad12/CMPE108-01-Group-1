# Work plan: evaluation and demo (Aaditya)

Owner: Aaditya. Teammates: Jeriel (model, training, HPC jobs), Aidana (data, labels, splits).
See `GUIDE.html` sections 8, 11, 12 and 13 for the underlying protocol.

Goal: everything needed to turn trained checkpoints into honest numbers, plots,
error examples and a working demo, plus the evaluation parts of the report.

Status key: `[x]` done, `[ ]` to do. Each item should land as its own small commit.

## Phase 0: Local environment

- [x] Isolated Python 3.10 venv on the Mac ([setup-mac.md](setup-mac.md))
- [x] `bootstrap.py`, 6 tests passing, first prediction matches the guide

## Phase 1: Understand the metrics (reading)

- [ ] Read `wlm/metrics.py`: token edit distance, exact match, within 1/2 tokens,
      normalized edit distance, token error rate
- [ ] Read `wlm/predict.py` (`evaluate`): what goes into `predictions.jsonl` and `metrics.json`
- [ ] Read `wlm/analyze.py`: seed averaging, paired gain vs. base, bootstrap over writers,
      forgetting
- [ ] Be able to explain why "within 1 token" is not an official CROHME symbol metric

## Phase 2: Base-model reference numbers (Mac, no HPC needed)

- [x] Export the bundled CROHME 2019 test set (1,199 images) to `data/general_2019.jsonl`
- [x] Smoke evaluation on 20 images; record time per image
- [x] Full base evaluation on CROHME 2019 on CPU: 62.64% exact (exploratory; feeds
      error-analysis tooling)
- [x] Record results in [eval-log.md](eval-log.md)

Final paper numbers for the base are re-run on the HPC on the same device as the
adapted models. That is one short job, not a sweep.

## Phase 3: Error-analysis tooling

The guide asks for 20–30 manually reviewed changed cases (base vs. adapted).

- [ ] Script that pairs two `predictions.jsonl` files by id and lists improved / regressed /
      unchanged cases with edit distances
- [ ] HTML contact sheet: image, ground truth, base prediction, adapted prediction
- [ ] Error categories to tag by hand: symbol substitution (1/l, x/×), lost braces or
      structure, missing/extra symbols, layout (fractions, scripts)
- [ ] Test it on two real prediction files (e.g. base vs. beam-size-2 base) before adapted
      models exist

## Phase 4: Team writer evaluation (after Aidana's pilot data)

- [ ] Audit my own 10-expression pilot with `wlm.prepare audit`
- [ ] Base evaluation per writer on the frozen `test.jsonl` with `--as-base`
- [ ] Leakage check (`wlm.leakage --require-formula-disjoint`) for every writer folder

## Phase 5: Demo

- [ ] `pip install -r requirements-ui.txt` into the venv; `python app.py --device cpu`
- [ ] Guide's demo checklist: base recognition, successful adaptation, new-image
      recognition, missing-label error, duplicate-filename error, unsupported-token error
- [ ] Measure adaptation latency on the Mac (LoRA, 10 images)
- [ ] Decide where the presentation demo runs (Mac CPU vs. HPC with SSH tunnel)

## Phase 6: Results and figures (after Jeriel's runs)

- [ ] `python -m wlm.analyze` on the study root; check `per_run.csv`, `summary.csv`,
      `forgetting.csv`, `learning_curve.png`
- [ ] Method cost table: trainable parameters, training time, peak GPU memory, checkpoint size
- [ ] Paired image / base / adapted figure from the error-analysis tool

## Phase 7: Report sections I own

- [ ] Evaluation protocol and metric definitions
- [ ] Experimental results: tables, learning curve, forgetting, error analysis
- [ ] Demo description

## Things to coordinate with the team

- **One shared `base.pt`.** `wlm.analyze` refuses to mix results from bases with different
  hashes. Copy one converted `checkpoints/base.pt` to the HPC instead of regenerating it
  per machine, and check `shasum -a 256 checkpoints/base.pt` matches. Current Mac copy:
  `2bb8c70d67f6be86d7caf32d2544289bad19e46848603eb11c46eb80e7b9109f`.
- **Shared HPC budget.** Each sweep job also decodes the general set with beam search.
  Time that in Jeriel's pilot before scaling; if it dominates, evaluate forgetting on a
  subset of runs.
- **No data or photos in git.** `data/`, `runs/`, `results/` stay ignored; the repo is public.
