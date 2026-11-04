# Work plan: SJSU HPC runs

Owner: **Jeriel** (model, training, HPC jobs, as in `GUIDE.html` section 15).
Inputs come from Aidana (writer data) and Aaditya (manifests, base hash, analysis).
Everything here is GPU work; anything that runs fine on a laptop is in
[plan-evaluation-and-demo.md](plan-evaluation-and-demo.md).

Status key: `[x]` done, `[ ]` to do. Commit small changes (job scripts, notes, timing
numbers) as they land; never commit data, checkpoints or `runs/`.

## What the HPC produces

For every run, the folder that `wlm.train` / `wlm.predict` writes:
`config.json`, `loss.jsonl`, `writer/metrics.json`, `writer/predictions.jsonl`,
and `general/metrics.json` + `general/predictions.jsonl` where a general set is used.
These small files are what the evaluation side needs. Checkpoints stay on the HPC.

## H0 · Access and environment (no data needed; do this first)

- [ ] Log in: `ssh SJSU_ID@coe-hpc1.sjsu.edu`, then `ssh coe-hpc3` (guide section 5)
- [ ] Check what the scheduler offers: `sinfo -o "%P %G %l %D"`, `scontrol show partition gpu`.
      If the partition is not called `gpu`, fix `--partition` in both `slurm/*.sbatch`
- [ ] `git clone https://github.com/aadityad12/CMPE108-01-Group-1.git ~/wlm && cd ~/wlm`
- [ ] Conda env with the **CUDA** wheel: `conda create -n wlm python=3.10 -y`,
      `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`,
      `pip install -r requirements-core.txt -r requirements-analysis.txt`
- [ ] `python bootstrap.py`, then check the base hash. It **must** be
      `2bb8c70d67f6be86d7caf32d2544289bad19e46848603eb11c46eb80e7b9109f`
      (`sha256sum checkpoints/base.pt`). If it differs, copy the Mac `base.pt` over with `scp`
      instead; `wlm.analyze` refuses to mix bases
- [ ] `python -m unittest discover -s tests -v` (9 tests)
- [ ] Interactive GPU check (guide section 5): `srun --partition=gpu --gres=gpu:1 ... --pty bash`,
      `nvidia-smi`, `python -c "import torch; print(torch.cuda.is_available())"`.
      Record the GPU model; it goes in the report
- [ ] `python -m pip freeze > env/hpc-gpu-lock.txt` and commit it

## H1 · GPU timing pilot (no team data needed)

Uses the same 10-image pilot support set as the Mac CPU pilot, so the numbers are
directly comparable. Build the pilot manifests exactly as in `docs/eval-log.md`
(2014 screen, then `scripts/cpu_pilot.sh` builds the held-out sets), or `scp` the
`data/pilot/` folder from Aaditya's Mac.

- [ ] Final base number on the general set, on GPU:
      `python -m wlm.prepare upstream --split 2019 --output data/general_2019.jsonl`, then
      `python -m wlm.predict evaluate --checkpoint checkpoints/base.pt --manifest data/general_2019.jsonl --output runs/study/base/general --kind general --as-base --device cuda`.
      Expected: 62.64% exact (the Mac CPU result). Record the decode time
- [ ] One 200-step run per method on `data/pilot/support_10.jsonl` with `--device cuda`.
      Record wall time, `peak_cuda_bytes` from `config.json`, and `du -h model.pt`
- [ ] Time decoding the 1,199-image general set per run. **This decides whether every sweep
      job can afford a forgetting evaluation** (on CPU it is ≈26 min)
- [ ] Write the per-run time *t* and memory into `docs/eval-log.md` (HPC section)

## H2 · Development tuning (needs development-writer data from Aidana)

The guide's learning rates are starting points. Tune on **development** writers only,
never on the three evaluation writers' test files.

- [ ] For each method, try 3 learning rates and 3 step budgets at k=10 and k=100 on one
      development writer. Pick by held-out development accuracy. The CPU pilot
      (`docs/eval-log.md`, 2026-11-03) found that the defaults memorize 10 images within
      50 steps and hurt unseen images, so search **downward**: LoRA {1e-4, 3e-4, 1e-3},
      decoder {1e-5, 2e-5, 5e-5}, full {2e-6, 5e-6, 1e-5}; steps {25, 50, 200}.
      That is 27 short runs per k, a few GPU-minutes each
- [ ] Freeze the choice in a short note (`docs/hpc-settings.md`): method, lr, steps, batch,
      augment on/off, LoRA rank. **Do not change it after the final sweep starts**

## H3 · Base evaluation per evaluation writer (needs frozen splits from Aidana)

- [ ] Get `data/writers/` (the frozen splits) from Aidana. Do not regenerate it on the HPC
- [ ] For each writer folder: `python -m wlm.leakage --left .../adapt_100.jsonl --right .../test.jsonl --require-formula-disjoint`
- [ ] For each writer folder: `python -m wlm.predict evaluate --checkpoint checkpoints/base.pt --manifest data/writers/FOLDER/test.jsonl --output runs/study/base/FOLDER --kind writer --as-base --device cuda`

## H4 · The personalization sweep

Size: writers × 3 methods × 4 budgets × seeds. With 3 team writers: 36 runs per seed,
108 for three seeds. Total GPU time ≈ runs × *t* (from H1).

- [ ] `python -m wlm.sweep plan --splits data/writers --base checkpoints/base.pt --output runs/study/personal --jobs jobs.jsonl --steps STEPS --seeds 0 --general data/general_2019.jsonl`
      (drop `--general` if H1 showed general decoding is too slow, and run forgetting
      only for k=100 afterwards)
- [ ] Run index 0 interactively first: `python -m wlm.sweep run --jobs jobs.jsonl --index 0 --device cuda`
- [ ] `mkdir -p logs && sbatch --array=0-LAST%2 slurm/sweep.sbatch jobs.jsonl`
- [ ] Check `squeue -u $USER`, then `sacct -j JOB --format=JobID,State,Elapsed,MaxRSS`; rerun failed indices
- [ ] Seed 0 complete → send results (below) → then seeds 1,2 with a new plan file
- [ ] `python -m wlm.analyze --root runs/study --output results/study` must run without
      errors (it refuses unbalanced seeds or cohorts; that means a job is missing)

## H5 · Optional, only if H4 is done

- [ ] Phone-domain base (guide section 9) if HME100K access arrived
- [ ] Cross-writer control: adapt to writer B, test on writer A (guide section 12)

## Hand-back after each stage

Copy back only the small files (from the laptop):

```bash
rsync -av --include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' \
  --include='*.png' --exclude='*' SJSU_ID@coe-hpc1.sjsu.edu:~/wlm/runs/study/ runs/study/
```

Also send: GPU model, the `sacct` lines for one run per method, and `du -h` of one
checkpoint per method. Those fill the method-cost table.

## What blocks what

| HPC stage | Blocked on | Can start |
|---|---|---|
| H0, H1 | nothing | now |
| H2 | development-writer data (Aidana) | after pilot data |
| H3, H4 | frozen evaluation-writer splits (Aidana) | after split freeze |
| Analysis, figures | H3 + H4 results (this doc → Aaditya) | after seed 0 |
