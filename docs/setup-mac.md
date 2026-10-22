# Local setup on macOS (CPU)

Verified 2026-10-21 on macOS 26.6 (Apple Silicon), Python 3.10.17, torch 2.5.1.
Exact package versions: [`env/mac-cpu-lock.txt`](../env/mac-cpu-lock.txt).

The Mac is for setup checks, single predictions, small evaluations, and the
Gradio demo. Training runs belong on the HPC (see `GUIDE.html`, section 5).

## Why a virtual environment

torch 2.5.1 has no wheels for Python 3.13+. Instead of changing the system
Python, everything lives in a project-local `.venv/` built on Python 3.10.
Nothing outside this folder is modified; deleting `.venv/` undoes it.
`.venv/` is git-ignored.

## Steps

Requires Homebrew `python@3.10` and `uv` (`brew install python@3.10 uv`).

```bash
cd write_like_me
uv venv .venv --python /opt/homebrew/bin/python3.10
uv pip install --python .venv/bin/python -r env/mac-cpu-lock.txt
source .venv/bin/activate        # `deactivate` to leave
python bootstrap.py              # downloads pinned CoMER + weights (~1 min)
python -m unittest discover -s tests -v
```

Expected: `checkpoints/base.pt`, `Model parameters: 6,389,041`, and 6 passing tests.

## First prediction

```bash
python -m wlm.predict image --checkpoint checkpoints/base.pt \
  --image third_party/CoMER/example/UN19_1041_em_595.bmp \
  --polarity light_on_dark --device cpu
```

Expected (and observed):

```
x ^ { 2 } = \sum \limits _ { a = 1 } ^ { 3 } x _ { a } ^ { 2 }
```

About 6 s wall time on CPU, including model loading.

## Notes

- Always pass `--device cpu` on the Mac; there is no CUDA.
- `bootstrap.py` writes `upstream-patch.diff`, which is committed so the exact
  changes to the third-party source are reviewable.
