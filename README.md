# Write Like Me — executable starter

Start with **GUIDE.html**. It explains every phase, why it matters, commands to run,
expected outputs, experiment design, troubleshooting, and the final report.

This package contains original experiment code. It downloads the pinned official
CoMER implementation and checkpoint when you run `python bootstrap.py`. Dataset
archives and pretrained weights are not redistributed in this ZIP.

Run commands from this directory. Core setup (Linux CPU example):

```bash
conda create -n wlm python=3.10 -y
conda activate wlm
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-core.txt
python bootstrap.py
python -m wlm.predict image --checkpoint checkpoints/base.pt --image third_party/CoMER/example/UN19_1041_em_595.bmp --polarity light_on_dark --device cpu
```

See **VALIDATION.html** for what was actually tested while preparing this package.
The supplied learning rates and update budgets are starting configurations to tune
on development writers. They are not claimed to be optimal.

Source pin: `ee15cdbe02467fd29f386fd554b4bc2185110fcc`.
