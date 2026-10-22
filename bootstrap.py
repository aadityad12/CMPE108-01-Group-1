"""Fetch the pinned official CoMER source; remove unused legacy trainer imports.

Run from the starter-project root. No third-party model code is bundled here.
The network architecture, vocabulary and beam search stay unchanged.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

COMMIT = "ee15cdbe02467fd29f386fd554b4bc2185110fcc"
ROOT = Path(__file__).resolve().parent

COMPAT = '''"""Minimal device property used by CoMER's model modules."""
import torch
from torch import nn

class LightningModule(nn.Module):
    @property
    def device(self):
        for value in self.parameters():
            return value.device
        for value in self.buffers():
            return value.device
        return torch.device("cpu")

class Metric(nn.Module):
    # The legacy ExpRateRecorder is imported but our runner does not use it.
    def __init__(self, **kwargs):
        super().__init__()
    def add_state(self, name, default, **kwargs):
        self.register_buffer(name, default)
'''


def patch(repo):
    for relative in ["model/comer.py", "model/encoder.py", "model/pos_enc.py",
                     "utils/generation_utils.py"]:
        path = repo / "comer" / relative
        source = path.read_text()
        old, new = "import pytorch_lightning as pl", "from comer import _compat as pl"
        if old in source:
            assert source.count(old) == 1, path
            path.write_text(source.replace(old, new))
        elif new not in source:
            raise RuntimeError(f"Unexpected upstream file: {path}")
    path = repo / "comer/utils/utils.py"
    source = path.read_text()
    path.write_text(source.replace("from torchmetrics import Metric",
                                   "from comer._compat import Metric"))
    (repo / "comer/datamodule/__init__.py").write_text(
        'from .vocab import vocab\nvocab_size = len(vocab)\n')
    (repo / "comer/_compat.py").write_text(COMPAT)
    diff = subprocess.check_output(["git", "-C", str(repo), "diff"], text=True)
    (ROOT / "upstream-patch.diff").write_text(diff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing official checkout (optional).")
    args = parser.parse_args()
    repo = ROOT / "third_party/CoMER"
    if args.source:
        # Useful for working offline after obtaining the official checkout.
        import shutil
        if repo.exists():
            raise SystemExit("Destination exists; omit --source to reuse it.")
        repo.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(args.source, repo)
    elif not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/Green-Wood/CoMER.git",
                        str(repo)], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(repo), "checkout", COMMIT], check=True)
    patch(repo)
    import sys
    sys.path.insert(0, str(repo))
    from wlm.model import load_official, save_base
    original = next((repo / "lightning_logs/version_0/checkpoints").glob("*.ckpt"))
    model, config, decode = load_official(original)
    target = ROOT / "checkpoints/base.pt"
    target.parent.mkdir(exist_ok=True)
    save_base(target, model, config, decode, {
        "method": "base", "k": 0, "upstream_commit": COMMIT,
        "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    })
    print(f"Ready: {target}\nPinned source: {COMMIT}\nModel parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
