"""Load official weights, choose trainable parameters, and save small adapters."""
import inspect
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/CoMER"))


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for parameter in base.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return self.base(x) + update * (self.alpha / self.rank)


def configure(model, method, rank=8, alpha=16):
    if method not in ("full", "decoder", "lora"):
        raise ValueError(method)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = method == "full" or (method == "decoder" and name.startswith("decoder."))
    targets = []
    if method == "lora":
        # CoMER attention uses raw projection tensors, not standard q_proj modules.
        # Target the real, called decoder FFN Linear modules instead.
        for name, module in list(model.named_modules()):
            if name.startswith("decoder.model.layers.") and name.rsplit(".", 1)[-1] in ("linear1", "linear2"):
                if not isinstance(module, nn.Linear):
                    raise TypeError(f"Unexpected LoRA target: {name}")
                parent_name, child = name.rsplit(".", 1)
                adapter = LoRALinear(module, rank, alpha).to(device=module.weight.device,
                                                           dtype=module.weight.dtype)
                setattr(model.get_submodule(parent_name), child, adapter)
                targets.append(name)
        if len(targets) != 6:
            raise RuntimeError(f"Expected six decoder FFN targets, found {targets}")
    return targets


def training_mode(model, method):
    model.train()
    if method in ("decoder", "lora"):
        model.encoder.eval()
    # Freeze running statistics for the tiny writer datasets, including full tuning.
    # BN affine parameters still train in full mode.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def load_official(path):
    from comer.model.comer import CoMER
    # The legacy checkpoint has ModelCheckpoint as a dictionary key. This inert
    # allowlisted name enables restricted tensor loading without installing Lightning.
    marker = type("ModelCheckpoint", (), {})
    marker.__module__ = "pytorch_lightning.callbacks.model_checkpoint"
    with torch.serialization.safe_globals([marker]):
        raw = torch.load(path, map_location="cpu", weights_only=True)
    hparams = raw["hyper_parameters"]
    config = {name: hparams[name] for name in inspect.signature(CoMER.__init__).parameters if name != "self"}
    decode = {name: hparams[name] for name in ("beam_size", "max_len", "alpha", "early_stopping", "temperature")}
    model = CoMER(**config)
    state = {name.removeprefix("comer_model."): value for name, value in raw["state_dict"].items()
             if name.startswith("comer_model.")}
    model.load_state_dict(state, strict=True)
    return model, config, decode


def cpu_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_base(path, model, config, decode, meta):
    torch.save({"format": "wlm-1", "kind": "full", "model_config": config,
                "decode": decode, "state_dict": cpu_state(model), "meta": meta}, path)


def load_model(path, device="cpu", base_override=None):
    from comer.model.comer import CoMER
    from .data import file_sha256
    path = Path(path).resolve()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if raw.get("format") != "wlm-1":
        raise ValueError("Run bootstrap.py first; use the converted base.pt checkpoint.")
    if raw["kind"] == "lora":
        base = Path(base_override or raw["base_checkpoint"])
        if not base.is_absolute():
            base = (path.parent / base).resolve()
        if file_sha256(base) != raw["base_sha256"]:
            raise ValueError("Adapter and base hashes differ. Select the exact training base.")
        model, _ = load_model(base, "cpu")
        configure(model, "lora", raw["rank"], raw["lora_alpha"])
        expected = {n for n, p in model.named_parameters() if p.requires_grad}
        if set(raw["adapter_state"]) != expected:
            raise ValueError("Adapter keys do not match this architecture.")
        model.load_state_dict(raw["adapter_state"], strict=False)
    else:
        model = CoMER(**raw["model_config"])
        model.load_state_dict(raw["state_dict"], strict=True)
    model.to(device)
    return model, raw


def choose_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Request a Slurm GPU allocation or use --device cpu.")
    return device
