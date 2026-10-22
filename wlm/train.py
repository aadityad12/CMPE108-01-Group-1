"""Single-GPU/CPU fine-tuning with fixed update budgets and resumable checkpoints."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .data import file_sha256, make_batch, read_rows, tokens
from .model import choose_device, configure, cpu_state, load_model, training_mode


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def validation_loss(model, rows, device, batch_size):
    from comer.utils.utils import ce_loss, to_bi_tgt_out
    from comer.datamodule import vocab
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(rows), batch_size):
        images, mask, labels = make_batch(rows[start:start+batch_size], device)
        target, output = to_bi_tgt_out(labels, device)
        loss = ce_loss(model(images, mask, target), output, reduction="sum")
        total += float(loss)
        count += int((output != vocab.PAD_IDX).sum())
    return total / max(1, count)


def run_training(base, manifest, output, method="decoder", steps=200, batch_size=4,
                 lr=None, seed=0, augment=False, val=None, device="auto", rank=8,
                 lora_alpha=16, save_every=50, resume=None):
    if steps < 1 or batch_size < 1 or save_every < 1:
        raise ValueError("steps, batch-size and save-every must be positive")
    device = choose_device(str(device))
    seed_all(seed)
    rows = read_rows(manifest)
    validation = read_rows(val) if val else None
    writers = {row["writer_id"] for row in rows}
    if validation is None and (len(writers) != 1 or None in writers):
        raise ValueError("Personalize on one known writer at a time. Domain training requires --val.")
    model, base_raw = load_model(base, device)
    if base_raw["kind"] == "lora":
        raise ValueError("Start each adaptation from a shared full checkpoint, not another adapter.")
    from comer.datamodule import vocab
    for row in rows + (validation or []):
        unknown = set(tokens(row["latex"])) - set(vocab.word2idx)
        if unknown or not tokens(row["latex"]) or len(tokens(row["latex"])) > 198:
            raise ValueError(f"Audit training label first: {row['id']}; unknown={unknown}")
    targets = configure(model, method, rank, lora_alpha)
    lr = lr if lr is not None else {"full": 1e-5, "decoder": 5e-5, "lora": 1e-3}[method]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "model.pt"
    train_config = {"base": str(Path(base).resolve()), "manifest": str(Path(manifest).resolve()),
                    "manifest_sha256": file_sha256(manifest), "method": method, "steps": steps,
                    "batch_size": batch_size, "lr": lr, "seed": seed, "augment": augment,
                    "rank": rank, "lora_alpha": lora_alpha,
                    "val": str(Path(val).resolve()) if val else None,
                    "val_sha256": file_sha256(val) if val else None}
    meta = {"method": method, "k": len(rows) if not validation else None,
            "writer_id": next(iter(writers)) if len(writers) == 1 else None,
            "seed": seed, "trainable_parameters": sum(p.numel() for p in trainable),
            "total_parameters": sum(p.numel() for p in model.parameters()), "targets": targets,
            "bn_running_statistics": "frozen", "base_sha256": file_sha256(base),
            "train_config": train_config}
    (folder / "config.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "train_config"}, indent=2), flush=True)
    start_step, best, elapsed_before = 0, float("inf"), 0.0
    if resume:
        old_model, old = load_model(resume, device, base_override=base)
        old_config = old["meta"]["train_config"]
        for key in train_config:
            if key != "steps" and old_config.get(key) != train_config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(old_model.state_dict(), strict=True)
        del old_model
        optimizer.load_state_dict(old["optimizer_state"])
        restore_rng(old["rng"])
        start_step, best = old["step"], old["best_val_loss"]
        elapsed_before = old["meta"].get("training_seconds", 0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    def save(step, path):
        meta["training_seconds"] = elapsed_before + time.monotonic() - started
        meta["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        raw = {"format": "wlm-1", "kind": "lora" if method == "lora" else "full",
               "model_config": base_raw["model_config"], "decode": base_raw["decode"],
               "meta": meta, "step": step, "optimizer_state": optimizer.state_dict(),
               "rng": rng_state(), "best_val_loss": best}
        if method == "lora":
            names = {name for name, p in model.named_parameters() if p.requires_grad}
            raw.update(adapter_state={name: value for name, value in cpu_state(model).items() if name in names},
                       base_checkpoint=os.path.relpath(Path(base).resolve(), Path(path).parent),
                       base_sha256=meta["base_sha256"], rank=rank, lora_alpha=lora_alpha)
        else:
            raw["state_dict"] = cpu_state(model)
        temporary = Path(str(path) + ".tmp")
        torch.save(raw, temporary)
        os.replace(temporary, path)

    log_mode = "a" if resume else "w"
    with (folder / "loss.jsonl").open(log_mode) as log:
        for step in range(start_step, steps):
            training_mode(model, method)
            # Step-indexed sampling is reproducible when a job is resumed.
            indices = random.Random(seed * 1000003 + step).choices(range(len(rows)), k=batch_size)
            examples = [rows[i] for i in indices]
            images, mask, labels = make_batch(examples, device, augment)
            from comer.utils.utils import ce_loss, to_bi_tgt_out
            target, output_ids = to_bi_tgt_out(labels, device)
            optimizer.zero_grad(set_to_none=True)
            loss = ce_loss(model(images, mask, target), output_ids)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            record = {"step": step + 1, "train_loss": float(loss.detach())}
            if (step + 1) % save_every == 0 or step + 1 == steps:
                if validation:
                    score = validation_loss(model, validation, device, batch_size)
                    record["val_loss"] = score
                    if score < best:
                        best = score
                        save(step + 1, folder / "best.pt")
                save(step + 1, checkpoint)
                print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
    return str(folder / "best.pt" if validation else checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--method", choices=["full", "decoder", "lora"], default="decoder")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume")
    args = parser.parse_args()
    print("Saved:", run_training(**vars(args)))


if __name__ == "__main__":
    main()
