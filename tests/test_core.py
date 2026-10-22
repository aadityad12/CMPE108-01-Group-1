"""Run after bootstrap: python -m unittest discover -s tests -v"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from PIL import Image
from wlm.data import canonical, image_tensor, read_rows, write_rows
from wlm.metrics import edit_distance, score_example
from wlm.model import configure, load_model, training_mode
from wlm.prepare import writer_splits
from wlm.train import run_training

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_tokens_and_metrics(self):
        self.assertEqual(canonical(r"$\frac{x}{2}$"), r"\frac { x } { 2 }")
        self.assertEqual(edit_distance(["x", "+", "1"], ["x", "+", "2"]), 1)
        self.assertEqual(score_example("x ^ { 2 }", "x ^ { 2 }")["ed"], 0)
        self.assertEqual(score_example("a", "a + b")["ed"], 2)

    def test_input_polarity(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "image.png"
            image = np.full((20, 20), 255, np.uint8)
            image[8:12, 8:12] = 0
            Image.fromarray(image).save(path)
            tensor = image_tensor({"image": str(path), "polarity": "dark_on_light"})
            self.assertEqual(float(tensor[0, 0, 0]), 0.0)
            self.assertEqual(float(tensor[0, 9, 9]), 1.0)

    def test_nested_writer_splits(self):
        import argparse
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            rows = [{"id": str(i), "image": str(folder / f"{i}.png"), "latex": f"x + {i}",
                     "writer_id": "writer01", "dataset": "unit", "polarity": "dark_on_light"}
                    for i in range(160)]
            manifest = folder / "rows.jsonl"
            write_rows(manifest, rows)
            writer_splits(argparse.Namespace(manifest=str(manifest), writers=None, seed=42,
                                             output=str(folder / "splits"), test_size=50))
            report = json.loads((folder / "splits/split_report.json").read_text())
            path = Path(report[0]["folder"])
            sets = [{r["id"] for r in read_rows(path / f"adapt_{k}.jsonl")} for k in (10, 25, 50, 100)]
            self.assertTrue(all(a < b for a, b in zip(sets, sets[1:])))
            self.assertFalse(sets[-1] & {r["id"] for r in read_rows(path / "test.jsonl")})

    def test_lora_identity_and_gradient(self):
        from comer.utils.utils import ce_loss, to_bi_tgt_out
        from wlm.data import make_batch
        model, _ = load_model(ROOT / "checkpoints/base.pt")
        row = {"image": str(ROOT / "third_party/CoMER/example/UN19_1041_em_595.bmp"),
               "polarity": "light_on_dark", "latex": "x ^ { 2 }"}
        images, mask, labels = make_batch([row], "cpu")
        target, truth = to_bi_tgt_out(labels, torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            reference = model(images, mask, target)
        encoder_before = model.encoder.model.conv1.weight.detach().clone()
        targets = configure(model, "lora")
        self.assertEqual(len(targets), 6)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 61440)
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(reference, model(images, mask, target), atol=1e-6))
        training_mode(model, "lora")
        loss = ce_loss(model(images, mask, target), truth)
        loss.backward()
        b_grad = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                     if "lora_B" in n and p.grad is not None)
        self.assertGreater(b_grad, 0)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(torch.equal(encoder_before, model.encoder.model.conv1.weight))

    def test_decoder_freeze(self):
        model, _ = load_model(ROOT / "checkpoints/base.pt")
        configure(model, "decoder")
        training_mode(model, "decoder")
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)))

    def test_adapter_save_reload_and_resume(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            image = ROOT / "third_party/CoMER/example/UN19_1041_em_595.bmp"
            manifest = folder / "adapt.jsonl"
            write_rows(manifest, [{"id": "integration", "image": str(image), "latex": "x ^ { 2 }",
                                   "writer_id": "integration", "dataset": "unit", "polarity": "light_on_dark"}])
            checkpoint = run_training(ROOT / "checkpoints/base.pt", manifest, folder / "run",
                                      method="lora", steps=1, batch_size=1, device="cpu", save_every=1)
            _, raw = load_model(checkpoint)
            self.assertEqual(raw["step"], 1)
            self.assertEqual(len(raw["adapter_state"]), 12)
            self.assertLess(Path(checkpoint).stat().st_size, 2_000_000)
            run_training(ROOT / "checkpoints/base.pt", manifest, folder / "run", method="lora",
                         steps=2, batch_size=1, device="cpu", save_every=1, resume=checkpoint)
            _, raw = load_model(checkpoint)
            self.assertEqual(raw["step"], 2)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
