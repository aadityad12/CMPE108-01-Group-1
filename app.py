"""Local Gradio demo: upload labeled images, adapt, then recognize a new image."""
import argparse
from pathlib import Path
import uuid
import gradio as gr
from wlm.data import write_rows
from wlm.model import choose_device, load_model
from wlm.predict import infer
from wlm.train import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="checkpoints/base.pt")
    parser.add_argument("--method", choices=["full", "decoder", "lora"], default="lora")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    device = choose_device(args.device)
    base = str(Path(args.base).resolve())

    def adapt(files, labels):
        if not files:
            raise gr.Error("Upload your labeled expression images first.")
        by_name = {Path(path).name: path for path in files}
        if len(by_name) != len(files):
            raise gr.Error("Use different filenames for each expression.")
        pairs = {}
        for number, line in enumerate(labels.splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2 or parts[0] not in by_name or not parts[1].strip():
                raise gr.Error(f"Line {number}: use filename, one TAB, then LaTeX.")
            if parts[0] in pairs:
                raise gr.Error(f"Duplicate label: {parts[0]}")
            pairs[parts[0]] = parts[1]
        if set(pairs) != set(by_name):
            raise gr.Error("Provide exactly one label for every image.")
        if not 10 <= len(pairs) <= 100:
            raise gr.Error("Use 10 to 100 labeled expression images.")
        session = uuid.uuid4().hex
        folder = Path("demo_sessions") / session
        rows = [{"id": f"{session}-{name}", "image": str(Path(by_name[name]).resolve()),
                 "latex": latex, "writer_id": session, "dataset": "demo",
                 "polarity": "dark_on_light"} for name, latex in pairs.items()]
        manifest = folder / "adapt.jsonl"
        write_rows(manifest, rows)
        try:
            checkpoint = run_training(base, manifest, folder / "run", method=args.method,
                                      steps=args.steps, lr=args.lr, device=str(device))
        except (KeyError, ValueError) as error:
            raise gr.Error(f"Check your labels and the supported vocabulary: {error}") from error
        return checkpoint, f"Adapted {args.method} model using {len(rows)} labeled examples.", checkpoint

    def recognize(image, checkpoint):
        if not image:
            raise gr.Error("Upload a cropped image containing one expression.")
        model, raw = load_model(checkpoint or base, device, base_override=base)
        prediction = infer(model, raw, [{"image": image, "polarity": "dark_on_light"}], device)[0]
        return prediction, "$$\n" + prediction + "\n$$"

    with gr.Blocks(title="Write Like Me") as demo:
        gr.Markdown("# Write Like Me\nPersonalize a math recognizer with 10–100 labeled images of your handwriting.")
        state = gr.State(None)
        with gr.Tab("1 · Adapt to my handwriting"):
            files = gr.File(label="Cropped expression photos", file_count="multiple", type="filepath", file_types=["image"])
            labels = gr.Textbox(label="One label per line: filename<TAB>LaTeX", lines=8,
                                placeholder="math001.jpg\tx ^ { 2 } + 1")
            button = gr.Button("Adapt recognizer")
            status = gr.Textbox(label="Status")
            download = gr.File(label="Download personalized checkpoint")
            button.click(adapt, [files, labels], [state, status, download], concurrency_limit=1)
        with gr.Tab("2 · Recognize a new expression"):
            image = gr.Image(type="filepath", label="New expression photo")
            recognize_button = gr.Button("Recognize")
            latex = gr.Textbox(label="LaTeX")
            rendered = gr.Markdown()
            recognize_button.click(recognize, [image, state], [latex, rendered], concurrency_limit=1)
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
