"""Manifest I/O, conservative LaTeX tokenization, and image preprocessing."""
import hashlib
import json
from pathlib import Path
import re

TOKEN_RE = re.compile(r"\\[A-Za-z]+|\\.|[^\s]")


def tokens(latex):
    text = latex.strip()
    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2]
    elif text.startswith("$") and text.endswith("$"):
        text = text[1:-1]
    return TOKEN_RE.findall(text)


def canonical(latex):
    # Only lexical whitespace and outer math delimiters are normalized.
    return " ".join(tokens(latex))


def read_rows(path):
    path = Path(path)
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "image", "latex", "writer_id", "dataset", "polarity"):
            if key not in row:
                raise ValueError(f"{path}:{line_number}: missing {key}")
        if row["polarity"] not in ("dark_on_light", "light_on_dark"):
            raise ValueError(f"{path}:{line_number}: invalid polarity")
        image = Path(row["image"])
        row["image"] = str((path.parent / image).resolve() if not image.is_absolute() else image)
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_tensor(row, augment=False):
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageOps, ImageFilter
    with Image.open(row["image"]) as raw:
        image = ImageOps.exif_transpose(raw).convert("L")
    # Augment in the familiar black-ink / white-paper representation.
    if row["polarity"] == "light_on_dark":
        image = ImageOps.invert(image)
    if augment:
        if np.random.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(float(np.random.uniform(0.1, 0.8))))
        pixels = np.asarray(image, dtype=np.float32)
        height, width = pixels.shape
        if np.random.random() < 0.35:
            # Mild illumination gradient; leave strokes and labels intact.
            shading = np.linspace(float(np.random.uniform(0.75, 1)), 1, width)
            pixels *= shading[None, :]
        if np.random.random() < 0.35:
            # Expand the canvas before perspective changes to avoid cropping ink.
            pixels = cv2.copyMakeBorder(pixels, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255)
            height, width = pixels.shape
            src = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
            jitter = np.random.uniform(-0.02, 0.02, (4, 2)) * [width, height]
            dst = np.float32(src + jitter)
            pixels = cv2.warpPerspective(pixels, cv2.getPerspectiveTransform(src, dst),
                                         (width, height), borderValue=255)
        image = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
    # CoMER expects light strokes on dark background. Keep aspect ratio.
    image = ImageOps.invert(image)
    width, height = image.size
    ratio = min(1.0, 1024 / width, 256 / height)
    if ratio < 1:
        image = image.resize((max(1, round(width*ratio)), max(1, round(height*ratio))),
                             Image.Resampling.BILINEAR)
    # Padding preserves geometry even for an extreme aspect ratio.
    canvas = Image.new("L", (max(16, image.width), max(16, image.height)), 0)
    canvas.paste(image, (0, 0))
    array = np.asarray(canvas, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array)[None, :, :]


def make_batch(rows, device, augment=False, labels=True):
    import torch
    images = [image_tensor(row, augment) for row in rows]
    height, width = max(x.shape[1] for x in images), max(x.shape[2] for x in images)
    batch = torch.zeros(len(images), 1, height, width)
    mask = torch.ones(len(images), height, width, dtype=torch.bool)
    for i, image in enumerate(images):
        h, w = image.shape[-2:]
        batch[i, :, :h, :w] = image
        mask[i, :h, :w] = False
    indices = None
    if labels:
        from comer.datamodule import vocab
        indices = [vocab.words2indices(tokens(row["latex"])) for row in rows]
    return batch.to(device), mask.to(device), indices
