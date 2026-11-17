from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

PAPERS = [(246, 241, 228), (238, 228, 205), (226, 214, 186), (244, 244, 240), (215, 203, 178), (232, 222, 210)]
INKS = [(25, 22, 20), (62, 42, 28), (88, 60, 38), (30, 34, 62), (52, 50, 48), (105, 82, 60)]


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], scale: float) -> np.ndarray:
    height, width = shape
    small = rng.standard_normal((max(2, int(height / scale) + 2), max(2, int(width / scale) + 2))).astype(np.float32)
    field = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    return field / (np.abs(field).max() + 1e-6)


def stroke_width(mask: np.ndarray, rng: np.random.Generator, x_height: float) -> np.ndarray:
    choice = rng.random()
    if choice < 0.2:
        radius = max(1, int(round(x_height / 18)))
        return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    if choice < 0.27 and x_height > 24:
        return cv2.erode(mask, np.ones((2, 2), np.uint8))
    return mask


def compose(mask: np.ndarray, rng: np.random.Generator, x_height: float) -> np.ndarray:
    mask = stroke_width(mask, rng, x_height)
    height, width = mask.shape
    paper = np.array(PAPERS[rng.integers(len(PAPERS))], dtype=np.float32) + rng.normal(0, 6, 3)
    ink = np.array(INKS[rng.integers(len(INKS))], dtype=np.float32) + rng.normal(0, 8, 3)
    texture = smooth_noise(rng, (height, width), rng.uniform(20, 120)) * rng.uniform(4, 22)
    background = np.clip(paper[None, None, :] + texture[..., None], 0, 255)
    density = 1.0 - rng.uniform(0.0, 0.45) * (0.5 + 0.5 * smooth_noise(rng, (height, width), rng.uniform(6, 40)))
    alpha = (mask.astype(np.float32) / 255.0) * np.clip(density, 0.35, 1.0) * rng.uniform(0.75, 1.0)
    image = background * (1 - alpha[..., None]) + ink[None, None, :] * alpha[..., None]
    if rng.random() < 0.25:
        stain = np.clip(smooth_noise(rng, (height, width), rng.uniform(40, 200)), 0, 1) * rng.uniform(10, 40)
        image = image - stain[..., None] * np.array([0.6, 0.8, 1.0], dtype=np.float32)
    return np.clip(image, 0, 255)


def finish(image: np.ndarray, rng: np.random.Generator) -> Image.Image:
    sigma = rng.choice([0.0, 0.0, rng.uniform(0.3, 1.3)])
    if sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    noise = rng.uniform(0, 9)
    image = np.clip(image + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)
    picture = Image.fromarray(image, "RGB")
    if rng.random() < 0.3:
        factor = rng.uniform(0.5, 0.85)
        small = picture.resize((max(1, int(picture.width * factor)), max(1, int(picture.height * factor))), Image.BILINEAR)
        picture = small.resize(picture.size, Image.BILINEAR)
    if rng.random() < 0.15:
        picture = picture.convert("L").convert("RGB")
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=int(rng.integers(55, 96)))
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
