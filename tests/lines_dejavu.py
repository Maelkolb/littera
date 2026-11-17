from __future__ import annotations

import io
import math
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


def render(text: str, size: int = 24, margin: int = 10, slant_deg: float = 0.0, neighbours: bool = False):
    font = ImageFont.truetype(str(FONT), size)
    ascent, descent = font.getmetrics()
    line = ascent + descent
    top = margin + (line if neighbours else 0)
    width = int(math.ceil(font.getlength(text))) + 2 * margin
    height = line + 2 * margin + (2 * line if neighbours else 0)
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    draw.text((margin, top), text, font=font, fill=0)
    if neighbours:
        draw.text((margin, top - int(0.9 * line)), "ghjpqy gyp jjq " * 4, font=font, fill=0)
        draw.text((margin, top + int(0.9 * line)), "bdfhkl Tdlh bkd " * 4, font=font, fill=0)
    cuts = [(i, margin + font.getlength(text[:i])) for i in range(len(text) + 1)]
    centre_y = top + ascent - 0.25 * size
    if slant_deg:
        tangent = math.tan(math.radians(slant_deg))
        offset = abs(tangent) * height
        size_out = (int(math.ceil(width + offset)), height)
        shift = offset if tangent > 0 else 0.0
        image = image.transform(size_out, Image.Transform.AFFINE, (1, tangent, -shift - tangent * 0, 0, 1, 0),
                                resample=Image.Resampling.BILINEAR, fillcolor=255)
        cuts = [(i, x + shift - tangent * centre_y) for i, x in cuts]
    return image, cuts, centre_y


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


WORDS = ["und", "mir", "den", "Tag", "Haus", "dem", "nun", "Brief", "schrieb", "am", "in", "Stadt", "kam", "alt", "Mann", "lieb"]


def sentences(n: int, seed: int = 0, words: int = 3) -> list[str]:
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, size=int(rng.integers(2, words + 1)))) for _ in range(n)]
