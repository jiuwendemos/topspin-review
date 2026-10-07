"""Image encoding and compositing helpers for the vision model."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path

from PIL import Image, ImageDraw


def to_data_url(img: Image.Image, max_width: int = 768, quality: int = 85) -> str:
    """Convert one image to a JPEG data URL, downscaled to ``max_width``."""
    width, height = img.size
    if width > max_width:
        scale = max_width / width
        img = img.resize((max_width, max(1, int(height * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def frames_to_data_urls(frames: list[Image.Image], max_width: int = 768, quality: int = 85) -> list[str]:
    """Convert frames to JPEG data URLs, downscaled to ``max_width``."""
    return [to_data_url(img, max_width=max_width, quality=quality) for img in frames]


def contact_sheet(
    frames: list[Image.Image],
    timestamps: list[float],
    cols: int = 4,
    cell: tuple[int, int] = (384, 216),
) -> Image.Image:
    """Grid of frames labelled with their timestamps (compact overview image)."""
    if not frames:
        raise ValueError("no frames")
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * cell[0], rows * cell[1]), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        thumb = frame.resize(cell)
        sheet.paste(thumb, (c * cell[0], r * cell[1]))
        label = f"#{i + 1} t={timestamps[i]:.2f}s" if i < len(timestamps) else f"#{i + 1}"
        draw.rectangle([c * cell[0], r * cell[1], c * cell[0] + 108, r * cell[1] + 16], fill=(0, 0, 0))
        draw.text((c * cell[0] + 4, r * cell[1] + 3), label, fill=(255, 235, 60))
    return sheet


def save_png(img: Image.Image, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
