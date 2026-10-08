"""Image compositing helpers for the vision model (contact sheets, PNG saving)."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw


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
