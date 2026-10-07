"""Create a synthetic table-tennis clip for demos (no camera needed).

    python scripts/make_sample.py

Writes runtime/data/sample.mp4 — a side view: a player silhouette with visible stance,
lateral footwork and a forehand arm swing, a table, a net, and a rallying ball.
It is still synthetic: good enough to demonstrate the pipeline and to let the
vision model comment on stance/footwork/movement. For real technique feedback,
record a real side-on clip.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 960, 540
FPS = 20
SECONDS = 6
OUT = Path(__file__).resolve().parent.parent / "runtime" / "data" / "sample.mp4"

DARK = (45, 48, 58)
SKIN = (210, 170, 140)
FLOOR = (60, 66, 80)


def _player(d: ImageDraw.ImageDraw, x: float, facing: int, swing: float, crouch: float) -> None:
    """Draw a side-view player silhouette facing ``facing`` (+1 right, -1 left)."""
    hip_y = 300 + 14 * crouch
    shoulder_y = 195 + 10 * crouch
    head_y = shoulder_y - 22

    # legs (stance widens when crouched)
    foot_back = x - facing * (26 + 14 * crouch)
    foot_front = x + facing * (26 + 14 * crouch)
    d.line([(x, hip_y), (foot_back, 470)], fill=DARK, width=12)
    d.line([(x, hip_y), (foot_front, 470)], fill=DARK, width=12)
    # feet
    d.ellipse([foot_back - 14, 462, foot_back + 14, 478], fill=(30, 32, 40))
    d.ellipse([foot_front - 14, 462, foot_front + 14, 478], fill=(30, 32, 40))

    # torso
    d.rounded_rectangle([x - 15, shoulder_y, x + 15, hip_y], radius=10, fill=DARK)
    # head
    d.ellipse([x - 15, head_y - 15, x + 15, head_y + 15], fill=SKIN)

    # arm holding the paddle (swings)
    hand_x = x + facing * (55 + 45 * swing)
    hand_y = shoulder_y + 5 - 25 * swing
    d.line([(x + facing * 8, shoulder_y + 6), (hand_x, hand_y)], fill=DARK, width=10)
    # paddle
    d.ellipse([hand_x - 12, hand_y - 16, hand_x + 12, hand_y + 16], fill=(200, 40, 40))


def make_frame(t: float) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), (235, 236, 240))
    d = ImageDraw.Draw(img)

    # floor
    d.rectangle([0, 470, WIDTH, HEIGHT], fill=FLOOR)

    # table (side view)
    d.rectangle([470, 360, 940, 376], fill=(30, 120, 85), outline=(250, 250, 250), width=3)
    d.line([(640, 452), (640, 470)], fill=(20, 22, 30), width=8)
    d.rectangle([560, 376, 568, 470], fill=(20, 22, 30))
    d.rectangle([900, 376, 908, 470], fill=(20, 22, 30))
    # net
    d.line([(720, 316), (720, 360)], fill=(240, 240, 240), width=5)

    # foreground player (moves laterally, crouches, swings)
    px = 320 + 55 * math.sin(t * 2.0)
    crouch = 0.5 + 0.5 * math.sin(t * 2.0)
    swing = 0.5 + 0.5 * math.sin(t * 4.0)
    _player(d, px, facing=1, swing=swing, crouch=crouch)

    # opponent (smaller, far side)
    ox = 880 - 30 * math.sin(t * 2.0 + 0.5)
    _player(d, ox, facing=-1, swing=0.5 + 0.5 * math.sin(t * 4.0 + 2.0), crouch=0.4)

    # ball arcing between them
    bx = 380 + (820 - 380) * (0.5 + 0.5 * math.sin(t * 2.0))
    by = 300 - 120 * abs(math.sin(t * 4.0)) + 120
    d.ellipse([bx - 8, by - 8, bx + 8, by + 8], fill=(255, 205, 40))

    d.text((20, 20), f"t={t:.1f}s", fill=(60, 60, 70))
    return img


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = FPS * SECONDS
    writer = imageio.get_writer(str(OUT), fps=FPS, codec="libx264", quality=8, macro_block_size=None)
    try:
        for i in range(n):
            writer.append_data(np.array(make_frame(i / FPS)))
    finally:
        writer.close()
    print(f"wrote {OUT} ({n} frames, {SECONDS}s @ {FPS}fps)")


if __name__ == "__main__":
    main()
