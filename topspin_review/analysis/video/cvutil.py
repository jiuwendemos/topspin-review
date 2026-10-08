"""Small dependency-free computer-vision helpers (numpy + PIL only)."""

from __future__ import annotations

import numpy as np


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an ``H x W x 3`` float array in 0..1 (RGB) to HSV in 0..1."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc > 0, delta / np.maximum(maxc, 1e-6), 0.0)

    nonzero = delta > 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        rc = np.where(nonzero, (maxc - r) / np.maximum(delta, 1e-6), 0.0)
        gc = np.where(nonzero, (maxc - g) / np.maximum(delta, 1e-6), 0.0)
        bc = np.where(nonzero, (maxc - b) / np.maximum(delta, 1e-6), 0.0)
    h = np.where(r == maxc, bc - gc, 0.0)
    h = np.where(g == maxc, 2.0 + rc - bc, h)
    h = np.where(b == maxc, 4.0 + gc - rc, h)
    h = np.where(nonzero, ((h / 6.0) % 1.0), 0.0)
    return np.stack([h, s, v], axis=-1)


def _flood(mask: np.ndarray, seed: tuple[int, int], seen: np.ndarray) -> dict:
    h, w = mask.shape
    y0, x0 = seed
    stack = [(y0, x0)]
    seen[y0, x0] = True
    keep = np.zeros_like(mask, dtype=bool)
    area = 0
    sx = sy = 0
    minx = maxx = x0
    miny = maxy = y0
    while stack:
        y, x = stack.pop()
        area += 1
        sx += x
        sy += y
        keep[y, x] = True
        if x < minx:
            minx = x
        if x > maxx:
            maxx = x
        if y < miny:
            miny = y
        if y > maxy:
            maxy = y
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                stack.append((ny, nx))
    return {
        "area": area,
        "bbox": (minx, miny, maxx, maxy),
        "cx": round(sx / area / w, 3),
        "cy": round(sy / area / h, 3),
        "mask": keep,
    }


def components(mask: np.ndarray) -> list[dict]:
    """4-connected components of a boolean mask, with area/bbox/centroid/mask."""
    seen = np.zeros_like(mask, dtype=bool)
    out: list[dict] = []
    for y, x in zip(*np.where(mask)):
        if not seen[y, x]:
            out.append(_flood(mask, (int(y), int(x)), seen))
    return out


def largest(mask: np.ndarray, min_area: int = 1) -> np.ndarray | None:
    """Boolean mask of the single largest component (or ``None``)."""
    comps = components(mask)
    if not comps:
        return None
    best = max(comps, key=lambda c: c["area"])
    return best["mask"] if best["area"] >= min_area else None
