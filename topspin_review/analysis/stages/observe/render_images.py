"""Render and save the images the vision agent reads."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from topspin_review.perception import imaging, metrics, pose


def save(image: Image.Image, media_dir: Path, name: str) -> str:
    path = media_dir / name
    imaging.save_png(image, path)
    return str(path)


def overview_sheet(frames: list[Image.Image], timestamps: list[float], media_dir: Path) -> str:
    sheet = imaging.contact_sheet(frames, timestamps, cols=min(4, len(frames)))
    return save(sheet, media_dir, "overview_sheet.png")


def detail_images(
    frames: list[Image.Image],
    timestamps: list[float],
    measured: dict,
    zoom_frames: list[Image.Image],
    media_dir: Path,
) -> list[str]:
    paths: list[str] = []
    if frames:
        sheet = imaging.contact_sheet(frames, timestamps, cols=min(4, len(frames)))
        paths.append(save(sheet, media_dir, "detail_sheet.png"))
    if len(frames) >= 2:
        motion_map = metrics.motion_map(frames)
        if motion_map is not None:
            paths.append(save(motion_map, media_dir, "motion_map.png"))
    if measured.get("pose") and frames:
        annotated = pose.overlay(frames, measured["pose"])
        if annotated:
            pose_sheet = imaging.contact_sheet(annotated, timestamps, cols=min(4, len(annotated)))
            paths.append(save(pose_sheet, media_dir, "pose_sheet.png"))
    for i, zoom in enumerate(zoom_frames):
        paths.append(save(zoom, media_dir, f"zoom_{i}.png"))
    return paths


def still_image(frame: Image.Image, media_dir: Path) -> str:
    return save(frame, media_dir, "still.png")
