"""Minimal HTTP API around the analysis service.

    pip install -r requirements.txt
    uvicorn topspin_review.interfaces.api:app --reload

Endpoints: ``GET /reports``, ``GET /reports/{stem}``, ``POST /analyze``.
"""

from __future__ import annotations

from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("API deps missing: pip install -r requirements.txt") from exc

from topspin_review.bootstrap import setup
from topspin_review.interfaces import service
from topspin_review.storage import store

setup()

app = FastAPI(title="Topspin Review")


class AnalyzeRequest(BaseModel):
    video_path: str
    region_box: list[float] | None = None


@app.get("/reports")
def list_reports() -> list[dict]:
    return [{"source": r.get("source"), "date": r.get("date"), "focus": r.get("focus")} for r in store.get_reports()]


@app.get("/reports/{stem}")
def get_report(stem: str) -> dict:
    report = store.find_report(stem)
    if report is None:
        raise HTTPException(status_code=404, detail="report not found")
    return report


@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> dict:
    if not Path(request.video_path).exists():
        raise HTTPException(status_code=400, detail=f"video not found: {request.video_path}")
    result = await service.analyze_video(request.video_path, region_box=service.parse_region_box(request.region_box))
    return {"report_path": result.get("report_path"), "report": result.get("report")}
