"""Minimal HTTP API around the analysis service.

    pip install -r requirements.txt
    uvicorn topspin_review.api:app --reload

Endpoints: ``GET /reports``, ``GET /reports/{stem}``, ``POST /analyze``.
"""

from __future__ import annotations

from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("API deps missing: pip install -r requirements.txt") from exc

from topspin_review.interfaces.service import analyze_video
from topspin_review.storage import runtime, store

runtime.setup()

app = FastAPI(title="Topspin Review")


class AnalyzeRequest(BaseModel):
    video_path: str
    region_box: list[float] | None = None


@app.get("/reports")
def list_reports() -> list[dict]:
    return [{"source": r.get("source"), "date": r.get("date"), "focus": r.get("focus")} for r in store.get_reports()]


@app.get("/reports/{stem}")
def get_report(stem: str) -> dict:
    for report in store.get_reports():
        if Path(report.get("source", "")).stem == stem:
            return report
    raise HTTPException(status_code=404, detail="report not found")


@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> dict:
    if not Path(request.video_path).exists():
        raise HTTPException(status_code=400, detail=f"video not found: {request.video_path}")
    box = tuple(request.region_box) if request.region_box and len(request.region_box) == 4 else None
    result = await analyze_video(request.video_path, region_box=box)
    return {"report_path": result.get("report_path"), "report": result.get("report")}
