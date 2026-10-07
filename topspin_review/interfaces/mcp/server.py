"""Stdio MCP server exposing ``analyze_sport_video``.

    pip install -r requirements.txt
    python -m topspin_review.interfaces.mcp.server
"""

from __future__ import annotations

from topspin_review.bootstrap import setup


def main() -> None:
    setup()
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MCP deps missing: pip install -r requirements.txt") from exc

    from topspin_review.interfaces import service

    server = FastMCP("topspin-review")

    @server.tool()
    def analyze_sport_video(video_path: str, region_box: list[float] | None = None) -> dict:
        """Analyze a racket-sport session video and return a coaching report.

        ``region_box`` optionally restricts analysis to [left, top, right, bottom] (0..1).
        """
        result = service.analyze_video_sync(video_path, region_box=service.parse_region_box(region_box))
        return {"report_path": result.get("report_path"), "report": result.get("report")}

    server.run()


if __name__ == "__main__":
    main()
