"""Export a report to Markdown and a self-contained HTML file.

PDF uses ``reportlab`` when installed (it is in ``requirements.txt``); otherwise
only the HTML is written, which prints to PDF from any browser.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

from topspin_review.domain import render
from topspin_review.storage import runtime


def to_markdown(report: dict) -> str:
    lines = [
        f"# Topspin Review report — {render.video_name(report)}",
        "",
        f"- Sport: {report.get('sport', '')}",
        f"- Date: {report.get('date', '')}",
        "",
        report.get("summary", ""),
        "",
    ]
    for label, key in (("Strengths", "strengths"), ("Drills", "drills"), ("Limitations", "limitations")):
        items = report.get(key) or []
        if items:
            lines.append(f"## {label}")
            lines += [f"- {item}" for item in items]
            lines.append("")
    issues = report.get("issues") or []
    if issues:
        lines.append("## Issues")
        lines += [f"- {render.issue_line(i)}" for i in issues]
        lines.append("")
    if report.get("focus"):
        lines += ["## Focus next session", report["focus"], ""]
    if report.get("progress"):
        lines += ["## Progress", report["progress"], ""]
    return "\n".join(lines)


def _data_uri(path: str) -> str:
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def to_html(report: dict) -> str:
    parts = [f"<h1>Topspin Review — {html.escape(render.video_name(report))}</h1>"]
    parts.append(f"<p><b>{html.escape(report.get('sport', ''))}</b> — {html.escape(report.get('date', ''))}</p>")
    if report.get("summary"):
        parts.append(f"<p>{html.escape(report['summary'])}</p>")
    for label, key in (("Strengths", "strengths"), ("Drills", "drills"), ("Limitations", "limitations")):
        items = report.get(key) or []
        if items:
            parts.append(f"<h2>{label}</h2><ul>" + "".join(f"<li>{html.escape(str(i))}</li>" for i in items) + "</ul>")
    issues = report.get("issues") or []
    if issues:
        parts.append(
            "<h2>Issues</h2><ul>" + "".join(f"<li>{html.escape(render.issue_line(i))}</li>" for i in issues) + "</ul>"
        )
    if report.get("focus"):
        parts.append(f"<h2>Focus next session</h2><p>{html.escape(report['focus'])}</p>")
    if report.get("progress"):
        parts.append(f"<h2>Progress</h2><p>{html.escape(report['progress'])}</p>")
    for key, caption in (("motion", "Motion map"), ("pose", "Pose overlay"), ("frames", "Sampled frames")):
        path = (report.get("artifacts") or {}).get(key)
        if path and Path(path).exists():
            parts.append(f"<h2>{caption}</h2><img src='{_data_uri(path)}' style='max-width:100%'>")
    body = "\n".join(parts)
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Topspin Review report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.5}</style>"
        f"</head><body>{body}</body></html>"
    )


def _write_pdf(report: dict, pdf_path: Path) -> bool:
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore

        lines = to_markdown(report).splitlines()
        pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        y = height - 40
        for line in lines:
            for chunk in [line[i : i + 100] for i in range(0, max(1, len(line)), 100)] or [""]:
                if y < 40:
                    pdf.showPage()
                    y = height - 40
                pdf.drawString(36, y, chunk)
                y -= 14
        pdf.save()
        return True
    except Exception:
        return False


def write(report: dict, out_dir: Path | None = None) -> dict:
    out = out_dir or (runtime.DATA_DIR / "exports")
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(report.get("source", "report")).stem
    md_path = out / f"{stem}_report.md"
    html_path = out / f"{stem}_report.html"
    md_path.write_text(to_markdown(report), encoding="utf-8")
    html_path.write_text(to_html(report), encoding="utf-8")
    paths = {"markdown": str(md_path), "html": str(html_path)}
    pdf_path = out / f"{stem}_report.pdf"
    if _write_pdf(report, pdf_path):
        paths["pdf"] = str(pdf_path)
    return paths
