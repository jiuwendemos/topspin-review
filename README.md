# Topspin Review

Record a session — table tennis, tennis, badminton, squash, padel — and get back **form and footwork insights**, the main things to fix, and drills. Built on [openjiuwen](https://github.com/openJiuwen-ai/agent-core).

**No third-party integrations.** A local video file goes in; a local report comes out. Two models are used: a **vision** model (`deepseek-v4-flash-vision-exp`) to read sampled frames, and a text model to write the coaching report.

## What it does

1. You point it at a video of a session.
2. **Measure, don't eyeball.** It probes the clip and samples frames **where motion happens**
   (motion-weighted, time-ordered). It isolates the **subject** (background subtraction) and
   computes objective metrics: per-step movement energy, global image shift, subject box/track,
   **upper/lower-body region motion** (a footwork/step proxy), a crouch trend, and a heuristic
   **ball/rally** signal. It also builds a **motion map** (red = where the image moved).
3. **Two vision passes.** Pass 1 gets a compact contact sheet + the metrics and proposes the time
   windows worth a closer look. Pass 2 zooms into those windows and returns structured findings,
   each with **evidence timestamps** and a **confidence**.
4. A text agent turns that into a **coaching report** saved as JSON: strengths, issues (with
   evidence + confidence), drills, a focus, a **progress** line vs the previous report, and
   limitations. History is kept per video.

It is honest about limits: it reasons from sampled frames plus measured motion, so it does **not**
claim to measure spin or exact ball speed, and it says so in the report's `limitations`.

## Setup

This machine already has a working openjiuwen environment. Point at one of the repo venvs — `pip`/`python` are not on PATH here.

```powershell
$PY = "C:\Workspace\openjiuwen\jiuwenswarm\.venv\Scripts\python.exe"

# everything the demo uses: analysis, pose, PDF, API, MCP, tests
& $PY -m pip install -r requirements.txt
```

Note: `mediapipe` (pose) and the other extras come from this one file. If you only
want the core CLI, `python-dotenv numpy Pillow imageio imageio-ffmpeg streamlit`
are enough; everything else degrades gracefully when missing.

`.env` is already filled in. If you reuse this elsewhere, copy `.env.example`.

## Use

Run from the `topspin-review` folder.

```powershell
# set your profile (sport, level, hand, goal)
& $PY -m topspin_review.interfaces.cli profile

# analyze a video (defaults to runtime/data/sample.mp4)
& $PY -m topspin_review.interfaces.cli analyze runtime/data/sample.mp4

# see past reports
& $PY -m topspin_review.interfaces.cli history
```

Web UI:

```powershell
& $PY -m streamlit run topspin_review/interfaces/web/ui.py
```

Make a synthetic sample video (no recording needed):

```powershell
& $PY scripts\make_sample.py
```

Self-check the analysis pipeline (no model calls):

```powershell
& $PY scripts\eval.py
```

## Architecture

Layered (hexagonal) — dependencies point inward only, enforced by
`tests/unit/test_architecture.py`:

```
interfaces → analysis → providers/perception → domain
        (observability / storage / config are neutral leaves)
```

```
topspin-review/
├── pyproject.toml            # packaging, extras (pose/pdf/api/mcp/web/dev), ruff/pytest config
├── runtime/                  # generated state, gitignored (created on first run)
│   ├── workspace/  logs/     # DeepAgent workspace scaffold + openjiuwen logs
│   └── data/                 # profile, <video>_report.json, cache/, artifacts/, exports/
├── topspin_review/
│   ├── __init__.py           # version only (no side effects)
│   ├── bootstrap.py          # runtime dirs + logging; called by entry points
│   ├── config.py             # application settings (sampling, feature toggles)
│   ├── reporting.py          # Markdown / HTML / PDF export
│   ├── domain/               # pure: report, progress, compare, evaluate, render
│   ├── perception/           # sampling, metrics, ball, pose, imaging, cvutil
│   ├── backend/              # all agentic under-the-hood: settings, models, agent,
│   │                         #   rails, tools, runner, logs, observability, providers/
│   ├── analysis/             # strategies/ (deterministic, agentic) + report/ (agent, tools, ...)
│   │                         #   prompts, progress, vision, session (shared run machinery)
│   ├── storage/              # runtime, cache, json_store, store
│   └── interfaces/           # cli, api, service, mcp/, web/
├── tests/
│   ├── unit/                 # pipeline + architecture tests
│   ├── integration/          # full-pipeline tests (need a model endpoint)
│   └── eval/expected.json    # labeled expectations for quality checks
├── scripts/                  # dev tools: make_sample, evaluate_reports
├── docs/architecture.md
├── .github/workflows/ci.yml
├── .env
└── requirements.txt          # everything: analysis, pose, PDF, API, MCP, tests
```

This project is **self-contained**: it carries its own runtime layout
(`storage/runtime.py`) and JSON helpers (`storage/json_store.py`) — no shared
package is required.

## How it uses openjiuwen

- `openjiuwen.core.foundation.llm.Model` — the vision model call (frames as `image_url` blocks) and the text model.
- `openjiuwen.harness.create_deep_agent` — the agent that writes and saves the report.
- `openjiuwen.core.foundation.tool.tool` — `get_profile`, `save_report`, `recent_reports`.
- `openjiuwen.core.runner.Runner.run_agent` — runs the agent.

## Install

```powershell
$PY = "C:\Workspace\openjiuwen\jiuwenswarm\.venv\Scripts\python.exe"
& $PY -m pip install -r requirements.txt
& $PY -m pip install -e .            # console script: topspin-review
# & $PY -m pip install -e .[all]      # + every optional extra
```

Installing is optional: running `-m topspin_review...` from the project folder
also works — the project is self-contained.

## Development

```powershell
# deterministic checks (no model calls)
& $PY -m pytest -q                 # unit + architecture tests
& $PY -m ruff check .
& $PY scripts\evaluate_reports.py  # quality gate against tests/eval/expected.json
```

Compare two analyzed sessions, export, and serve:

```powershell
& $PY -m topspin_review.interfaces.cli compare video11.mp4 video12.mp4
& $PY -m topspin_review.interfaces.cli export video12.mp4
& $PY -m uvicorn topspin_review.interfaces.api:app
& $PY -m topspin_review.interfaces.mcp.server
```

CI (`.github/workflows/ci.yml`) runs all three. Offline runs use the mock vision
backend so no API key is needed:

```powershell
$env:VISION_BACKEND = "mock"
& $PY -m topspin_review.interfaces.cli analyze runtime/data/sample.mp4
```

Programmatic use:

```python
from topspin_review.interfaces.service import analyze_video_sync
result = analyze_video_sync("runtime/data/session.mp4")
```

`mcp_tools.analyze_sport_video` exposes the same as an openjiuwen `@tool`.

## Knobs (`.env`)

- `VISION_MAX_FRAMES` (default 12) — frames sampled across the clip (motion-weighted).
- `VISION_ZOOM_FRAMES` (default 6) — extra frames inside the windows the first pass picks.
- `VISION_MAX_WINDOWS` (default 3) — how many zoom windows are allowed.
- `VISION_CACHE` (default true) — cache sampled frames per video (keyed by path+mtime+size).
- `VISION_BACKEND` (default `openai`) — set `mock` for a deterministic offline backend (tests).
- `VISION_MAX_SECONDS` (default 0 = whole clip) — cap analysis for long videos.
- `VISION_PROBE_BUDGET` (default 64) — frames read while locating motion-heavy windows.
- `LLM_RETRIES` (default 2) — retries per vision call (timeout uses `LLM_TIMEOUT`).
- `AGENTIC_MODE` (default false) — let the model drive analysis (`analyze --agentic`).
- `RETRIEVAL` (default true) — add lexically-retrieved past sessions to the report context.
- `RAILS` (default true) — enable openjiuwen rails (token-budget guard; `MemoryRail` when `EMBED_*` set).
- `TRACE_CALLBACKS` (default false) — capture usage for all calls via `Runner.callback_framework`.
- `TOKEN_BUDGET` (default 0 = unlimited) — abort a run past this many tokens.

## Notes

- Each analyzed video is saved as its own report file named after the video:
  `video11.mp4` -> `runtime/data/video11_report.json`. Re-analyzing the same video
  overwrites that video's report; other videos keep their own.
- Every issue in the report cites the timestamps it is based on and a confidence, and the
  report lists its limitations, so hallucinated claims are easy to spot.
- **Primary subject:** the foreground union is reduced to its largest connected blob, so other
  people in the hall are excluded from the motion map and metrics.
- The **Progress** tab and report `progress` are computed from all saved reports
  (`progress.py`): recurring themes, issue counts per session, focus history, and whether last
  session's focus was **achieved**.
- **Export** a report to Markdown + self-contained HTML: `analyze` does it automatically, or run
  `python -m topspin_review.interfaces.cli export [video]`. Open the HTML and print to PDF if needed.
- **Player region:** if the auto primary-subject pick is wrong, pass
  `--box left,top,right,bottom` (0..1) or use the UI sliders to restrict analysis.
- Real joint pose: `pip install -r requirements.txt` (installs `mediapipe`), then `motion.pose_estimate` returns
  knee/elbow angles and stance width (fed into the prompt and saved as a skeleton overlay);
  without it the demo uses the region/box proxy.
- Each run records vision **token/latency usage** in the report (`usage`) and the UI.
- Ball/rally detection is heuristic (blob + nearest-neighbour tracker, not a trained model) and is
  always labelled low confidence.
- Every report is scored by `evaluate.py` (evidence coverage, confidence mix, hallucination flags
  for spin/speed/angle claims); the score is shown in the UI and the benchmark.
- Coaching quality depends on video quality and camera angle (side view of the player works best).
- Not medical or professional coaching advice.
