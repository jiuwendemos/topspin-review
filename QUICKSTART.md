# Quick start

Record a session — table tennis, tennis, badminton, squash, padel — and get back a coaching
report (strengths, issues with evidence, drills, a focus, progress vs the last report). A local
video file goes in; a local report comes out.

Run every command **from the `topspin-review` folder**. On this machine `pip`/`python` are not on
PATH, so use the repo venv's python:

```powershell
$PY = "C:\Workspace\openjiuwen\jiuwenswarm\.venv\Scripts\python.exe"
```

## 1. Install (once)

```powershell
& $PY -m pip install -r requirements.txt
```

`.env` is already filled in with the model endpoint. If you reuse this elsewhere, copy
`.env.example`. Analysis needs a model endpoint (text + vision); the unit tests are fully offline.

## 2. Run

**Web UI (recommended)** — opens http://localhost:8501:

```powershell
& $PY -m streamlit run topspin_review/interfaces/web/ui.py
```

**CLI** — set your profile, then analyze a video:

```powershell
# sport, level, hand, goal
& $PY -m topspin_review.interfaces.cli profile

# analyze a video (defaults to runtime/data/sample.mp4)
& $PY -m topspin_review.interfaces.cli analyze runtime/data/sample.mp4

# past reports
& $PY -m topspin_review.interfaces.cli history
```

No recording handy? Make a synthetic sample video:

```powershell
& $PY scripts\make_sample.py
```

## 3. Useful extras

```powershell
# restrict analysis to a player region: left,top,right,bottom (0..1)
& $PY -m topspin_review.interfaces.cli analyze runtime/data/sample.mp4 --box 0.2,0.1,0.8,0.9

# let the model drive the analysis (agentic mode)
& $PY -m topspin_review.interfaces.cli analyze runtime/data/sample.mp4 --agentic

# compare two analyzed sessions, export, serve
& $PY -m topspin_review.interfaces.cli compare video11.mp4 video12.mp4
& $PY -m topspin_review.interfaces.cli export video12.mp4
& $PY -m uvicorn topspin_review.interfaces.api:app
& $PY -m topspin_review.interfaces.mcp.server
```

Programmatic use:

```python
from topspin_review.interfaces.service import analyze_video_sync
result = analyze_video_sync("runtime/data/session.mp4")
```

## 4. Offline checks (no model calls)

```powershell
& $PY -m pytest -q                 # unit + architecture tests
& $PY -m ruff check .              # lint
& $PY scripts\evaluate_reports.py  # quality gate against tests/eval/expected.json
```

See [README.md](README.md) for what it does, the architecture, and the `.env` knobs.
