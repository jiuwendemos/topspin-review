$ErrorActionPreference = "Stop"
$PY = "C:\Workspace\openjiuwen\jiuwenswarm\.venv\Scripts\python.exe"
if (-not (Test-Path "runtime\data\sample.mp4")) { & $PY scripts\make_sample.py }
& $PY -m topspin_review.interfaces.cli analyze runtime/data/sample.mp4
