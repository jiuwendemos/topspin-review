# Integration tests

These tests exercise the full pipeline (sampling → two vision passes → report
agent) and therefore need a configured model endpoint (`.env`) and cost tokens.

They are skipped by default. Run explicitly, e.g.:

```powershell
$env:VISION_BACKEND = "openai"
& $PY -m pytest tests\integration -q
```

For fully offline runs use the mock vision backend (`VISION_BACKEND=mock`), which
still calls the text model.
