# Integration tests

These tests exercise the full pipeline (sampling → two vision passes → report
agent) and therefore need a configured model endpoint (`.env`) and cost tokens.

They are skipped by default. Run explicitly, e.g.:

```powershell
& $PY -m pytest tests\integration -q
```

Both text and vision go through model calls, so an endpoint (`.env`) is required;
there is no offline mock.
