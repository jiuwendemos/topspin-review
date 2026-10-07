# Labeled evaluation

`expected.json` maps a video **stem** to what its report should contain:

```json
{
  "video11": {"themes": ["footwork"], "min_score": 60}
}
```

- `themes` — coaching themes (see `progress.THEMES`) that must appear among the report's issues.
- `min_score` — minimum `evaluate.score()['score']`.

Check saved reports against it:

```powershell
& $PY scripts\evaluate_reports.py
```

With no reports (or no matching expectations) it prints a summary and passes, so it is safe in CI.
Record real clips and add expectations to turn this into a behavioral regression gate.
