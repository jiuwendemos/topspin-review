# Architecture

Topspin Review is layered (hexagonal). Dependencies point **inward** only.

```
interfaces  ──▶  analysis  ──▶  providers / perception  ──▶  domain
     │                │
     └────────────▶ storage / reporting
```

## Layers

| Layer | Package | Responsibility | May depend on |
|---|---|---|---|
| Domain | `domain/` | Pure rules: report schema, progress, comparison, scoring. No I/O, no third-party libs. | stdlib only |
| Perception | `perception/` | Frames → measurements (sampling, motion, ball, pose, imaging). | domain, storage, config |
| Providers | `providers/` | Model/vision adapters behind a `Protocol` (openai / mock). | config, analysis.usage |
| Analysis | `analysis/` | Orchestration: two-pass vision, report agent, agent tools, usage. | perception, providers, storage, domain, reporting, config |
| Storage | `storage/` | Runtime path layout + persistence (per-video JSON). | config |
| Reporting | `reporting/` | Outbound artifacts (Markdown/HTML/PDF). | storage |
| Interfaces | `interfaces/` | Inbound adapters: CLI, HTTP API, MCP, Streamlit UI, service facade. | anything |

The rules are enforced by `tests/unit/test_architecture.py` (import-direction +
"domain is pure"), so a violation fails CI.

## Data flow

```
video ─▶ sampling (motion-weighted, cached)
      ─▶ perception.analyze (energy, shift, subject, mechanics, ball, pose)
      ─▶ vision.coarse (contact sheet + metrics) ─▶ windows
      ─▶ sampling.sample_window (zoom)
      ─▶ vision.fine (frames + motion map + pose) ─▶ observations/signals
      ─▶ analysis.agent (DeepAgent + tools) ─▶ report.json
      ─▶ reporting.export (md/html/pdf) + progress/compare
```

## Boundaries & conventions

- **No import side effects.** `__init__.py` only exposes the shared-package path
  bootstrap. Logging is configured by `storage.runtime.setup()` from each
  interface entry point (and before heavy imports).
- **Ports & adapters.** `providers/backends.py` implements the vision port; swap
  `VISION_BACKEND=mock` for offline runs.
- **Single source of paths.** All generated state lives under `runtime/` via
  `storage.runtime` (`runtime.DATA_DIR`, `ARTIFACTS_DIR`, `CACHE_DIR`, ...).
- **Schema at the boundary.** `domain.report.normalize/validate` runs inside the
  `save_report` tool, so stored reports always match the schema.

## Extending

- New model provider → add a class in `providers/backends.py` and register it.
- New interface (e.g. gRPC) → add a module under `interfaces/`; reuse
  `interfaces.service`.
- New perception signal → add to `perception/`, surface it through
  `perception.motion.analyze` and `metrics_text`.
