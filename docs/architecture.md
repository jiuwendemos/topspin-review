# Architecture

Topspin Review is layered (hexagonal). Dependencies point **inward** only.

```
interfaces  ──▶  analysis  ──▶  providers / perception  ──▶  domain
                     │                  │
                     └────▶ observability / storage / reporting
```

## Layers

| Layer | Package | Responsibility | May depend on |
|---|---|---|---|
| Domain | `domain/` | Pure rules: report schema, progress, comparison, scoring, text rendering. No I/O, no third-party libs. | stdlib only |
| Observability | `observability.py` | Token/latency usage capture (neutral leaf). | stdlib only |
| Perception | `perception/` | Frames → measurements (sampling, metrics, ball, pose, imaging). | config, storage |
| Providers | `providers.py` | Model/vision adapters behind a `Protocol` (openai / mock). | config, observability |
| Analysis | `analysis/` | Orchestration: pipeline, two-pass vision, prompts, report agent, tools. | perception, providers, storage, domain, reporting, observability, config |
| Storage | `storage/` | Runtime path layout, JSON helpers, per-video store, frame cache. | config |
| Reporting | `reporting.py` | Outbound artifacts (Markdown/HTML/PDF). | domain, storage |
| Interfaces | `interfaces/` | Inbound adapters: CLI, HTTP API, MCP, Streamlit UI, service facade. | anything |

The rules are enforced by `tests/unit/test_architecture.py` (import direction +
"domain is pure"), so a violation fails CI.

## Package map

```
topspin_review/
├── __init__.py            # version only (no side effects)
├── bootstrap.py           # runtime dirs + logging; called by entry points
├── config.py              # single settings surface (vision_backend(), validate())
├── observability.py       # token/latency usage (neutral)
├── providers.py           # vision backend seam (openai / mock)
├── reporting.py           # Markdown / HTML / PDF export
├── domain/                # report, progress, compare, evaluate, render
├── perception/            # sampling, metrics, ball, pose, imaging, cvutil
├── analysis/              # pipeline, agentic, report_agent, prompts, vision, retrieval, rails, tools
├── storage/               # runtime, cache, json_store, store
└── interfaces/            # cli, api, service, mcp/, web/
```

## openjiuwen integration

- **Model clients** — `config.make_model` / `make_vision_model` (lazy `openjiuwen...Model`).
- **Agent** — `create_deep_agent` in `analysis/report_agent.py` (fixed pipeline) and
  `analysis/agentic.py` (model-driven mode with an `inspect_window` tool).
- **Tools** — `@tool` in `analysis/tools.py` and `interfaces/mcp/tools.py`.
- **Rails** — `analysis/rails.py`: a `TokenBudgetRail` and the built-in `MemoryRail`
  (when `EMBED_*` is set).
- **Runner** — `Runner.start` / `Runner.run_agent` in the pipelines; optional
  `Runner.callback_framework` usage trace (`observability.CallbackTrace`).
- **Logging** — `storage/runtime.py` configures openjiuwen logging into `runtime/logs/`.

Everything above is optional at import time (lazy imports), so the deterministic
parts (perception/domain/storage) run without openjiuwen installed.

## Data flow

```
video ─▶ perception.sampling (motion-weighted, cached in storage.cache)
      ─▶ perception.metrics.analyze (energy, shift, subject, mechanics, ball, pose)
      ─▶ analysis.vision.coarse (contact sheet + prompts.metrics_text) ─▶ windows
      ─▶ sampling.sample_window (zoom)
      ─▶ analysis.vision.fine (frames + motion map + pose) ─▶ observations/signals
      ─▶ analysis.report_agent (DeepAgent + tools) ─▶ report.json
      ─▶ reporting.export + domain.progress / domain.compare
```

## Boundaries & conventions

- **No import side effects.** `__init__.py` only sets `__version__`. Logging is
  configured by `bootstrap.setup()`, called by every interface entry point (and by
  `conftest.py` in tests) so openjiuwen logs always land in `runtime/logs/`.
- **Ports & adapters.** `providers.py` implements the vision port; swap
  `VISION_BACKEND=mock` for offline runs. Backend selection lives in `config`.
- **Single source of paths.** All generated state lives under `runtime/` via
  `storage.runtime` (`runtime.DATA_DIR`, `ARTIFACTS_DIR`, `CACHE_DIR`, ...).
- **Schema at the boundary.** `domain.report.normalize/validate` runs inside the
  `save_report` tool, so stored reports always match the schema.
- **One implementation each** of: report rendering (`domain.render`), report lookup
  (`storage.store.find_report`), and the tool contract (`interfaces.service.describe`).

## Extending

- New model provider → add a class in `providers.py` and register it.
- New interface (e.g. gRPC) → add a module under `interfaces/`; reuse
  `interfaces.service`.
- New perception signal → add to `perception/`, surface it through
  `perception.metrics.analyze` and `analysis.prompts.metrics_text`.
