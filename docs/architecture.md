# Architecture

Topspin Review is layered (hexagonal). Dependencies point **inward** only.

```
interfaces  ──▶  analysis  ──▶  backend / perception  ──▶  domain
                     │                  │
                     └────▶ storage / reporting
```

## Layers

| Layer | Package | Responsibility | May depend on |
|---|---|---|---|
| Domain | `domain/` | Pure rules: report schema, progress, comparison, scoring, text rendering. No I/O, no third-party libs. | stdlib only |
| Perception | `perception/` | Frames → measurements (sampling, metrics, ball, pose, imaging). | config, storage |
| Backend | `backend/` | Under-the-hood agentic machinery and provider access: openjiuwen integration (settings, models, agent, rails, tools, runner, logs), telemetry/observability, and one module per vision provider. Imports no application module. | stdlib, openjiuwen |
| Analysis | `analysis/` | Orchestration: the two analysis strategies, two-pass vision, prompts, and the report-writing agent (agent, tools, retrieval, verification). | perception, backend, storage, domain, reporting, config |
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
├── config.py              # application settings only (sampling, feature toggles)
├── reporting.py           # Markdown / HTML / PDF export
├── domain/                # report, progress, compare, evaluate, render
├── perception/            # sampling, metrics, ball, pose, imaging, cvutil
├── backend/               # provider access (see below)
├── analysis/              # orchestration (see below)
├── storage/               # runtime, cache, json_store, store
└── interfaces/            # cli, api, service, mcp/, web/
```

### The backend package

`backend/` is the **single home for everything under-the-hood of the agentic
system**: every openjiuwen import (no `from openjiuwen...` exists anywhere else),
provider/model access, rails, and telemetry (observability). `config.py` holds only
*application* settings. Vision providers are **pure transport**: they return a
`VisionResult` and import no application module — usage accounting, call tracing
and event sequencing happen in `analysis/vision_telemetry.py`.

```
backend/
├── settings.py            # backend env: provider/keys, model names, timeouts, embeddings, rails/budget/tracing
├── models.py              # make_text_model() / make_vision_model() — openjiuwen client construction
├── agent.py               # the agents file: create_agent() builds + instruments the model, then the DeepAgent
├── rails.py               # AgentRail base + TokenBudgetRail + memory_rail() + build_rails()
├── observability.py       # usage/trace capture + the execution timeline
├── tools.py               # the @tool decorator (re-exported)
├── runner.py              # Runner lifecycle, run_agent(), callback-event bridge
├── logs.py                # route openjiuwen logging to a directory
└── providers/             # one module per vision provider
    ├── base.py            #   VisionBackend protocol + VisionResult
    ├── openai.py          #   OpenAI-compatible (default)
    ├── mock.py            #   offline, deterministic (tests)
    └── registry.py        #   PROVIDERS registry + get_backend()
```

### The analysis package

`analysis/` is split by role. The two **strategies** produce the same report
shape; everything they share lives in `session.py` and `report/`, so there is one
place for the Runner lifecycle, tracing, usage accounting, artifact writing and
the DeepAgent construction. Callers obtain a strategy from the `strategies` package
(registry + factory: `strategies.resolve(agentic=...)` / `get_strategy(name)`) and
run `strategy.analyze(...)` — they never import a concrete strategy module.

```
analysis/
├── strategies/            # strategy pattern: obtain one via strategies.resolve()/get_strategy()
│   ├── base.py            #   Strategy contract (AnalyzeFn + Strategy)
│   ├── registry.py        #   STRATEGIES registry + get_strategy()/resolve()
│   ├── deterministic.py   #   default: measure → two-pass vision → write report
│   └── agentic.py         #   opt-in: the agent inspects the clip itself
├── report/                # the report-writing layer
│   ├── agent.py           #   build_agent() — the DeepAgent (reused by both strategies)
│   ├── tools.py           #   get_profile / recent_reports / save_report
│   ├── retrieval.py       #   lexical retrieval over past reports
│   └── verification.py    #   prune issues the evidence doesn't support
├── session.py             # shared run machinery: Runner, traces, usage, artifacts
├── text_agent.py          # single-turn tool-less agent helper (verification, Q&A)
├── vision.py              # two-pass vision model calls (coarse / fine / still)
├── vision_telemetry.py    # recording wrapper: usage + call I/O around a transport backend
├── prompts.py             # all prompt text + metric rendering
└── progress.py            # thread-safe progress reporting
```

## openjiuwen integration

All framework access is funnelled through `backend/`; the rest of the app imports
only `topspin_review.backend`.

- **Model clients** — `backend.models` is **internal** (`make_text_model` /
  `make_vision_model`, lazy `openjiuwen...Model`, reading `backend.settings`). External
  code never imports it.
- **Agent** — `backend.agent.create_agent` is the agents file: it builds (and
  instruments) the model, then the DeepAgent. `analysis/report/agent.build_agent` is
  the report policy over it; the deterministic and agentic strategies both use it
  (the agentic one overrides the prompt, tools and iterations).
- **Tools** — `backend.tools.tool` (the `@tool` decorator), used by
  `analysis/report/tools.py`, the agentic strategy's `inspect_window`, and
  `interfaces/mcp/tools.py`.
- **Rails** — `backend.rails` holds both the framework adapters (`AgentRail`,
  `TokenBudgetRail`, `memory_rail()`) and the policy (`build_rails()`): token budget
  from backend settings; `MemoryRail` when `EMBED_*` is set.
- **Runner** — `backend.runner.start` / `run_agent`, and its callback-event bridge
  (`on_tool_calls` / `on_llm_output`) used by `backend.observability` traces.
- **Telemetry** — `backend.observability` captures usage/traces and builds the
  execution timeline; `analysis/vision_telemetry.py` records the vision calls (the
  text model is recorded via `backend.observability.attach`).
- **Agents only.** Application code never calls `Model.invoke` directly; every text
  LLM interaction goes through an openjiuwen agent + `Runner` — the report writer
  (`analysis/report/agent`), and the tool-less single-turn helper
  (`analysis/text_agent`) used by verification and report Q&A. The one exception is
  the vision provider (`backend/providers/openai.py`), which is the framework
  transport that carries image parts to the model.
- **Logging** — `backend.logs.configure` routes openjiuwen logging into `runtime/logs/`.

Everything above is optional at import time (lazy imports), so the deterministic
parts (perception/domain/storage) run without openjiuwen installed.

## Data flow

```
video ─▶ perception.sampling (motion-weighted, cached in storage.cache)
      ─▶ perception.metrics.analyze (energy, shift, subject, mechanics, ball, pose)
      ─▶ analysis.vision.coarse (contact sheet + prompts.metrics_text) ─▶ windows
      ─▶ sampling.sample_window (zoom)
      ─▶ analysis.vision.fine (frames + motion map + pose) ─▶ observations/signals
      ─▶ analysis.report.agent (DeepAgent + tools) ─▶ report.json
      ─▶ reporting.export + domain.progress / domain.compare
```

## Boundaries & conventions

- **No import side effects.** `__init__.py` only sets `__version__`. Logging is
  configured by `bootstrap.setup()`, called by every interface entry point (and by
  `conftest.py` in tests) so openjiuwen logs always land in `runtime/logs/`.
- **Ports & adapters.** `backend/providers/` implements the vision port; swap
  `VISION_BACKEND=mock` for offline runs. Provider selection lives in `backend.settings`.
- **Single source of paths.** All generated state lives under `runtime/` via
  `storage.runtime` (`runtime.DATA_DIR`, `ARTIFACTS_DIR`, `CACHE_DIR`, ...).
- **Schema at the boundary.** `domain.report.normalize/validate` runs inside the
  `save_report` tool, so stored reports always match the schema.
- **One implementation each** of: report rendering (`domain.render`), report lookup
  (`storage.store.find_report`), and the tool contract (`interfaces.service.describe`).

## Extending

- New model provider → add a module under `backend/providers/` and register it in `PROVIDERS`.
- New interface (e.g. gRPC) → add a module under `interfaces/`; reuse
  `interfaces.service`.
- New perception signal → add to `perception/`, surface it through
  `perception.metrics.analyze` and `analysis.prompts.metrics_text`.
