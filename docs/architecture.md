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
model/agent access, rails, and telemetry. `config.py` holds only *application*
settings. Both text and vision go through agents built by the agent builder —
there is no separate provider transport. The package exposes a small façade
(`build`, `TextParams`/`VisionParams`, `run_agent`, `run_text`, `ConfigError`,
`configure_logging`); everything else is internal. Internally it's grouped by
concern:

```
backend/
├── __init__.py            # the façade
├── settings.py            # backend env: keys, model names, timeouts, embeddings, budget/tracing
├── logs.py                # route openjiuwen logging to a directory
├── agent/                 # build & run an agent
│   ├── builder/           #   the agent builder
│   │   ├── build.py       #     build(params): the single entry point
│   │   ├── params.py      #     Params / TextParams / VisionParams
│   │   ├── base.py        #     AgentBuilder (owns recorder + instrumentation)
│   │   ├── text.py        #     TextBuilder
│   │   ├── vision.py      #     VisionBuilder
│   │   └── result.py      #     BuildResult
│   ├── models/            #   generic model construction
│   │   ├── builder.py     #     build_model(params) (constructs an openjiuwen Model)
│   │   └── params.py      #     ModelParams
│   ├── rails.py           #   AgentRail base + TokenBudgetRail + memory_rail() + resolve(names)
│   ├── tools.py           #   make_tool()/make_tools() (openjiuwen tool decoration)
│   └── runner.py          #   Runner lifecycle, run_agent(), callback-event bridge
└── telemetry/             # run telemetry
    ├── usage.py           #   token/latency extraction + UsageCollector
    ├── traces.py          #   CallTrace / ToolTrace / CallbackTrace
    └── recorder.py        #   Recorder + attach (the builder-owned run recorder)
```

### The analysis package

`analysis/` is grouped by **business area**: understanding the video, producing the
coaching report, and the analysis modes. The two **strategies** produce the same
report shape; everything they share lives in `run_session.py` and `coaching/`. Callers
obtain a strategy from the `strategies` package (registry: `strategies.resolve(...)`
/ `get_strategy(name)`) and run `strategy.analyze(...)` — they never import a
concrete strategy module.

```
analysis/
├── vision/                # video frames → observations
│   ├── build_agent.py     #   build the vision DeepAgent
│   ├── render_images.py   #   render/save the images the model reads
│   ├── ask_agent.py       #   ask the vision agent + parse JSON
│   └── run_passes.py      #   the overview / detail / still passes
├── coaching/              # observations → coaching report
│   ├── build_agent.py     #   build the report DeepAgent
│   ├── report_tools.py    #   get_profile / recent_reports / save_report
│   ├── retrieve_reports.py#   lexical retrieval over past reports
│   └── verify_issues.py   #   prune issues the evidence doesn't support
├── strategies/            # the analysis modes
│   ├── strategy.py        #   Strategy contract (AnalyzeFn + Strategy)
│   ├── resolve_strategy.py#   STRATEGIES registry + get_strategy()/resolve()
│   ├── deterministic.py   #   default: measure → two-pass vision → write report
│   └── agentic.py         #   opt-in: the agent inspects the clip itself
├── run_session.py         # shared run machinery: Runner, recorder, vision agent
├── prompts.py             # all prompt text + metric rendering
└── progress.py            # thread-safe progress reporting
```

## openjiuwen integration

All framework access is funnelled through `backend/`; the rest of the app imports
only `topspin_review.backend`.

- **Model clients** — `backend.agent.models` is **internal** (`make_text_model` /
  `make_vision_model`, lazy `openjiuwen...Model`, reading `backend.settings`). External
  code never imports it.
- **Agent builder** — `backend.agent.builder.build(params)` is the single entry
  point: it builds (and instruments) the model, decorates tools, resolves rails and
  constructs the DeepAgent. The `backend.agent.params` dataclasses (`TextParams`,
  `VisionParams`) describe the agent. `analysis/coaching/build_agent.build_agent` is the
  report policy over `TextParams`; the deterministic and agentic strategies both use
  it (the agentic one overrides the prompt, tools and iterations).
- **Tools** — application code passes plain callables
  (`analysis/coaching/report_tools.py`, the agentic strategy's `inspect_window`); the builder
  decorates them internally (auto-extracting name/description/schema from the
  function/docstring/signature). Nothing about tooling decoration is known outside
  `backend`. (External exposure is over MCP — `interfaces/mcp/server.py`, FastMCP.)
- **Rails** — application code only names rails (`config.rails()`, e.g.
  `["token_budget", "memory"]`); `backend.agent.builder.build` resolves the names via
  `backend.agent.rails.resolve()` and builds/attaches the rails. The implementations
  (`AgentRail`, `TokenBudgetRail`, `memory_rail()`) never leave `backend`.
- **Runner** — `backend.agent.runner.run_agent`, and its callback-event bridge
  (`on_tool_calls` / `on_llm_output`) used by `backend.telemetry` traces.
- **Vision** — `analysis/vision/agent.build_agent` builds (via the backend builder) a
  DeepAgent whose model is the vision model, reading rendered images through the
  `read_file` tool (`SysOperationRail` + `enable_read_image_multimodal`).
  `analysis/vision/passes.py` persists contact sheets / motion maps / frames and asks
  the agent to read them.
- **Telemetry** — the builder creates a run **Recorder** (`backend.telemetry.recorder`,
  internal) that captures usage/traces; callers never touch it directly. The UI shapes
  it for display in `interfaces/web/timeline.py` (presentation, not backend).
- **Agents only.** Application code never calls `Model.invoke` directly; every model
  interaction — text, verification, Q&A, and vision — goes through an openjiuwen
  agent + `Runner` built by the backend agent builder.
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
      ─▶ analysis.coaching.build_agent (DeepAgent + tools) ─▶ report.json
      ─▶ reporting.export + domain.progress / domain.compare
```

## Boundaries & conventions

- **No import side effects.** `__init__.py` only sets `__version__`. Logging is
  configured by `bootstrap.setup()`, called by every interface entry point (and by
  `conftest.py` in tests) so openjiuwen logs always land in `runtime/logs/`.
- **Agents, not providers.** Text and vision both go through the backend agent
  builder; there is no separate provider transport to swap.
- **Single source of paths.** All generated state lives under `runtime/` via
  `storage.runtime` (`runtime.DATA_DIR`, `ARTIFACTS_DIR`, `CACHE_DIR`, ...).
- **Schema at the boundary.** `domain.report.normalize/validate` runs inside the
  `save_report` tool, so stored reports always match the schema.
- **One implementation each** of: report rendering (`domain.render`), report lookup
  (`storage.store.find_report`), and the tool contract (`interfaces.service.describe`).

## Extending

- New interface (e.g. gRPC) → add a module under `interfaces/`; reuse
  `interfaces.service`.
- New perception signal → add to `perception/`, surface it through
  `perception.metrics.analyze` and `analysis.prompts.metrics_text`.
