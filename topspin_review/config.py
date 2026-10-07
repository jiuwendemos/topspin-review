"""Model configuration (text + vision), read from the environment (.env)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def vision_backend() -> str:
    """Selected vision backend name (``openai`` or ``mock``)."""
    return os.getenv("VISION_BACKEND", "openai")


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _make(model_name: str, temperature: float, timeout: int):
    from openjiuwen.core.foundation.llm import (
        Model,
        ModelClientConfig,
        ModelRequestConfig,
    )

    client = ModelClientConfig(
        client_provider=os.getenv("MODEL_PROVIDER", "OpenAI"),
        api_key=os.getenv("API_KEY", ""),
        api_base=os.getenv("API_BASE", ""),
        timeout=timeout,
        verify_ssl=_bool("LLM_SSL_VERIFY", "false"),
    )
    request = ModelRequestConfig(model=model_name, temperature=temperature, top_p=0.9)
    return Model(model_client_config=client, model_config=request)


def make_model():
    """Text model that writes the coaching report."""
    return _make(
        os.getenv("MODEL_NAME", ""),
        float(os.getenv("LLM_TEMPERATURE", "0.2")),
        int(os.getenv("LLM_TIMEOUT", "180")),
    )


def make_vision_model():
    """Vision model that reads sampled video frames."""
    return _make(
        os.getenv("VISION_MODEL_NAME", "deepseek-v4-flash-vision-exp"),
        float(os.getenv("VISION_TEMPERATURE", "0.2")),
        int(os.getenv("LLM_TIMEOUT", "180")),
    )


def max_frames() -> int:
    return int(os.getenv("VISION_MAX_FRAMES", "12"))


def max_width() -> int:
    return int(os.getenv("VISION_MAX_WIDTH", "768"))


def zoom_frames() -> int:
    """Frames sampled inside the windows the first pass asks to inspect."""
    return int(os.getenv("VISION_ZOOM_FRAMES", "6"))


def max_windows() -> int:
    return int(os.getenv("VISION_MAX_WINDOWS", "3"))


def zoom_fps() -> float:
    """Target frames-per-second when sampling a zoom burst inside a window."""
    return float(os.getenv("VISION_ZOOM_FPS", "8"))


def verify_reports() -> bool:
    """Run a verification pass that drops unsupported issues."""
    return _bool("VERIFY_REPORTS", "true")


def clips_enabled() -> bool:
    """Cut short highlight clips around flagged moments."""
    return _bool("HIGHLIGHT_CLIPS", "true")


def use_cache() -> bool:
    return _bool("VISION_CACHE", "true")


def max_seconds() -> float:
    """Cap analysis to the first N seconds of a clip (0 = whole clip)."""
    return float(os.getenv("VISION_MAX_SECONDS", "0"))


def probe_budget() -> int:
    """How many cheap frames to read when locating motion-heavy windows."""
    return int(os.getenv("VISION_PROBE_BUDGET", "64"))


def llm_retries() -> int:
    return int(os.getenv("LLM_RETRIES", "2"))


def llm_timeout() -> float:
    return float(os.getenv("LLM_TIMEOUT", "180"))


def agentic_mode() -> bool:
    """Run the model-driven analysis pipeline instead of the fixed one."""
    return _bool("AGENTIC_MODE", "false")


def retrieval_enabled() -> bool:
    """Add lexically-retrieved past sessions to the report context."""
    return _bool("RETRIEVAL", "true")


def rails_enabled() -> bool:
    return _bool("RAILS", "true")


def trace_callbacks() -> bool:
    """Use Runner.callback_framework to capture usage across all model calls."""
    return _bool("TRACE_CALLBACKS", "false")


def token_budget() -> int:
    """Abort the run once cumulative tokens exceed this (0 = unlimited)."""
    return int(os.getenv("TOKEN_BUDGET", "0"))


def has_embedding() -> bool:
    return bool(os.getenv("EMBED_API_KEY")) and bool(os.getenv("EMBED_API_BASE"))


def embedding_config():
    """Embedding config for the memory rail (only when ``EMBED_*`` is set)."""
    from openjiuwen.core.foundation.store.base_embedding import EmbeddingConfig

    return EmbeddingConfig(
        model_name=os.getenv("EMBED_MODEL_NAME", "text-embedding-v3"),
        base_url=os.getenv("EMBED_API_BASE", ""),
        api_key=os.getenv("EMBED_API_KEY", ""),
    )


def validate() -> None:
    missing = [k for k in ("API_KEY", "API_BASE", "MODEL_NAME") if not os.getenv(k)]
    if missing:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ".\nCopy .env.example to .env and fill it in."
        )
