"""Backend connection settings, read from the environment (.env).

Everything here is about *reaching* model/embedding endpoints: which provider,
credentials, base URL, model names, timeouts/retries. Application behaviour
(sampling, feature toggles, budgets) lives in :mod:`topspin_review.config`.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required backend configuration is missing or invalid."""


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def vision_backend() -> str:
    """Selected vision provider name (``openai`` or ``mock``)."""
    return os.getenv("VISION_BACKEND", "openai")


def model_provider() -> str:
    return os.getenv("MODEL_PROVIDER", "OpenAI")


def api_base() -> str:
    return os.getenv("API_BASE", "")


def api_key() -> str:
    return os.getenv("API_KEY", "")


def text_model_name() -> str:
    return os.getenv("MODEL_NAME", "")


def vision_model_name() -> str:
    return os.getenv("VISION_MODEL_NAME", "deepseek-v4-flash-vision-exp")


def text_temperature() -> float:
    return float(os.getenv("LLM_TEMPERATURE", "0.2"))


def vision_temperature() -> float:
    return float(os.getenv("VISION_TEMPERATURE", "0.2"))


def llm_retries() -> int:
    return int(os.getenv("LLM_RETRIES", "2"))


def llm_timeout() -> float:
    return float(os.getenv("LLM_TIMEOUT", "180"))


def verify_ssl() -> bool:
    return _bool("LLM_SSL_VERIFY", "false")


# --- run constraints & observability (under-the-hood behaviour) ------------- #
def rails_enabled() -> bool:
    """Attach openjiuwen rails (token budget / memory) to report agents."""
    return _bool("RAILS", "true")


def token_budget() -> int:
    """Abort a run once cumulative model tokens exceed this (0 = unlimited)."""
    return int(os.getenv("TOKEN_BUDGET", "0"))


def trace_callbacks() -> bool:
    """Use Runner.callback_framework to capture usage across all model calls."""
    return _bool("TRACE_CALLBACKS", "false")


def save_call_io() -> bool:
    """Persist the full prompt/output text of each model call (privacy-sensitive)."""
    return _bool("SAVE_CALL_IO", "true")


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
    """Ensure the required backend environment variables are present."""
    missing = [k for k in ("API_KEY", "API_BASE", "MODEL_NAME") if not os.getenv(k)]
    if missing:
        raise ConfigError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ".\nCopy .env.example to .env and fill it in."
        )
