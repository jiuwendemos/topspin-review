"""Model configuration (text + vision), read from the environment (.env)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


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


def validate() -> None:
    missing = [k for k in ("API_KEY", "API_BASE", "MODEL_NAME") if not os.getenv(k)]
    if missing:
        raise SystemExit(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ".\nCopy .env.example to .env and fill it in."
        )
