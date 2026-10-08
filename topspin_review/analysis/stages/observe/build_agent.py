"""Vision-agent policy: the prompt for the video-understanding agent.

Like the report agent (``coaching/agent.py``), this is a thin business policy over
the backend agent builder. It returns the build result, whose recorder is the run
recorder (the vision agent is the first agent built in a run).
"""

from __future__ import annotations

from typing import Any

from topspin_review.analysis.stages.observe import prompts
from topspin_review.backend import VisionParams, build
from topspin_review.bootstrap import setup
from topspin_review.storage import runtime


def build_agent(*, media_dir: str, workspace: str | None = None) -> Any:
    """Build the vision DeepAgent (reads rendered images via ``read_file``)."""
    setup()
    return build(
        VisionParams(
            system_prompt=prompts.VISION_AGENT_SYSTEM,
            workspace=workspace or str(runtime.ARTIFACTS_DIR),
            media_dir=media_dir,
        )
    )
