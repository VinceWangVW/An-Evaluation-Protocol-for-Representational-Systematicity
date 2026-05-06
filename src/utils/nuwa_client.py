from __future__ import annotations

import os
from typing import Any


DEFAULT_NUWA_BASE_URL = "https://api.nuwaapi.com/v1"
# Prefer the strongest model we have actually verified on the current provider.
# Users can still override this with NUWA_MODEL when a stronger channel opens up.
DEFAULT_NUWA_MODEL = "gpt-4.1"


def build_nuwa_client() -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["NUWA_API_KEY"],
        base_url=os.environ.get("NUWA_BASE_URL", DEFAULT_NUWA_BASE_URL),
    )


def get_nuwa_model_name() -> str:
    return os.environ.get("NUWA_MODEL", DEFAULT_NUWA_MODEL)
