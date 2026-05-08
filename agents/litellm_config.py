"""Uniform LiteLLM/OpenRouter configuration for agent workflows.

Provides model rotation, fallback on 429 rate-limits, and automatic
capability registration so ``with_structured_output()`` works for any
model routed through OpenRouter.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, cast

from dotenv import load_dotenv

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/openai/gpt-oss-120b:free"

# Fallback models tried (in order) when the primary model returns a 429 or
# other transient error.  Override at call-site via ``fallback_models=``.
DEFAULT_FALLBACK_MODELS: list[str] = [
    "openrouter/openai/gpt-oss-120b:free",
    "openrouter/google/gemma-3-27b-it:free",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic capability registration
# ---------------------------------------------------------------------------
_registered_models: set[str] = set()


def _ensure_openrouter_capabilities(model_id: str) -> None:
    """Register *model_id* as function-calling-capable if not already known.

    OpenRouter normalises tool-calling across providers, but LiteLLM's static
    registry often lags behind.  This is called automatically so every model
    that flows through OpenRouter is treated as tool-use capable.
    """
    if model_id in _registered_models:
        return

    import litellm

    litellm.register_model(
        model_cost={
            model_id: {
                "supports_function_calling": True,
                "supports_response_schema": True,
            },
        },
    )
    _registered_models.add(model_id)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def load_openrouter_env(env_file: str = ".env") -> None:
    """Load only OpenRouter-oriented env defaults used by LiteLLM clients."""
    load_dotenv(env_file, override=False)
    if os.environ.get("OPENROUTER_API_KEY"):
        os.environ.setdefault("OR_SITE_URL", "https://github.com/havriil/smellai")
        os.environ.setdefault("OR_APP_NAME", "smellai")


def normalize_openrouter_model(
    model: str | None, *, default: str = DEFAULT_OPENROUTER_MODEL
) -> str:
    """Return a LiteLLM model id forced through the OpenRouter provider."""
    raw = (model or default).strip()
    if not raw:
        raw = default

    if raw.startswith("openrouter/"):
        return raw
    if raw.startswith("anthropic/"):
        return f"openrouter/{raw}"
    if raw.startswith("claude-"):
        return f"openrouter/anthropic/{raw}"
    return f"openrouter/{raw}"


def current_datetime_context() -> str:
    """Current local date/time string for injecting into LLM prompts."""
    return datetime.now().astimezone().strftime(
        "Current date/time: %Y-%m-%d %H:%M:%S %Z (%z)"
    )


# ---------------------------------------------------------------------------
# Single-model factory  (unchanged public API)
# ---------------------------------------------------------------------------

def openrouter_chat_kwargs(model: str | None, **extra: Any) -> dict[str, Any]:
    """Keyword args for ChatLiteLLM that avoid non-OpenRouter provider keys."""
    load_openrouter_env()

    model_id = normalize_openrouter_model(model)
    _ensure_openrouter_capabilities(model_id)

    kwargs: dict[str, Any] = {
        "model": model_id,
        "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY"),
        "api_base": OPENROUTER_API_BASE,
        "custom_llm_provider": "openrouter",
    }
    kwargs.update(extra)
    return kwargs


def make_openrouter_chat_model(model: str | None, **extra: Any):
    """Construct ChatLiteLLM with uniform OpenRouter routing (no fallbacks)."""
    from langchain_litellm import ChatLiteLLM

    return ChatLiteLLM(**openrouter_chat_kwargs(model, **extra))


# ---------------------------------------------------------------------------
# Router-based factory  (model rotation + fallback on 429)
# ---------------------------------------------------------------------------

def _build_model_list(
    primary: str,
    fallbacks: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, list[str]]]]:
    """Build ``model_list`` and ``fallbacks`` dicts for ``litellm.Router``.

    Each model gets a unique ``model_name`` so the Router can fall back from
    the primary to each fallback in order.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")

    def _deployment(model_id: str, name: str) -> dict[str, Any]:
        return {
            "model_name": name,
            "litellm_params": {
                "model": model_id,
                "api_key": api_key,
                "api_base": OPENROUTER_API_BASE,
                "custom_llm_provider": "openrouter",
            },
        }

    primary_name = "primary"
    model_list = [_deployment(primary, primary_name)]

    fallback_names: list[str] = []
    for i, fb in enumerate(fallbacks):
        if fb == primary:
            continue
        name = f"fallback_{i}"
        model_list.append(_deployment(fb, name))
        fallback_names.append(name)

    fallback_map = [{primary_name: fallback_names}] if fallback_names else []
    return model_list, fallback_map


def make_openrouter_chat_model_with_fallbacks(
    model: str | None,
    *,
    fallback_models: list[str] | None = None,
    num_retries: int = 3,
    retry_after: int = 2,
    cooldown_time: int = 6,
    **extra: Any,
):
    """Construct a ``ChatLiteLLMRouter`` with automatic fallback on 429s.

    Usage — drop-in replacement for ``make_openrouter_chat_model``::

        model = make_openrouter_chat_model_with_fallbacks(
            "minimax/minimax-m2.7",
            fallback_models=["openai/gpt-oss-120b:free", "google/gemma-3-27b-it:free"],
            temperature=0,
        )
        structured = model.with_structured_output(MySchema, method="function_calling")

    Parameters
    ----------
    model:
        Primary model (will be normalised to ``openrouter/…``).
    fallback_models:
        Ordered list of fallback model strings.  Each is normalised.
        Defaults to :data:`DEFAULT_FALLBACK_MODELS`.
    num_retries:
        Retries *per deployment* before moving to the next fallback.
    retry_after:
        Minimum seconds between retries (exponential back-off base).
    cooldown_time:
        Seconds a deployment stays on cooldown after a failure.
    **extra:
        Forwarded to ``ChatLiteLLMRouter`` (e.g. ``temperature=0``).
    """
    from langchain_litellm import ChatLiteLLMRouter
    from litellm import Router

    load_openrouter_env()

    primary = normalize_openrouter_model(model)
    fallbacks_raw = fallback_models if fallback_models is not None else DEFAULT_FALLBACK_MODELS
    fallbacks_norm = [normalize_openrouter_model(m) for m in fallbacks_raw]

    # Register capabilities for every model in the rotation.
    for m in [primary, *fallbacks_norm]:
        _ensure_openrouter_capabilities(m)

    model_list, fallback_map = _build_model_list(primary, fallbacks_norm)

    router = Router(
        model_list=model_list,
        # mypy: Router's stub is loose for this field; typed payload is runtime-safe.
        fallbacks=cast(list[Any], fallback_map or []),
        num_retries=num_retries,
        retry_after=retry_after,
        cooldown_time=cooldown_time,
        routing_strategy="simple-shuffle",
    )

    return ChatLiteLLMRouter(router=router, model_name="primary", **extra)