"""LLM provider loader.

Centralises construction of chat models so we can swap providers via env vars.
Supports NVIDIA by default and OpenAI when `langchain-openai` is installed.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

DEFAULT_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"


class LLMConfigError(RuntimeError):
    """Raised when the requested LLM provider cannot be initialised."""


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip() or default


def _resolve_provider() -> str:
    return (_env("LLM_PROVIDER", "groq") or "groq").lower()


def _resolve_model(default: str) -> str:
    return _env("LLM_MODEL", default) or default


def create_chat_model(temperature: float = 0.1) -> BaseChatModel:
    """Return a LangChain chat model for the configured provider."""

    provider = _resolve_provider()

    if provider in {"groq"}:
        try:
            from langchain_groq import ChatGroq  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMConfigError(
                "Groq provider selected but langchain-groq is not installed. "
                "Run `pip install langchain-groq` or switch LLM_PROVIDER."
            ) from exc

        api_key = _env("LLM_API_KEY") or _env("GROQ_API_KEY")
        if not api_key:
            raise LLMConfigError(
                "Groq provider selected but no API key found. "
                "Set LLM_API_KEY or GROQ_API_KEY."
            )

        model = _resolve_model("llama-3.1-8b-instant")
        return ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=api_key,
        )

    if provider in {"nvidia", "nv", "nvcf"}:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA  # local import to avoid hard dep when unused

        api_key = (
            _env("LLM_API_KEY")
            or _env("NVIDIA_API_KEY")
            or _env("NVCF_API_KEY")
        )
        if not api_key:
            raise LLMConfigError(
                "NVIDIA provider selected but no API key found. "
                "Set LLM_API_KEY or NVIDIA_API_KEY."
            )

        base_url = _env("LLM_BASE_URL", DEFAULT_NVIDIA_BASE)
        model = _resolve_model("meta/llama-3.1-8b-instruct")
        return ChatNVIDIA(
            model=model,
            temperature=temperature,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
        )

    if provider in {"openai", "oa"}:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMConfigError(
                "OpenAI provider selected but langchain-openai is not installed. "
                "Run `pip install langchain-openai` or switch LLM_PROVIDER back to 'nvidia'."
            ) from exc

        api_key = _env("LLM_API_KEY") or _env("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigError(
                "OpenAI provider selected but no API key found. "
                "Set LLM_API_KEY or OPENAI_API_KEY."
            )

        base_url = _env("LLM_BASE_URL") or _env("OPENAI_BASE_URL")
        model = _resolve_model("gpt-4o-mini")

        kwargs = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
        }
        if base_url:
            kwargs["base_url"] = base_url.rstrip("/")

        return ChatOpenAI(**kwargs)

    raise LLMConfigError(
        f"Unsupported LLM_PROVIDER '{provider}'. Expected 'groq', 'nvidia' or 'openai'."
    )


def get_chat_model(temperature: float = 0.1) -> BaseChatModel:
    """Public entry point used by the rest of the app."""

    return create_chat_model(temperature=temperature)
