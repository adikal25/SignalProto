"""Bring-your-own-key model providers.

Each agent picks a provider in config (`llm.signal.provider`, `llm.outbound.provider`):

- ollama             local, no key, no per-call cost (default)
- anthropic          Claude via the official `anthropic` SDK, JSON-schema structured outputs
- openai, groq, openrouter, gemini, openai_compatible
                     any OpenAI-compatible /chat/completions endpoint with a JSON-schema response_format

Keys come only from environment variables and are never logged, stored or returned by the API.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from src.llm.base import LLM, LLMError, T, Usage, last_usage
from src.llm.ollama_runtime import OllamaChat, OllamaConfig
from src.settings import LLMSettings


@dataclass(frozen=True)
class Preset:
    key_env: str
    base_url: Optional[str] = None


PRESETS: dict[str, Preset] = {
    "anthropic": Preset("ANTHROPIC_API_KEY"),
    "openai": Preset("OPENAI_API_KEY", "https://api.openai.com/v1"),
    "groq": Preset("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    "openrouter": Preset("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "gemini": Preset("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai"),
    "openai_compatible": Preset("LLM_API_KEY"),
}


def key_env_for(cfg: LLMSettings) -> Optional[str]:
    if cfg.provider == "ollama":
        return None
    return cfg.api_key_env or PRESETS[cfg.provider].key_env


def _api_key(cfg: LLMSettings) -> str:
    env = key_env_for(cfg)
    key = os.getenv(env or "")
    if not key:
        raise ValueError(f"llm provider '{cfg.provider}' needs an API key in ${env}")
    return key


# --- Anthropic -------------------------------------------------------------------

# Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicChat:
    provider = "anthropic"

    def __init__(self, cfg: LLMSettings, api_key: str):
        import anthropic  # optional dependency: only needed when this provider is configured

        self._anthropic = anthropic
        self.cfg = cfg
        self.client = anthropic.Anthropic(api_key=api_key, timeout=float(cfg.timeout_seconds))

    def chat_json(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {
            "model": self.cfg.model_id,
            "max_tokens": self.cfg.max_new_tokens,
            "system": system,
            "messages": turns,
            "output_format": schema,
        }
        if self.cfg.effort:
            kwargs["output_config"] = {"effort": self.cfg.effort}
        if self.cfg.model_id in _FALLBACK_MODELS:
            # If the model declines, the API re-runs the request on a fallback model in the same call.
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        a = self._anthropic
        try:
            response = self.client.beta.messages.parse(**kwargs)  # SDK retries 429 / 5xx itself
        except a.AuthenticationError as e:
            raise LLMError(f"anthropic: invalid API key ({e.message})") from e
        except a.RateLimitError as e:
            raise LLMError("anthropic: rate limited after retries") from e
        except a.APIStatusError as e:
            raise LLMError(f"anthropic: HTTP {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError(f"anthropic: connection error: {e}") from e
        last_usage.set(Usage(response.usage.input_tokens, response.usage.output_tokens))
        if response.stop_reason == "refusal":
            raise LLMError("anthropic: model declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"anthropic: hit max_tokens={self.cfg.max_new_tokens}; raise llm.*.max_new_tokens")
        if response.parsed_output is None:
            raise LLMError("anthropic: no structured output in response")
        return response.parsed_output


# --- OpenAI-compatible -----------------------------------------------------------

class OpenAICompatibleChat:
    """Any server implementing POST /chat/completions with `response_format: json_schema`."""

    RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, cfg: LLMSettings, api_key: str, max_retries: int = 2):
        base_url = cfg.base_url or PRESETS[cfg.provider].base_url
        if not base_url:
            raise ValueError(f"llm provider '{cfg.provider}' needs base_url")
        self.provider = cfg.provider
        self.cfg = cfg
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.max_retries = max_retries

    def _payload(self, messages: list[dict[str, str]], schema: type[T]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.cfg.model_id,
            "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": False,
            }},
        }
        # OpenAI's own API renamed the limit; other compatible servers still use max_tokens.
        payload["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = self.cfg.max_new_tokens
        if self.cfg.temperature is not None:
            payload["temperature"] = self.cfg.temperature
        return payload

    def chat_json(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        payload = self._payload(messages, schema)
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(self.url, json=payload, headers=self.headers, timeout=self.cfg.timeout_seconds)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise LLMError(f"{self.provider}: connection error: {e}") from e
                time.sleep(2 ** attempt + random.random())
                continue
            if r.status_code in self.RETRY_STATUSES and attempt < self.max_retries:
                retry_after = r.headers.get("retry-after", "")
                time.sleep(float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2 ** attempt + random.random())
                continue
            if r.status_code == 401:
                raise LLMError(f"{self.provider}: invalid API key")
            if r.status_code >= 400:
                raise LLMError(f"{self.provider}: HTTP {r.status_code}: {r.text[:300]}")
            break
        try:
            body = r.json()
            usage = body.get("usage") or {}
            last_usage.set(Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")))
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                raise LLMError(f"{self.provider}: hit max tokens; raise llm.*.max_new_tokens")
            return schema.model_validate_json(choice["message"]["content"])
        except LLMError:
            raise
        except Exception as e:  # malformed body or schema-invalid JSON
            raise LLMError(f"{self.provider}: bad response: {e}") from e


# --- factory ---------------------------------------------------------------------

def make_llm(cfg: LLMSettings) -> LLM:
    if cfg.provider == "ollama":
        return OllamaChat(OllamaConfig(cfg.model_id, cfg.max_new_tokens, cfg.temperature or 0.0, cfg.timeout_seconds))
    if cfg.model_id == LLMSettings().model_id:
        raise ValueError(f"llm provider '{cfg.provider}' needs a model_id for that provider (got the local default)")
    key = _api_key(cfg)
    if cfg.provider == "anthropic":
        return AnthropicChat(cfg, key)
    return OpenAICompatibleChat(cfg, key)


def provider_status(cfg: LLMSettings) -> dict[str, Any]:
    """Safe to expose: says whether a key is present, never the key itself."""
    env = key_env_for(cfg)
    return {"provider": cfg.provider, "model": cfg.model_id, "key_env": env,
            "key_configured": True if env is None else bool(os.getenv(env))}
