"""Client for a local Ollama server. No hosted API, so no per-call cost."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import requests
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def chat_json(self, messages: list[dict[str, str]], schema: type[T]) -> T: ...


class LLMError(RuntimeError):
    pass


@dataclass
class OllamaConfig:
    model_id: str
    max_new_tokens: int = 400
    temperature: float = 0.0
    timeout_seconds: int = 120


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


class OllamaChat:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self.base_url = ollama_base_url()

    def chat_json(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        """Chat with Ollama structured outputs: the model is constrained to `schema`'s JSON schema."""
        payload: dict[str, Any] = {
            "model": self.cfg.model_id,
            "messages": messages,
            "format": schema.model_json_schema(),
            "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_new_tokens},
            "stream": False,
        }
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.cfg.timeout_seconds)
            r.raise_for_status()
            content = r.json()["message"]["content"]
            return schema.model_validate_json(content)
        except Exception as e:  # network, HTTP, malformed or truncated JSON
            raise LLMError(f"{self.cfg.model_id}: {e}") from e


def ollama_health(model_ids: list[str]) -> dict[str, Any]:
    """Report whether Ollama is reachable and which configured models are pulled."""
    try:
        r = requests.get(f"{ollama_base_url()}/api/tags", timeout=3)
        r.raise_for_status()
        available = {m["name"] for m in r.json().get("models", [])}
    except requests.RequestException as e:
        return {"reachable": False, "error": str(e), "missing_models": model_ids}
    missing = [m for m in model_ids if m not in available]
    return {"reachable": True, "missing_models": missing}
