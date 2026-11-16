"""What every model backend implements, plus per-call token usage reporting."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    def chat_json(self, messages: list[dict[str, str]], schema: type[T]) -> T: ...


class LLMError(RuntimeError):
    """Any failure to get a schema-valid answer: network, auth, rate limit, refusal, bad JSON."""


@dataclass
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


# Set by a provider after each call; read by TracedLLM. A context variable keeps it correct
# under FastAPI's threadpool without changing the LLM interface.
last_usage: ContextVar[Optional[Usage]] = ContextVar("last_usage", default=None)
