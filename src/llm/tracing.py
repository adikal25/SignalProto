"""LLM observability: every call is logged with node, model, prompt version, latency and output.

prompt_version = hash of everything except the final user message (system prompt + few-shot
examples), so editing a prompt changes the version automatically, with no manual bumping.

ReplayLLM serves recorded outputs keyed by (prompt_version, input_hash): the eval runs in CI
without a model server, and fails loudly when a prompt changed and needs re-recording.
"""
from __future__ import annotations

import hashlib
import json
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Optional, Protocol, TypeVar

from pydantic import BaseModel

from src.llm.base import LLM, LLMError, last_usage

T = TypeVar("T", bound=BaseModel)

current_trace_id: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def prompt_version(messages: list[dict[str, str]]) -> str:
    return _sha(json.dumps(messages[:-1], sort_keys=True))


def input_hash(messages: list[dict[str, str]]) -> str:
    return _sha(messages[-1]["content"])


class CallSink(Protocol):
    def log_llm_call(self, *, trace_id: Optional[str], node: str, model: str, prompt_version: str,
                     input_hash: str, input: str, output: Optional[str], error: Optional[str],
                     latency_ms: int, provider: str = "ollama", input_tokens: Optional[int] = None,
                     output_tokens: Optional[int] = None) -> None: ...


class TracedLLM:
    def __init__(self, inner: LLM, sink: CallSink, *, node: str, model: str, provider: str = "ollama"):
        self.inner, self.sink, self.node, self.model, self.provider = inner, sink, node, model, provider

    def chat_json(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        last_usage.set(None)
        start = time.perf_counter()
        output: Optional[str] = None
        error: Optional[str] = None
        try:
            result = self.inner.chat_json(messages, schema)
            output = result.model_dump_json()
            return result
        except LLMError as e:
            error = str(e)
            raise
        finally:
            usage = last_usage.get()
            self.sink.log_llm_call(
                provider=self.provider,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                trace_id=current_trace_id.get(), node=self.node, model=self.model,
                prompt_version=prompt_version(messages), input_hash=input_hash(messages),
                input=messages[-1]["content"], output=output, error=error,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )


class CassetteMiss(LookupError):
    pass


class RecordingLLM:
    """Pass-through that records outputs into a cassette file (JSONL)."""

    def __init__(self, inner: LLM, cassette: Path):
        self.inner, self.cassette = inner, cassette
        self.cassette.parent.mkdir(parents=True, exist_ok=True)
        self.cassette.write_text("")

    def chat_json(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        result = self.inner.chat_json(messages, schema)
        entry = {"prompt_version": prompt_version(messages), "input_hash": input_hash(messages),
                 "output": json.loads(result.model_dump_json())}
        with self.cassette.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return result


class ReplayLLM:
    def __init__(self, cassette: Path):
        self.entries: dict[tuple[str, str], dict] = {}
        for line in cassette.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                self.entries[(e["prompt_version"], e["input_hash"])] = e["output"]

    def chat_json(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        key = (prompt_version(messages), input_hash(messages))
        if key not in self.entries:
            raise CassetteMiss(f"no recorded output for prompt_version={key[0]} input={key[1]}; "
                               "the prompt or eval set changed: re-record with `--record`")
        return schema.model_validate(self.entries[key])
