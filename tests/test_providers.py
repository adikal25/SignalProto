"""Bring-your-own-key providers. All network calls are faked."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.llm.base import LLMError, last_usage
from src.llm.ollama_runtime import OllamaChat
from src.llm.providers import AnthropicChat, OpenAICompatibleChat, make_llm, provider_status
from src.llm.tracing import TracedLLM
from src.schemas import SignalVerdict
from src.settings import LLMSettings
from src.store import Store

MESSAGES = [
    {"role": "system", "content": "You verify signals."},
    {"role": "user", "content": "example"},
    {"role": "assistant", "content": '{"signal_type": "none"}'},
    {"role": "user", "content": "SENTENCE: We're hiring a nurse."},
]
VERDICT = {"signal_type": "hiring", "strength": "strong", "detail": "nurse", "reason": "role"}


def http(status: int, body: dict | None = None, headers: dict | None = None):
    r = MagicMock(status_code=status, headers=headers or {}, text=json.dumps(body or {}))
    r.json.return_value = body or {}
    return r


def completion(content: dict, finish: str = "stop") -> dict:
    return {"choices": [{"message": {"content": json.dumps(content)}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}}


# --- factory & keys ----------------------------------------------------------------

def test_default_is_local_ollama():
    assert isinstance(make_llm(LLMSettings()), OllamaChat)


def test_hosted_provider_needs_key_and_model(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        make_llm(LLMSettings(provider="groq", model_id="some-model"))
    with pytest.raises(ValueError, match="model_id"):
        make_llm(LLMSettings(provider="groq"))  # still the local default model name


def test_custom_key_env_and_base_url(monkeypatch):
    monkeypatch.setenv("MY_VLLM_KEY", "k")
    llm = make_llm(LLMSettings(provider="openai_compatible", model_id="m", base_url="http://gpu:8000/v1",
                               api_key_env="MY_VLLM_KEY"))
    assert isinstance(llm, OpenAICompatibleChat) and llm.url == "http://gpu:8000/v1/chat/completions"
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleChat(LLMSettings(provider="openai_compatible", model_id="m"), "k")


def test_provider_status_never_exposes_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    status = provider_status(LLMSettings(provider="openai", model_id="m"))
    assert status["key_configured"] is True and status["key_env"] == "OPENAI_API_KEY"
    assert "sk-secret-value" not in json.dumps(status)


# --- OpenAI-compatible -------------------------------------------------------------

def test_openai_compatible_request_and_usage():
    llm = OpenAICompatibleChat(LLMSettings(provider="groq", model_id="m", max_new_tokens=99), "k")
    with patch("src.llm.providers.requests.post", return_value=http(200, completion(VERDICT))) as post:
        verdict = llm.chat_json(MESSAGES, SignalVerdict)
    assert verdict.signal_type == "hiring"
    body = post.call_args.kwargs["json"]
    assert body["response_format"]["type"] == "json_schema" and body["max_tokens"] == 99
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer k"
    assert last_usage.get().input_tokens == 120


def test_openai_uses_max_completion_tokens_and_can_omit_temperature():
    llm = OpenAICompatibleChat(LLMSettings(provider="openai", model_id="m", temperature=None), "k")
    with patch("src.llm.providers.requests.post", return_value=http(200, completion(VERDICT))) as post:
        llm.chat_json(MESSAGES, SignalVerdict)
    body = post.call_args.kwargs["json"]
    assert "max_completion_tokens" in body and "max_tokens" not in body and "temperature" not in body


def test_openai_compatible_retries_rate_limit_then_succeeds():
    llm = OpenAICompatibleChat(LLMSettings(provider="openrouter", model_id="m"), "k")
    responses = [http(429, headers={"retry-after": "0"}), http(200, completion(VERDICT))]
    with patch("src.llm.providers.requests.post", side_effect=responses) as post, \
         patch("src.llm.providers.time.sleep"):
        assert llm.chat_json(MESSAGES, SignalVerdict).signal_type == "hiring"
    assert post.call_count == 2


@pytest.mark.parametrize("response,match", [
    (http(401), "invalid API key"),
    (http(400, {"error": "bad"}), "HTTP 400"),
    (http(200, completion(VERDICT, finish="length")), "max tokens"),
    (http(200, completion({"signal_type": "maybe"})), "bad response"),
])
def test_openai_compatible_errors_become_llm_errors(response, match):
    llm = OpenAICompatibleChat(LLMSettings(provider="openai", model_id="m"), "k")
    with patch("src.llm.providers.requests.post", return_value=response), pytest.raises(LLMError, match=match):
        llm.chat_json(MESSAGES, SignalVerdict)


# --- Anthropic ---------------------------------------------------------------------

def anthropic_llm(model: str = "claude-opus-5-5", **cfg) -> tuple[AnthropicChat, MagicMock]:
    llm = AnthropicChat(LLMSettings(provider="anthropic", model_id=model, max_new_tokens=4096, **cfg), "sk-ant-test")
    parse = MagicMock()
    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse)))
    return llm, parse


def parsed(stop_reason: str = "end_turn", output=None):
    return SimpleNamespace(stop_reason=stop_reason, parsed_output=output,
                           usage=SimpleNamespace(input_tokens=500, output_tokens=80))


def test_anthropic_structured_output_request():
    llm, parse = anthropic_llm(effort="low")
    parse.return_value = parsed(output=SignalVerdict(**VERDICT))
    assert llm.chat_json(MESSAGES, SignalVerdict).detail == "nurse"
    kw = parse.call_args.kwargs
    assert kw["system"] == "You verify signals." and kw["output_format"] is SignalVerdict
    assert [m["role"] for m in kw["messages"]] == ["user", "assistant", "user"]
    assert kw["output_config"] == {"effort": "low"} and "temperature" not in kw
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert last_usage.get().output_tokens == 80


def test_anthropic_no_fallbacks_for_models_without_them():
    llm, parse = anthropic_llm(model="claude-haiku-5-5")
    parse.return_value = parsed(output=SignalVerdict(**VERDICT))
    llm.chat_json(MESSAGES, SignalVerdict)
    assert "fallbacks" not in parse.call_args.kwargs and "output_config" not in parse.call_args.kwargs


@pytest.mark.parametrize("stop_reason,match", [("refusal", "declined"), ("max_tokens", "max_tokens")])
def test_anthropic_refusal_and_truncation(stop_reason, match):
    llm, parse = anthropic_llm()
    parse.return_value = parsed(stop_reason=stop_reason)
    with pytest.raises(LLMError, match=match):
        llm.chat_json(MESSAGES, SignalVerdict)


# --- tracing & health ----------------------------------------------------------------

def test_traced_calls_record_provider_and_tokens(tmp_path):
    store = Store(tmp_path / "t.db")
    llm, parse = anthropic_llm()
    parse.return_value = parsed(output=SignalVerdict(**VERDICT))
    TracedLLM(llm, store, node="signal_agent", model="claude-opus-5-5", provider="anthropic").chat_json(
        MESSAGES, SignalVerdict)
    [call] = store.llm_calls()
    assert (call.provider, call.input_tokens, call.output_tokens) == ("anthropic", 500, 80)


def test_health_reports_per_agent_providers(monkeypatch):
    from tests.test_pipeline import deps
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    d = deps()
    d.settings.llm.outbound = LLMSettings(provider="anthropic", model_id="claude-opus-5-5")
    with patch("src.api.main.ollama_health", return_value={"reachable": True, "missing_models": []}) as oh:
        body = TestClient(create_app(d)).get("/health").json()
    assert body["agents"]["outbound"] == {"provider": "anthropic", "model": "claude-opus-5-5",
                                          "key_env": "ANTHROPIC_API_KEY", "key_configured": True}
    assert oh.call_args.args[0] == ["llama3.2:3b"]  # only local models are checked against Ollama
    assert "sk-ant-secret" not in json.dumps(body)



def test_missing_byok_key_is_a_clear_503(monkeypatch, tmp_path):
    from src.settings import Settings, StorageSettings
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = Settings(storage=StorageSettings(db_path=str(tmp_path / "t.db")))
    s.llm.outbound = LLMSettings(provider="anthropic", model_id="claude-opus-5-5")
    with patch("src.graph.load_settings", return_value=s):
        r = TestClient(create_app()).get("/health")
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
