"""Storage, change detection, scoring, tracing and the eval gate."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from eval.run_eval import gate
from src.api.main import create_app
from src.graph import Deps, Pipeline
from src.integrations.booking import NoBooker
from src.integrations.crm import Account
from src.llm.ollama_runtime import LLMError
from src.llm.tracing import CassetteMiss, RecordingLLM, ReplayLLM, TracedLLM, prompt_version
from src.schemas import SignalVerdict
from src.scoring import score_prospect
from src.settings import CrawlSettings, ICPSettings, ScoringSettings, Settings
from src.signals.changes import fingerprint, normalize_snippet
from src.store import AccountRow, SignalRow, Store
from src.watch import run_watch
from tests.test_pipeline import FakeLLM, _card

HOME = "<h1>Grand opening of our new Apex clinic on May 2!</h1>"
CAREERS = "<h2>We're hiring a dental hygienist.</h2>"


class Site:
    """Mutable fake website so tests can change pages between runs."""

    def __init__(self, pages: dict[str, str]):
        self.pages = pages

    def __call__(self, url: str) -> str:
        path = "/" + url.split("acme.test/", 1)[1]
        if path not in self.pages:
            raise requests.HTTPError("404")
        return f"<html><body>{self.pages[path]}</body></html>"


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "t.db")


def make_pipeline(store: Store, site: Site, crm=None, signal_llm=None) -> Pipeline:
    settings = Settings(crawl=CrawlSettings(paths=["/", "/careers"]))
    llm = signal_llm or TracedLLM(FakeLLM(), store, node="signal_agent", model="fake")
    return Pipeline(Deps(settings=settings, store=store, signal_llm=llm, outbound_llm=FakeLLM(),
                         booker=NoBooker(), crm=crm or MagicMock(list_accounts=MagicMock(return_value=[])),
                         fetcher=site))


ACME = Account(domain="acme.test", company="Acme")


# --- store -----------------------------------------------------------------------

def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "t.db"
    Store(path).upsert_account(AccountRow(domain="a.test", company="A", employees=10))
    s = Store(path)  # reopening must not re-run migrations or lose data
    s.upsert_account(AccountRow(domain="a.test"))  # None fields keep stored values
    assert s.get_account("a.test").employees == 10


# --- change detection ------------------------------------------------------------

def test_fingerprint_ignores_dates_and_punctuation():
    a, b = _card(), _card()
    a.snippet, b.snippet = "Grand opening June 14!", "grand opening, June 21"
    assert normalize_snippet(a.snippet) == "grand opening june"
    assert fingerprint("acme.test", a) == fingerprint("ACME.test", b)


def test_unchanged_site_skips_llm_and_drafts_nothing(store):
    site = Site({"/": HOME, "/careers": CAREERS})
    signal_llm = FakeLLM()
    p = make_pipeline(store, site, signal_llm=signal_llm)
    first = p.run(ACME)
    assert len(first.new_signals) == 2 and first.email is not None
    calls_after_first = signal_llm.calls

    second = p.run(ACME)
    assert second.skipped_reason == "no_page_changes" and second.email is None
    assert signal_llm.calls == calls_after_first  # no LLM calls for unchanged pages
    assert len(store.outreach("acme.test")) == 1


def test_force_rereads_but_known_signals_are_not_new(store):
    p = make_pipeline(store, Site({"/": HOME}))
    p.run(ACME)
    again = p.run(ACME, force=True)
    assert len(again.detected) == 1 and again.new_signals == []
    assert again.skipped_reason == "no_new_signals"


def test_new_signal_triggers_outreach_and_removed_one_goes_stale(store):
    site = Site({"/": HOME, "/careers": "<p>Benefits and culture.</p>"})
    p = make_pipeline(store, site)
    p.run(ACME)
    assert [s.signal_type for s in store.signals("acme.test")] == ["expansion"]

    site.pages["/careers"] = CAREERS          # hiring appears
    site.pages["/"] = "<h1>Welcome!</h1>"     # opening announcement removed
    r = p.run(ACME)
    assert [d.card.signal_type for d in r.new_signals] == ["hiring"]
    assert r.email is not None and r.stale_count == 1
    assert [s.signal_type for s in store.signals("acme.test", status="stale")] == ["expansion"]


def test_watch_alerts_only_on_new_signals(store):
    site = Site({"/": HOME})
    p = make_pipeline(store, site)
    p.deps.settings.alerts.webhook_url = "https://hooks.example/x"
    store.upsert_account(AccountRow(domain="acme.test", company="Acme"))
    with patch("src.watch.requests.post") as post:
        report = run_watch(p)
        assert report.alert_sent and len(report.new_signals) == 1
        assert "1 new signal" in post.call_args.kwargs["json"]["text"]
        report = run_watch(p)
        assert not report.alert_sent and post.call_count == 1


def test_watch_isolates_per_account_failures(store):
    p = make_pipeline(store, Site({"/": HOME}))
    store.upsert_account(AccountRow(domain="acme.test"))
    store.upsert_account(AccountRow(domain="broken.test"))
    with patch.object(p, "run", side_effect=[RuntimeError("boom"), p.run(ACME)]):
        report = run_watch(p)
    assert set(report.errors) == {"acme.test"} and report.accounts_checked == 2


# --- scoring ---------------------------------------------------------------------

def _sig(t: str, conf: float, age_days: int = 0) -> SignalRow:
    seen = datetime.now(timezone.utc) - timedelta(days=age_days)
    return SignalRow(fingerprint=f"{t}{conf}{age_days}", domain="a", signal_type=t, url="https://a/",
                     snippet="s", detail=None, confidence=conf, explain="", first_seen=seen,
                     last_seen=seen, status="active")


def test_score_orders_by_strength_diversity_recency_and_fit():
    cfg = ScoringSettings(icp=ICPSettings(min_employees=5, max_employees=200, verticals=["dental"]))
    acct = AccountRow(domain="a", employees=40, vertical="Dental clinic")
    both = score_prospect(acct, [_sig("expansion", 0.9), _sig("hiring", 0.9)], cfg)
    one = score_prospect(acct, [_sig("expansion", 0.9)], cfg)
    old = score_prospect(acct, [_sig("expansion", 0.9, age_days=120)], cfg)
    off_icp = score_prospect(AccountRow(domain="a", employees=5000, vertical="Banking"), [_sig("expansion", 0.9)], cfg)
    assert both.total > one.total > old.total
    assert one.total > off_icp.total
    assert score_prospect(acct, [], cfg).total == 0


def test_unknown_size_and_no_icp_are_neutral():
    s = score_prospect(AccountRow(domain="a"), [_sig("hiring", 0.8)], ScoringSettings())
    assert s.size == 0.5 and s.fit == 0.5


# --- tracing & replay ------------------------------------------------------------

def test_pipeline_llm_calls_are_traced_with_run_id(store):
    r = make_pipeline(store, Site({"/": HOME})).run(ACME)
    calls = store.llm_calls(trace_id=r.trace_id)
    assert calls and all(c.node == "signal_agent" and c.model == "fake" for c in calls)
    assert "expansion" in calls[-1].output


def test_traced_llm_logs_errors(store):
    llm = TracedLLM(FakeLLM(fail=True), store, node="n", model="m")
    with pytest.raises(LLMError):
        llm.chat_json([{"role": "user", "content": "SENTENCE: x"}], SignalVerdict)
    [call] = store.llm_calls()
    assert call.error and call.output is None


def test_prompt_version_tracks_prompt_not_input():
    a = [{"role": "system", "content": "v1"}, {"role": "user", "content": "x"}]
    b = [{"role": "system", "content": "v1"}, {"role": "user", "content": "y"}]
    c = [{"role": "system", "content": "v2"}, {"role": "user", "content": "x"}]
    assert prompt_version(a) == prompt_version(b) != prompt_version(c)


def test_record_then_replay_and_miss_on_prompt_change(tmp_path):
    cassette = tmp_path / "c.jsonl"
    msgs = [{"role": "system", "content": "v1"}, {"role": "user", "content": "SENTENCE: We're hiring"}]
    recorded = RecordingLLM(FakeLLM(), cassette).chat_json(msgs, SignalVerdict)
    assert ReplayLLM(cassette).chat_json(msgs, SignalVerdict) == recorded
    with pytest.raises(CassetteMiss):
        ReplayLLM(cassette).chat_json([{"role": "system", "content": "v2"}, msgs[1]], SignalVerdict)


def test_eval_gate():
    base = {"precision": 0.9, "recall": 0.9, "f1": 0.9}
    assert gate({"precision": 0.88, "recall": 0.9, "f1": 0.89}, base, 0.03) == 0
    assert gate({"precision": 0.80, "recall": 0.9, "f1": 0.85}, base, 0.03) == 1
    assert gate(base, None, 0.03) == 1


# --- API -------------------------------------------------------------------------

def test_api_prospects_detail_traces_and_dashboard(store):
    p = make_pipeline(store, Site({"/": HOME, "/careers": CAREERS}))
    client = TestClient(create_app(p.deps))
    run = client.post("/pipeline/run", json={"domain": "acme.test", "company": "Acme", "employees": 30}).json()
    assert run["pages_changed"] == 2 and all(s["is_new"] for s in run["signals"])

    [prospect] = client.get("/prospects").json()
    assert prospect["account"]["employees"] == 30 and prospect["score"]["total"] > 0

    detail = client.get("/prospects/detail", params={"domain": "acme.test"}).json()
    assert len(detail["outreach"]) == 1
    assert client.get("/prospects/detail", params={"domain": "nope"}).status_code == 404

    traces = client.get("/traces", params={"trace_id": run["trace_id"]}).json()
    assert len(traces) == 2

    page = client.get("/")
    assert page.status_code == 200 and "SignalProto" in page.text
