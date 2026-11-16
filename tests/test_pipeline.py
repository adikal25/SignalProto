from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from src.agents.outbound import DraftCopy
from src.api.main import create_app
from src.crawler import crawl, normalize_base_url
from src.graph import Deps, Pipeline
from src.integrations.booking import CalComBooker, LinkBooker, NoBooker, _spread
from src.integrations.crm import Account, CSVCRM, HubSpotCRM, SignalRecord
from src.llm.ollama_runtime import LLMError
from src.schemas import EvidenceCard, Page, SignalVerdict, Slot
from src.settings import BookingSettings, CrawlSettings, FreshnessSettings, Settings
from src.signals.agent import detect_signals, freshness_weight, score
from src.store import Store
from src.signals.keyword import candidates_from_pages, keyword_classify
from src.tools.web import extract_date, visible_text

SITE = {
    "https://acme.test/": """<html><body><nav>Home | Services | Careers | Contact</nav>
        <h1>Grand opening of our new Apex clinic on May 2!</h1>
        <p>We are open Monday through Friday.</p></body></html>""",
    "https://acme.test/careers": """<html><head><meta property="article:published_time" content="2099-01-01"></head>
        <body><h2>We're hiring a dental hygienist.</h2><p>We are not hiring receptionists at this time.</p></body></html>""",
}


def fake_fetch(url: str) -> str:
    if url not in SITE:
        raise requests.HTTPError("404")
    return SITE[url]


class FakeLLM:
    """Rule-based stand-in for the local model, keyed on sentence content."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def chat_json(self, messages, schema):
        self.calls += 1
        if self.fail:
            raise LLMError("down")
        if schema is DraftCopy:
            return DraftCopy(subject="Congrats on Apex", opener="Saw the Apex opening.", value="We help.")
        sentence = messages[-1]["content"].split("SENTENCE: ", 1)[1]
        if "Grand opening" in sentence:
            return SignalVerdict(signal_type="expansion", strength="strong", detail="Apex", reason="new clinic")
        if "We're hiring" in sentence:
            return SignalVerdict(signal_type="hiring", strength="strong", detail="dental hygienist", reason="role")
        return SignalVerdict(signal_type="none", strength="weak", detail="", reason="not an event")


def deps(**overrides) -> Deps:
    settings = Settings(crawl=CrawlSettings(paths=["/", "/careers", "/jobs"]))
    store = Store(Path(tempfile.mkdtemp()) / "test.db")
    base = dict(settings=settings, store=store, signal_llm=FakeLLM(), outbound_llm=FakeLLM(), booker=NoBooker(),
                crm=MagicMock(), fetcher=fake_fetch)
    base.update(overrides)
    return Deps(**base)


# --- crawling & text -------------------------------------------------------------

def test_normalize_base_url():
    assert normalize_base_url("acme.com") == "https://acme.com/"
    assert normalize_base_url("http://localhost:8000") == "http://localhost:8000/"


def test_crawl_skips_404_and_keys_by_path():
    pages = crawl("acme.test", CrawlSettings(paths=["/", "/careers", "/jobs"]), fetcher=fake_fetch)
    assert [p.path for p in pages] == ["/", "/careers"]


def test_visible_text_separates_blocks():
    text = visible_text("<h1>Now open</h1><p>Visit us</p><script>x=1</script>")
    assert "Now open. Visit us" in text and "x=1" not in text


def test_extract_date_is_timezone_aware():
    dt = extract_date('<meta property="article:published_time" content="2024-03-01">')
    assert dt is not None and dt.tzinfo is not None


# --- keyword baseline ------------------------------------------------------------

@pytest.mark.parametrize("sentence,expected", [
    ("We're hiring two technicians", "hiring"),
    ("Grand opening of our new location", "expansion"),
    ("Call us for a quote", None),
])
def test_keyword_classify(sentence, expected):
    assert keyword_classify(sentence) == expected


def test_candidates_dedup_and_limit():
    page = Page(url="https://a/", path="/", text="We're hiring. We're hiring. Now open. Jobs.")
    assert len(candidates_from_pages([page], limit=10)) == 3
    assert len(candidates_from_pages([page], limit=1)) == 1


# --- signal agent ----------------------------------------------------------------

def test_signal_agent_filters_keyword_false_positives():
    pages = crawl("acme.test", CrawlSettings(paths=["/", "/careers"]), fetcher=fake_fetch)
    cards = detect_signals(pages, FakeLLM(), freshness=FreshnessSettings(), max_candidates=20)
    assert {c.signal_type for c in cards} == {"expansion", "hiring"}
    assert all("Careers |" not in c.snippet and "not hiring" not in c.snippet for c in cards)


def test_signal_agent_survives_llm_outage():
    pages = [Page(url="https://a/", path="/", text="We're hiring a nurse.")]
    assert detect_signals(pages, FakeLLM(fail=True), freshness=FreshnessSettings(), max_candidates=5) == []


def test_freshness_decays_and_floors():
    cfg = FreshnessSettings(weekly_decay=0.5, floor=0.3)
    assert freshness_weight(None, cfg) == 1.0
    assert freshness_weight(datetime.now(timezone.utc) - timedelta(weeks=1), cfg) == pytest.approx(0.5)
    assert freshness_weight(datetime.now(timezone.utc) - timedelta(weeks=52), cfg) == 0.3


def test_score_rewards_strength_and_detail():
    strong = SignalVerdict(signal_type="hiring", strength="strong", detail="RN", reason="r")
    weak = SignalVerdict(signal_type="hiring", strength="weak", detail="", reason="r")
    assert score(strong, None, FreshnessSettings())[0] > score(weak, None, FreshnessSettings())[0]


# --- pipeline graph --------------------------------------------------------------

def test_pipeline_end_to_end_drafts_and_syncs():
    crm = MagicMock()
    crm.record_signal.return_value = "note-1"
    result = Pipeline(deps(crm=crm)).run(Account(domain="acme.test", company="Acme", crm_id="42"), sync_crm=True)
    assert result.top_new is not None and result.email is not None
    assert result.crm_record_id == "note-1"
    record = crm.record_signal.call_args.args[0]
    assert record.account.crm_id == "42" and record.email.subject == "Congrats on Apex"


def test_pipeline_gate_blocks_outbound():
    d = deps()
    d.settings.gate.min_confidence = 1.0
    d.settings.freshness.weekly_decay = 0.5
    result = Pipeline(d).run(Account(domain="acme.test"))
    assert result.email is None and result.skipped_reason == "below_confidence_gate"


def test_pipeline_no_pages():
    result = Pipeline(deps(fetcher=lambda url: (_ for _ in ()).throw(requests.ConnectionError()))).run(
        Account(domain="down.test"))
    assert result.skipped_reason == "no_pages_fetched"


def test_outbound_falls_back_to_template_and_keeps_link():
    d = deps(outbound_llm=FakeLLM(fail=True), booker=LinkBooker("https://cal.com/alex/15min"))
    result = Pipeline(d).run(Account(domain="acme.test", company="Acme"))
    assert result.email is not None and "https://cal.com/alex/15min" in result.email.body


def test_outbound_offers_real_slots_only():
    slot = Slot(start=datetime(2099, 1, 2, 15, tzinfo=timezone.utc))
    booker = MagicMock()
    booker.available_slots.return_value = [slot]
    booker.booking_link.return_value = None
    result = Pipeline(deps(booker=booker)).run(Account(domain="acme.test", company="Acme"))
    assert result.email.call_to_action == "Would any of these work for a 15-minute call: Fri Jan 2, 10:00 AM EST?"
    assert result.email.call_to_action in result.email.body


# --- integrations ----------------------------------------------------------------

def _resp(json_body):
    r = MagicMock()
    r.json.return_value = json_body
    r.raise_for_status.return_value = None
    return r


def test_calcom_slots_and_booking():
    booker = CalComBooker("cal_test", 7, BookingSettings(slots_to_offer=2))
    slots_body = {"status": "success", "data": {
        "2099-01-02": [{"start": "2099-01-02T09:00:00.000-05:00"}, {"start": "2099-01-02T10:00:00.000-05:00"}],
        "2099-01-03": [{"start": "2099-01-03T09:00:00.000-05:00"}]}}
    with patch("src.integrations.booking.requests.get", return_value=_resp(slots_body)) as get:
        slots = booker.available_slots()
    assert [s.start.day for s in slots] == [2, 3]  # spread across days
    assert get.call_args.kwargs["headers"]["cal-api-version"] == "2024-09-04"

    book_body = {"status": "success", "data": {"uid": "abc", "start": "2099-01-02T14:00:00Z", "status": "accepted"}}
    with patch("src.integrations.booking.requests.post", return_value=_resp(book_body)) as post:
        booking = booker.book(slots[0].start, "Pat", "pat@acme.test", "America/New_York")
    assert booking.id == "abc"
    body = post.call_args.kwargs["json"]
    assert body["start"] == "2099-01-02T14:00:00Z" and body["eventTypeId"] == 7


def test_spread_fills_from_same_day_when_needed():
    base = datetime(2099, 1, 2, 9, tzinfo=timezone.utc)
    slots = [Slot(start=base + timedelta(hours=i)) for i in range(3)]
    assert len(_spread(slots, 2)) == 2


def _card() -> EvidenceCard:
    now = datetime.now(timezone.utc)
    return EvidenceCard(signal_type="hiring", canonical_url="https://a.test/careers", first_seen=now,
                        last_seen=now, snippet="We're hiring <RNs>", confidence=0.9, explain="x")


def test_hubspot_lists_companies_with_paging_and_writes_note():
    crm = HubSpotCRM("tok")
    pages = [
        _resp({"results": [{"id": "1", "properties": {"domain": "a.test", "name": "A"}},
                           {"id": "2", "properties": {"domain": None, "name": "NoDomain"}}],
               "paging": {"next": {"after": "2"}}}),
        _resp({"results": [{"id": "3", "properties": {"domain": "c.test", "name": "C"}}]}),
    ]
    with patch("src.integrations.crm.requests.get", side_effect=pages):
        accounts = crm.list_accounts(10)
    assert [a.crm_id for a in accounts] == ["1", "3"]

    with patch("src.integrations.crm.requests.post", return_value=_resp({"id": "n1"})) as post:
        note_id = crm.record_signal(SignalRecord(account=accounts[0], card=_card()))
    assert note_id == "n1"
    body = post.call_args.kwargs["json"]
    assert body["associations"][0]["types"][0]["associationTypeId"] == 190
    assert "&lt;RNs&gt;" in body["properties"]["hs_note_body"]


def test_csv_crm_roundtrip(tmp_path):
    accounts_csv = tmp_path / "a.csv"
    accounts_csv.write_text("domain,company\nacme.test, Acme \n")
    crm = CSVCRM(str(accounts_csv), str(tmp_path / "out.jsonl"))
    [acct] = crm.list_accounts(5)
    assert acct.company == "Acme"
    crm.record_signal(SignalRecord(account=acct, card=_card()))
    assert "acme.test" in (tmp_path / "out.jsonl").read_text()


# --- REST API --------------------------------------------------------------------

def test_api_detect_run_and_booking_errors():
    client = TestClient(create_app(deps()))
    r = client.post("/signals/detect", json={"domain": "acme.test"})
    assert r.status_code == 200 and r.json()["pages_crawled"] == 2 and len(r.json()["signals"]) == 2

    r = client.post("/pipeline/run", json={"domain": "acme.test", "company": "Acme"})
    assert r.status_code == 200 and r.json()["email"]["subject"] == "Congrats on Apex"

    r = client.post("/meetings/book", json={"start": "2099-01-02T14:00:00Z", "name": "Pat", "email": "pat@acme.com"})
    assert r.status_code == 501

    r = client.post("/meetings/book", json={"start": "2099-01-02T14:00:00Z", "name": "Pat", "email": "not-an-email"})
    assert r.status_code == 422


def test_api_watch_pulls_crm_accounts_and_reports_new_signals():
    crm = MagicMock()
    crm.list_accounts.return_value = [Account(domain="acme.test", company="Acme", crm_id="9")]
    crm.record_signal.return_value = "note-9"
    client = TestClient(create_app(deps(crm=crm)))
    r = client.post("/watch/run", json={"sync_crm": True})
    assert r.status_code == 200
    body = r.json()
    assert body["accounts_checked"] == 1 and len(body["new_signals"]) == 2
    assert crm.record_signal.call_count == 1
    # second sweep: nothing changed, nothing new, no second CRM write
    body = client.post("/watch/run", json={"sync_crm": True}).json()
    assert body["new_signals"] == [] and body["pages_changed"] == 0
    assert crm.record_signal.call_count == 1
