"""LangGraph orchestration, stateful across runs:

crawl (skip unchanged pages) -> signal agent -> diff vs. known signals -> gate on NEW signals
-> outbound agent -> CRM sync. Every run gets a trace id that tags its LLM calls.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from src.agents.outbound import draft_email
from src.crawler import Fetcher, crawl
from src.integrations.booking import Booker, make_booker
from src.integrations.crm import Account, CRMAdapter, SignalRecord, make_crm
from src.llm.ollama_runtime import LLM, OllamaChat, OllamaConfig
from src.llm.tracing import TracedLLM, current_trace_id
from src.schemas import EmailDraft, EvidenceCard, Page
from src.settings import LLMSettings, Settings, load_settings
from src.signals.agent import detect_signals
from src.signals.changes import content_hash, fingerprint
from src.store import AccountRow, Store

log = logging.getLogger(__name__)


class DetectedSignal(BaseModel):
    fingerprint: str
    card: EvidenceCard
    is_new: bool


class PipelineState(BaseModel):
    account: Account
    trace_id: str = ""
    sync_crm: bool = False
    force: bool = False
    pages: list[Page] = []
    changed_urls: list[str] = []
    detected: list[DetectedSignal] = []
    stale_count: int = 0
    email: Optional[EmailDraft] = None
    crm_record_id: Optional[str] = None
    skipped_reason: Optional[str] = None

    @property
    def cards(self) -> list[EvidenceCard]:
        return [d.card for d in self.detected]

    @property
    def new_signals(self) -> list[DetectedSignal]:
        return [d for d in self.detected if d.is_new]

    @property
    def top_new(self) -> Optional[DetectedSignal]:
        return self.new_signals[0] if self.new_signals else None


def _traced(cfg: LLMSettings, store: Store, node: str) -> LLM:
    inner = OllamaChat(OllamaConfig(cfg.model_id, cfg.max_new_tokens, cfg.temperature))
    return TracedLLM(inner, store, node=node, model=cfg.model_id)


@dataclass
class Deps:
    settings: Settings
    store: Store
    signal_llm: LLM
    outbound_llm: LLM
    booker: Booker
    crm: CRMAdapter
    fetcher: Optional[Fetcher] = field(default=None)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Deps":
        s = settings or load_settings()
        store = Store(s.storage.db_path)
        return cls(settings=s, store=store,
                   signal_llm=_traced(s.llm.signal, store, "signal_agent"),
                   outbound_llm=_traced(s.llm.outbound, store, "outbound_agent"),
                   booker=make_booker(s.booking), crm=make_crm(s.crm))


def build_graph(deps: Deps):
    s, store = deps.settings, deps.store

    def crawl_node(state: PipelineState) -> dict:
        domain = state.account.domain
        pages = crawl(domain, s.crawl, fetcher=deps.fetcher)
        if not pages:
            return {"pages": [], "skipped_reason": "no_pages_fetched"}
        hashes = {p.url: content_hash(p.text) for p in pages}
        previous = store.snapshot_hashes(domain)
        changed = [u for u, h in hashes.items() if state.force or previous.get(u) != h]
        store.touch_signals_on_pages(domain, [u for u in hashes if u not in changed])
        store.save_snapshots(domain, hashes)
        return {"pages": pages, "changed_urls": changed,
                "skipped_reason": None if changed else "no_page_changes"}

    def signal_node(state: PipelineState) -> dict:
        domain = state.account.domain
        changed_pages = [p for p in state.pages if p.url in state.changed_urls]
        if not changed_pages:
            return {"detected": []}
        cards = detect_signals(changed_pages, deps.signal_llm, freshness=s.freshness,
                               max_candidates=s.signals.max_candidates_per_domain)
        known = store.known_fingerprints(domain)
        detected: list[DetectedSignal] = []
        for card in cards:
            fp = fingerprint(domain, card)
            if any(d.fingerprint == fp for d in detected):
                continue
            detected.append(DetectedSignal(fingerprint=fp, card=card, is_new=fp not in known))
            store.upsert_signal(domain, fp, card)
        stale = store.mark_stale(domain, state.changed_urls, {d.fingerprint for d in detected})
        return {"detected": detected, "stale_count": stale}

    def gate(state: PipelineState) -> str:
        top = state.top_new
        return "outbound" if top and top.card.confidence >= s.gate.min_confidence else "below_gate"

    def below_gate_node(state: PipelineState) -> dict:
        if state.skipped_reason:
            reason = state.skipped_reason
        elif not state.detected:
            reason = "no_verified_signal"
        elif not state.new_signals:
            reason = "no_new_signals"
        else:
            reason = "below_confidence_gate"
        return {"skipped_reason": reason}

    def outbound_node(state: PipelineState) -> dict:
        top = state.top_new
        assert top is not None
        try:
            slots = deps.booker.available_slots()
        except Exception as e:  # a booking outage should not block the draft
            log.warning("slot lookup failed: %s", e)
            slots = []
        company = state.account.company or state.account.domain
        email = draft_email(deps.outbound_llm, top.card, company, s.sender, slots,
                            deps.booker.booking_link(), s.booking.timezone)
        store.add_outreach(state.account.domain, top.fingerprint, email)
        return {"email": email}

    def crm_node(state: PipelineState) -> dict:
        top = state.top_new
        if not state.sync_crm or top is None:
            return {"crm_record_id": None}
        record = SignalRecord(account=state.account, card=top.card, email=state.email)
        return {"crm_record_id": deps.crm.record_signal(record)}

    g = StateGraph(PipelineState)
    g.add_node("crawl", crawl_node)
    g.add_node("signal_agent", signal_node)
    g.add_node("below_gate", below_gate_node)
    g.add_node("outbound_agent", outbound_node)
    g.add_node("crm_sync", crm_node)
    g.set_entry_point("crawl")
    g.add_edge("crawl", "signal_agent")
    g.add_conditional_edges("signal_agent", gate, {"outbound": "outbound_agent", "below_gate": "below_gate"})
    g.add_edge("outbound_agent", "crm_sync")
    g.add_edge("below_gate", "crm_sync")
    g.add_edge("crm_sync", END)
    return g.compile()


class Pipeline:
    def __init__(self, deps: Deps):
        self.deps = deps
        self.graph = build_graph(deps)

    def run(self, account: Account, *, sync_crm: bool = False, force: bool = False) -> PipelineState:
        self.deps.store.upsert_account(AccountRow(**account.model_dump()))
        trace_id = uuid.uuid4().hex[:16]
        token = current_trace_id.set(trace_id)
        try:
            result = self.graph.invoke(PipelineState(account=account, trace_id=trace_id,
                                                     sync_crm=sync_crm, force=force))
        finally:
            current_trace_id.reset(token)
        return PipelineState.model_validate(result)
