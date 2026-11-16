"""FastAPI REST API + dashboard in front of the local-model pipeline.

Run:  uvicorn src.api.main:app --reload
UI:   http://localhost:8000/        Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field

from src.crawler import crawl
from src.graph import DetectedSignal, Deps, Pipeline
from src.integrations.booking import BookingNotSupported
from src.integrations.crm import Account
from src.llm.ollama_runtime import ollama_health
from src.schemas import Booking, EmailDraft, EvidenceCard, Slot
from src.signals.agent import detect_signals
from src.store import LLMCallRow, OutreachRow, SignalRow
from src.watch import Prospect, WatchReport, ranked_prospects, run_watch

DASHBOARD = Path(__file__).resolve().parents[1] / "web" / "index.html"


class DetectRequest(BaseModel):
    domain: str = Field(examples=["acmedental.com"])
    company: Optional[str] = None


class DetectResponse(BaseModel):
    domain: str
    pages_crawled: int
    signals: list[EvidenceCard]


class RunRequest(DetectRequest):
    crm_id: Optional[str] = None
    employees: Optional[int] = Field(default=None, ge=0)
    vertical: Optional[str] = None
    sync_crm: bool = False
    force: bool = Field(default=False, description="Re-read every page even if unchanged since last run.")


class RunResponse(BaseModel):
    domain: str
    company: Optional[str]
    trace_id: str
    pages_crawled: int
    pages_changed: int
    signals: list[DetectedSignal]
    stale_signals: int
    email: Optional[EmailDraft]
    crm_record_id: Optional[str]
    skipped_reason: Optional[str]


class WatchRequest(BaseModel):
    sync_crm: bool = False


class ProspectDetail(BaseModel):
    prospect: Prospect
    outreach: list[OutreachRow]
    stale_signals: list[SignalRow]


class BookRequest(BaseModel):
    start: datetime
    name: str
    email: EmailStr
    timezone: str = "America/New_York"


def create_app(deps: Deps | None = None) -> FastAPI:
    app = FastAPI(title="SignalProto", version="0.3.0",
                  description="Signal-to-meeting prospecting agents on local models.")
    state: dict[str, Pipeline] = {}

    def pipeline() -> Pipeline:
        if "p" not in state:
            state["p"] = Pipeline(deps or Deps.from_settings())
        return state["p"]

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(DASHBOARD)

    @app.get("/health")
    def health(p: Pipeline = Depends(pipeline)) -> dict:
        s = p.deps.settings
        return {"status": "ok", "ollama": ollama_health([s.llm.signal.model_id, s.llm.outbound.model_id]),
                "models": {"signal": s.llm.signal.model_id, "outbound": s.llm.outbound.model_id},
                "crm": s.crm.provider, "booking": s.booking.provider,
                "alerts": bool(s.alerts.webhook_url)}

    @app.post("/signals/detect", response_model=DetectResponse)
    def detect(req: DetectRequest, p: Pipeline = Depends(pipeline)) -> DetectResponse:
        """Stateless: crawl + signal agent only. Nothing is stored, no email, no CRM write."""
        s = p.deps.settings
        pages = crawl(req.domain, s.crawl, fetcher=p.deps.fetcher)
        cards = detect_signals(pages, p.deps.signal_llm, freshness=s.freshness,
                               max_candidates=s.signals.max_candidates_per_domain)
        return DetectResponse(domain=req.domain, pages_crawled=len(pages), signals=cards)

    @app.post("/pipeline/run", response_model=RunResponse)
    def run(req: RunRequest, p: Pipeline = Depends(pipeline)) -> RunResponse:
        """Track the account and run: changed pages -> signals -> gate on NEW signals -> draft -> CRM."""
        account = Account(domain=req.domain, company=req.company, crm_id=req.crm_id,
                          employees=req.employees, vertical=req.vertical)
        r = p.run(account, sync_crm=req.sync_crm, force=req.force)
        return RunResponse(domain=req.domain, company=req.company, trace_id=r.trace_id,
                           pages_crawled=len(r.pages), pages_changed=len(r.changed_urls), signals=r.detected,
                           stale_signals=r.stale_count, email=r.email, crm_record_id=r.crm_record_id,
                           skipped_reason=r.skipped_reason)

    @app.post("/watch/run", response_model=WatchReport)
    def watch(req: WatchRequest, p: Pipeline = Depends(pipeline)) -> WatchReport:
        """Pull CRM accounts, re-check every tracked account, alert on new signals."""
        return run_watch(p, sync_crm=req.sync_crm)

    @app.get("/prospects", response_model=list[Prospect])
    def prospects(include_empty: bool = False, p: Pipeline = Depends(pipeline)) -> list[Prospect]:
        """Tracked accounts ranked by tier, then priority (geometric mean of fit and intent)."""
        return ranked_prospects(p, include_empty=include_empty)

    @app.get("/prospects/detail", response_model=ProspectDetail)
    def prospect_detail(domain: str = Query(...), p: Pipeline = Depends(pipeline)) -> ProspectDetail:
        match = [x for x in ranked_prospects(p, include_empty=True) if x.account.domain == domain]
        if not match:
            raise HTTPException(status_code=404, detail=f"unknown account {domain!r}")
        store = p.deps.store
        stale = [s for s in store.signals(domain, status="stale")]
        return ProspectDetail(prospect=match[0], outreach=store.outreach(domain), stale_signals=stale)

    @app.post("/outreach/{outreach_id}/sent", response_model=OutreachRow)
    def mark_sent(outreach_id: int, p: Pipeline = Depends(pipeline)) -> OutreachRow:
        """Record that a draft was sent; starts the account's contact cooldown in scoring."""
        row = p.deps.store.mark_outreach_sent(outreach_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown outreach id {outreach_id}")
        return row

    @app.get("/traces", response_model=list[LLMCallRow])
    def traces(trace_id: Optional[str] = None, limit: int = Query(default=100, ge=1, le=1000),
               p: Pipeline = Depends(pipeline)) -> list[LLMCallRow]:
        """Logged LLM calls: node, model, prompt version, latency, output or error."""
        return p.deps.store.llm_calls(trace_id=trace_id, limit=limit)

    @app.get("/meetings/slots", response_model=list[Slot])
    def slots(p: Pipeline = Depends(pipeline)) -> list[Slot]:
        return p.deps.booker.available_slots()

    @app.post("/meetings/book", response_model=Booking, status_code=201)
    def book(req: BookRequest, p: Pipeline = Depends(pipeline)) -> Booking:
        try:
            return p.deps.booker.book(req.start, req.name, req.email, req.timezone)
        except BookingNotSupported as e:
            raise HTTPException(status_code=501, detail=str(e))

    return app


app = create_app()
