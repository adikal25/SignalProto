"""The product loop: re-check every tracked account, act only on what is new, alert on it.

Schedule it with cron / launchd, e.g. daily at 7am:
    0 7 * * *  cd /path/to/SignalProto && .venv/bin/python -m src.app watch --sync-crm
"""
from __future__ import annotations

import logging
from typing import Optional

import requests
from pydantic import BaseModel

from src.graph import Pipeline
from src.integrations.crm import Account
from src.schemas import EmailDraft, EvidenceCard
from src.scoring import ScoreBreakdown, score_prospect
from src.store import AccountRow, SignalRow

log = logging.getLogger(__name__)


class NewSignal(BaseModel):
    domain: str
    company: Optional[str]
    card: EvidenceCard
    email: Optional[EmailDraft]


class WatchReport(BaseModel):
    accounts_checked: int
    pages_changed: int
    new_signals: list[NewSignal]
    stale_signals: int
    errors: dict[str, str]
    alert_sent: bool


class Prospect(BaseModel):
    account: AccountRow
    score: ScoreBreakdown
    signals: list[SignalRow]


def ranked_prospects(pipeline: Pipeline, include_empty: bool = False) -> list[Prospect]:
    store, cfg = pipeline.deps.store, pipeline.deps.settings.scoring
    out = []
    for account in store.list_accounts():
        signals = store.signals(account.domain)
        if signals or include_empty:
            out.append(Prospect(account=account, score=score_prospect(account, signals, cfg), signals=signals))
    return sorted(out, key=lambda p: p.score.total, reverse=True)


def run_watch(pipeline: Pipeline, *, sync_crm: bool = False, crm_limit: int = 200) -> WatchReport:
    deps = pipeline.deps
    try:
        for a in deps.crm.list_accounts(crm_limit):  # pull any new accounts from the CRM first
            deps.store.upsert_account(AccountRow(**a.model_dump()))
    except Exception as e:
        log.warning("CRM account pull failed, continuing with stored accounts: %s", e)

    accounts = deps.store.list_accounts()
    new: list[NewSignal] = []
    errors: dict[str, str] = {}
    pages_changed = stale = 0
    for row in accounts:
        account = Account(**row.model_dump(exclude={"updated_at"}))
        try:
            result = pipeline.run(account, sync_crm=sync_crm)
        except Exception as e:  # one bad site must not stop the sweep
            log.exception("watch failed for %s", row.domain)
            errors[row.domain] = str(e)
            continue
        pages_changed += len(result.changed_urls)
        stale += result.stale_count
        for i, d in enumerate(result.new_signals):
            new.append(NewSignal(domain=row.domain, company=row.company, card=d.card,
                                 email=result.email if i == 0 else None))

    alert_sent = _send_alert(deps.settings.alerts.webhook_url, new)
    return WatchReport(accounts_checked=len(accounts), pages_changed=pages_changed, new_signals=new,
                       stale_signals=stale, errors=errors, alert_sent=alert_sent)


def _send_alert(url: Optional[str], new: list[NewSignal]) -> bool:
    if not url or not new:
        return False
    lines = [f"*{len(new)} new signal(s)*"] + [
        f"• {n.company or n.domain}: {n.card.signal_type} ({n.card.confidence:.2f}) - \"{n.card.snippet[:140]}\""
        for n in new[:20]
    ]
    try:  # `text` renders in Slack; the structured list is there for Zapier / n8n
        payload = {"text": "\n".join(lines), "signals": [n.model_dump(mode="json") for n in new]}
        requests.post(url, json=payload, timeout=15).raise_for_status()
        return True
    except requests.RequestException as e:
        log.warning("alert webhook failed: %s", e)
        return False
