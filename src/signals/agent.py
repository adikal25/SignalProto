"""Signal agent: keyword recall stage -> local-LLM verification -> scored evidence cards.

Keyword matching fires on "careers" in a footer, "apply sunscreen", "we are not hiring",
"opened in 1998". The agent sends each keyword hit, with its surrounding context, to a
local model constrained to a JSON schema, and keeps only current, first-party events.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from src.llm.base import LLM, LLMError
from src.schemas import Candidate, EvidenceCard, Page, SignalVerdict
from src.settings import FreshnessSettings
from src.signals.keyword import candidates_from_pages

log = logging.getLogger(__name__)

SYSTEM = """You verify B2B sales triggers found on a small business's own website.
Decide whether the SENTENCE announces a CURRENT, FIRST-PARTY event of one of these types:
- hiring: the business itself is actively recruiting now (open roles, "we're hiring", a job posting).
- expansion: the business itself is opening, has just opened, or will open a new location, office, clinic, store or service area.
Return signal_type "none" when:
- the keyword is used in another sense ("apply sunscreen", "open 9am-5pm", "book a job", "jobs done right", "expand your smile");
- the statement is negated or closed ("not hiring", "positions filled", "closed our Austin office");
- it is historical background ("founded in 1998", "opened our doors in 2005");
- it is generic navigation or boilerplate with no event ("Careers", "Privacy | Careers | Contact");
- it is about someone else (a client, a competitor, a news story about another company).
strength is "strong" for a specific, dated or concrete announcement (a named role, a named city or address, a date),
otherwise "weak". detail is the role or the new location if the text names one, else ""."""

EXAMPLES: list[tuple[str, str, SignalVerdict]] = [
    (
        "Grand opening of our second clinic in Cary on March 3!",
        "Grand opening of our second clinic in Cary on March 3! Book your first visit today.",
        SignalVerdict(signal_type="expansion", strength="strong", detail="Cary", reason="Announces a new clinic location with a date."),
    ),
    (
        "Home | Services | Careers | Contact",
        "Home | Services | Careers | Contact",
        SignalVerdict(signal_type="none", strength="weak", detail="", reason="Navigation link, not a hiring announcement."),
    ),
    (
        "We're hiring a front desk coordinator and two dental hygienists.",
        "Join our growing team. We're hiring a front desk coordinator and two dental hygienists. Apply below.",
        SignalVerdict(signal_type="hiring", strength="strong", detail="front desk coordinator, dental hygienists", reason="Names specific open roles."),
    ),
    (
        "Proudly serving Raleigh since we opened in 1998.",
        "Family owned. Proudly serving Raleigh since we opened in 1998.",
        SignalVerdict(signal_type="none", strength="weak", detail="", reason="Historical opening date, not a current event."),
    ),
]


def _user_message(sentence: str, context: str, url: str) -> str:
    return f"URL: {url}\nCONTEXT: {context}\nSENTENCE: {sentence}"


def build_messages(c: Candidate) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM}]
    for sentence, context, verdict in EXAMPLES:
        messages.append({"role": "user", "content": _user_message(sentence, context, "https://example.com/")})
        messages.append({"role": "assistant", "content": verdict.model_dump_json()})
    messages.append({"role": "user", "content": _user_message(c.sentence, c.context, c.url)})
    return messages


def verify(llm: LLM, candidate: Candidate) -> Optional[SignalVerdict]:
    try:
        return llm.chat_json(build_messages(candidate), SignalVerdict)
    except LLMError as e:
        log.warning("verification failed for %r: %s", candidate.sentence[:80], e)
        return None


def freshness_weight(published_at: Optional[datetime], cfg: FreshnessSettings) -> float:
    if published_at is None:
        return 1.0
    weeks = max(0.0, (datetime.now(timezone.utc) - published_at).days / 7)
    return max(cfg.floor, cfg.weekly_decay ** weeks)


def score(verdict: SignalVerdict, published_at: Optional[datetime], cfg: FreshnessSettings) -> tuple[float, str]:
    base = 0.75 if verdict.strength == "strong" else 0.5
    reasons = [verdict.strength]
    if verdict.detail.strip():
        base += 0.15
        reasons.append("specific_detail")
    w = freshness_weight(published_at, cfg)
    reasons.append(f"freshness={w:.2f}")
    return round(min(1.0, base * w), 3), "; ".join(reasons + [verdict.reason])


def detect_signals(
    pages: list[Page], llm: LLM, *, freshness: FreshnessSettings, max_candidates: int
) -> list[EvidenceCard]:
    """Return verified signals, strongest first."""
    now = datetime.now(timezone.utc)
    cards: list[EvidenceCard] = []
    for c in candidates_from_pages(pages, limit=max_candidates):
        verdict = verify(llm, c)
        if verdict is None or verdict.signal_type == "none":
            continue
        conf, explain = score(verdict, c.published_at, freshness)
        cards.append(EvidenceCard(
            signal_type=verdict.signal_type,
            canonical_url=c.url,
            first_seen=c.published_at or now,
            last_seen=now,
            snippet=c.sentence[:320],
            detail=verdict.detail.strip() or None,
            confidence=conf,
            explain=explain[:500],
        ))
    return sorted(cards, key=lambda card: card.confidence, reverse=True)
