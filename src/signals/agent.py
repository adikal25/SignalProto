"""Signal agent: keyword recall stage -> local-LLM verification -> scored evidence cards.

Keyword matching fires on "careers" in a footer, "apply sunscreen", "we are not hiring",
"opened in 1998". The agent sends each keyword hit, with its surrounding context, to a
local model constrained to a JSON schema, and keeps only current, first-party events.
The types it looks for, their keywords and the few-shot examples come from settings.signals.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal, Optional

from pydantic import create_model

from src.llm.base import LLM, LLMError
from src.schemas import Candidate, EvidenceCard, Page, SignalVerdict
from src.settings import FreshnessSettings, SignalSettings
from src.signals.keyword import KeywordMatcher

log = logging.getLogger(__name__)

SYSTEM_HEAD = """You verify B2B sales triggers found on a small business's own website.
Decide whether the SENTENCE announces a CURRENT, FIRST-PARTY event of one of these types:"""

SYSTEM_RULES = """Return signal_type "none" when:
- the keyword is used in another sense ("apply sunscreen", "open 9am-5pm", "book a job", "jobs done right", "expand your smile");
- the statement is negated or closed ("not hiring", "positions filled", "closed our Austin office");
- it is historical background ("founded in 1998", "opened our doors in 2005");
- it is generic navigation or boilerplate with no event ("Careers", "Privacy | Careers | Contact");
- it is about someone else (a client, a competitor, a news story about another company).
strength is "strong" for a specific, dated or concrete announcement ({strong}, a date),
otherwise "weak". detail is {detail} if the text names one, else \"\"."""

DEFAULT_SIGNALS = SignalSettings()


def _or_join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def system_prompt(cfg: SignalSettings) -> str:
    """The verifier's instructions, built from the configured signal types."""
    types = cfg.types.values()
    lines = [f"- {name}: {t.description}" for name, t in cfg.types.items()]
    rules = SYSTEM_RULES.format(strong=", ".join(t.strong_when for t in types),
                                detail=_or_join([t.detail for t in types]))
    return "\n".join([SYSTEM_HEAD, *lines, rules])


@lru_cache(maxsize=32)
def _verdict_schema(types: tuple[str, ...]) -> type[SignalVerdict]:
    return create_model("SignalVerdict", __base__=SignalVerdict,
                        signal_type=(Literal[types + ("none",)], ...))  # type: ignore[valid-type]


def verdict_schema(cfg: SignalSettings) -> type[SignalVerdict]:
    """SignalVerdict with signal_type narrowed to the configured types, so the JSON schema constrains the model."""
    return _verdict_schema(tuple(cfg.types))


def _user_message(sentence: str, context: str, url: str) -> str:
    return f"URL: {url}\nCONTEXT: {context}\nSENTENCE: {sentence}"


def build_messages(c: Candidate, cfg: SignalSettings = DEFAULT_SIGNALS) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt(cfg)}]
    for ex in cfg.examples:
        verdict = SignalVerdict(signal_type=ex.signal_type, strength=ex.strength, detail=ex.detail, reason=ex.reason)
        messages.append({"role": "user", "content": _user_message(ex.sentence, ex.context, "https://example.com/")})
        messages.append({"role": "assistant", "content": verdict.model_dump_json()})
    messages.append({"role": "user", "content": _user_message(c.sentence, c.context, c.url)})
    return messages


def verify(llm: LLM, candidate: Candidate, cfg: SignalSettings = DEFAULT_SIGNALS) -> Optional[SignalVerdict]:
    try:
        verdict = llm.chat_json(build_messages(candidate, cfg), verdict_schema(cfg))
    except LLMError as e:
        log.warning("verification failed for %r: %s", candidate.sentence[:80], e)
        return None
    if verdict.signal_type != "none" and verdict.signal_type not in cfg.types:
        log.warning("verifier returned unknown signal type %r; ignoring", verdict.signal_type)
        return None
    return verdict


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
    pages: list[Page], llm: LLM, *, freshness: FreshnessSettings, signals: SignalSettings = DEFAULT_SIGNALS
) -> list[EvidenceCard]:
    """Return verified signals, strongest first."""
    now = datetime.now(timezone.utc)
    cards: list[EvidenceCard] = []
    matcher = KeywordMatcher(signals.types)
    for c in matcher.candidates(pages, limit=signals.max_candidates_per_domain):
        verdict = verify(llm, c, signals)
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
