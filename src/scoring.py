"""Prospect scoring modelled on how B2B teams score accounts in practice.

Two independent axes, never blended into one opaque number:
- FIT    (who they are): firmographics vs. the ICP. Changes slowly. Graded A-D.
- INTENT (what they're doing now): buying signals. Decays fast. Graded 1-4.

The grade ("A1" = great fit, hot intent) maps to a tier and a next action, the way
Marketo / HubSpot-style lead grades drive SDR queues. Every point added or removed is
recorded as a human-readable reason, so a rep can see *why* an account is on top.

Intent details:
- each signal is worth  base_points(type) x confidence x relevance x decay(age, half-life of its type)
- signals stack with diminishing returns (noisy-OR): 1 - prod(1 - s_i)
- a "surge" bonus when several *different* signal types land inside a short window
Negative scoring: hard disqualifiers (excluded vertical, far outside size range) and a
cooldown after outreach, so the same account is not hammered.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel

from src.settings import ScoringSettings
from src.store import AccountRow, SignalRow

Tier = Literal["hot", "warm", "nurture", "disqualified"]


class Reason(BaseModel):
    axis: Literal["fit", "intent", "priority"]
    points: float
    text: str


class SignalContribution(BaseModel):
    fingerprint: str
    signal_type: str
    value: float          # 0-1 after confidence, relevance and decay
    age_days: int
    relevant: bool


class ScoreBreakdown(BaseModel):
    fit: float             # 0-100
    intent: float          # 0-100
    priority: float        # 0-100, used for ranking
    grade: str             # e.g. "A1"
    tier: Tier
    action: str
    reasons: list[Reason]
    signals: list[SignalContribution]
    contacted_days_ago: Optional[int] = None
    in_cooldown: bool = False


def _aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _age_days(dt: datetime, now: datetime) -> int:
    return max(0, (now - _aware(dt)).days)


# --- fit -------------------------------------------------------------------------

def score_fit(account: AccountRow, cfg: ScoringSettings) -> tuple[float, list[Reason], Optional[str]]:
    """Returns (fit 0-100, reasons, disqualification reason or None)."""
    icp = cfg.icp
    reasons: list[Reason] = []
    vertical = (account.vertical or "").lower()

    if vertical and any(x.lower() in vertical for x in icp.excluded_verticals):
        return 0.0, [Reason(axis="fit", points=-100, text=f"Excluded vertical: {account.vertical}")], "excluded vertical"

    points = 0.0
    # Company size: points by band; far outside the range disqualifies.
    emp = account.employees
    if emp is None:
        points += icp.unknown_size_points
        reasons.append(Reason(axis="fit", points=icp.unknown_size_points, text="Company size unknown"))
    elif icp.disqualify_above and emp > icp.disqualify_above:
        return 0.0, [Reason(axis="fit", points=-100, text=f"{emp} employees: above {icp.disqualify_above} limit")], "too large"
    else:
        band = next((b for b in icp.size_bands if b.min <= emp <= b.max), None)
        size_points = band.points if band else 0.0
        points += size_points
        label = f"{band.min}-{band.max}" if band else "outside target bands"
        reasons.append(Reason(axis="fit", points=size_points, text=f"{emp} employees ({label})"))

    # Vertical.
    if not icp.verticals:
        points += icp.vertical_points / 2
        reasons.append(Reason(axis="fit", points=icp.vertical_points / 2, text="No vertical preference set"))
    elif not vertical:
        points += icp.vertical_points / 2
        reasons.append(Reason(axis="fit", points=icp.vertical_points / 2, text="Vertical unknown"))
    elif any(t.lower() in vertical for t in icp.verticals):
        points += icp.vertical_points
        reasons.append(Reason(axis="fit", points=icp.vertical_points, text=f"Target vertical: {account.vertical}"))
    else:
        reasons.append(Reason(axis="fit", points=0, text=f"Non-target vertical: {account.vertical}"))

    max_points = max(b.points for b in icp.size_bands) + icp.vertical_points if icp.size_bands else icp.vertical_points
    return round(100 * points / max_points, 1), reasons, None


# --- intent ----------------------------------------------------------------------

def _is_relevant(signal: SignalRow, cfg: ScoringSettings) -> bool:
    keywords = cfg.relevant_keywords.get(signal.signal_type, [])
    if not keywords:
        return True  # no preference configured: every signal of this type counts fully
    text = f"{signal.snippet} {signal.detail or ''}".lower()
    return any(k.lower() in text for k in keywords)


def score_intent(signals: list[SignalRow], cfg: ScoringSettings, now: datetime) -> tuple[float, list[Reason], list[SignalContribution]]:
    reasons: list[Reason] = []
    contributions: list[SignalContribution] = []
    for s in signals:
        tcfg = cfg.signal_types.get(s.signal_type)
        if tcfg is None:
            continue
        age = _age_days(s.first_seen, now)
        decay = 0.5 ** (age / tcfg.half_life_days)
        relevant = _is_relevant(s, cfg)
        value = tcfg.base_value * s.confidence * decay * (1.0 if relevant else cfg.irrelevant_multiplier)
        contributions.append(SignalContribution(fingerprint=s.fingerprint, signal_type=s.signal_type,
                                                value=round(value, 3), age_days=age, relevant=relevant))

    contributions.sort(key=lambda c: c.value, reverse=True)
    by_fp = {s.fingerprint: s for s in signals}
    if not contributions:
        return 0.0, [Reason(axis="intent", points=0, text="No active buying signals")], []

    # Diminishing returns: the strongest signal counts fully, each extra one fills part of the gap.
    combined = 1 - math.prod(1 - c.value for c in contributions)
    running = 0.0
    for c in contributions:
        gained = (1 - running) * c.value
        running += gained
        reasons.append(Reason(axis="intent", points=round(100 * gained, 1), text=_signal_reason(c, by_fp)))

    recent_types = {c.signal_type for c in contributions if c.age_days <= cfg.surge_window_days}
    if len(recent_types) >= 2:
        bonus = cfg.surge_bonus * (1 - combined)
        combined += bonus
        reasons.append(Reason(axis="intent", points=round(100 * bonus, 1),
                              text=f"Surge: {len(recent_types)} signal types within {cfg.surge_window_days} days"))
    return round(100 * min(1.0, combined), 1), reasons, contributions


def _signal_reason(c: SignalContribution, by_fp: dict[str, SignalRow]) -> str:
    detail = by_fp[c.fingerprint].detail
    age = "today" if c.age_days == 0 else f"{c.age_days}d ago"
    text = f"{c.signal_type.capitalize()}{': ' + detail if detail else ''} ({age})"
    return text if c.relevant else text + ", outside your target roles"


# --- grade, tier, priority ---------------------------------------------------------

def _fit_letter(fit: float) -> str:
    return "A" if fit >= 80 else "B" if fit >= 60 else "C" if fit >= 40 else "D"


def _intent_digit(intent: float) -> str:
    return "1" if intent >= 70 else "2" if intent >= 45 else "3" if intent >= 20 else "4"


def score_prospect(account: AccountRow, signals: list[SignalRow], cfg: ScoringSettings,
                   last_contacted: Optional[datetime] = None, now: Optional[datetime] = None) -> ScoreBreakdown:
    now = now or datetime.now(timezone.utc)
    fit, fit_reasons, disqualified = score_fit(account, cfg)
    intent, intent_reasons, contributions = score_intent(signals, cfg, now)
    reasons = fit_reasons + intent_reasons
    grade = _fit_letter(fit) + _intent_digit(intent)

    # Geometric mean: a great fit with no intent, or loud intent from a bad fit, both rank low.
    priority = math.sqrt(fit * intent)
    contacted_days = _age_days(last_contacted, now) if last_contacted else None
    in_cooldown = contacted_days is not None and contacted_days < cfg.contact_cooldown_days
    if in_cooldown:
        penalty = priority * cfg.cooldown_penalty
        priority -= penalty
        reasons.append(Reason(axis="priority", points=-round(penalty, 1),
                              text=f"Contacted {contacted_days}d ago (cooldown {cfg.contact_cooldown_days}d)"))

    if disqualified:
        tier: Tier = "disqualified"
        action = f"Skip: {disqualified}"
        priority = 0.0
    elif grade[0] in "AB" and grade[1] in "12":
        tier, action = "hot", "Reach out today"
    elif grade[0] in "ABC" and grade[1] in "123":
        tier, action = "warm", "Add to this week's sequence"
    else:
        tier, action = "nurture", "Keep watching; no outreach yet"
    if tier in ("hot", "warm") and in_cooldown:
        action = f"Already contacted {contacted_days}d ago; follow up after cooldown"

    return ScoreBreakdown(fit=fit, intent=intent, priority=round(priority, 1), grade=grade, tier=tier,
                          action=action, reasons=reasons, signals=contributions,
                          contacted_days_ago=contacted_days, in_cooldown=in_cooldown)
