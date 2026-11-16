"""Prospect scoring: a transparent weighted sum, so every rank can be explained.

Learned weights need outcome data (replies, meetings); until then, explicit beats opaque.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from src.store import AccountRow, SignalRow
from src.settings import ScoringSettings

NEUTRAL = 0.5  # unknown size / no ICP preference neither helps nor hurts


class ScoreBreakdown(BaseModel):
    signal: float
    diversity: float
    recency: float
    size: float
    fit: float
    total: float  # 0-100


def _recency(signals: list[SignalRow], decay: float) -> float:
    newest = max(s.first_seen for s in signals)
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)
    weeks = max(0.0, (datetime.now(timezone.utc) - newest).days / 7)
    return decay ** weeks


def _size(employees: Optional[int], cfg: ScoringSettings) -> float:
    lo, hi = cfg.icp.min_employees, cfg.icp.max_employees
    if employees is None or (lo is None and hi is None):
        return NEUTRAL
    in_range = (lo is None or employees >= lo) and (hi is None or employees <= hi)
    return 1.0 if in_range else 0.2


def _fit(vertical: Optional[str], cfg: ScoringSettings) -> float:
    if not cfg.icp.verticals or not vertical:
        return NEUTRAL
    v = vertical.lower()
    return 1.0 if any(target.lower() in v for target in cfg.icp.verticals) else 0.2


def score_prospect(account: AccountRow, signals: list[SignalRow], cfg: ScoringSettings) -> ScoreBreakdown:
    if not signals:
        return ScoreBreakdown(signal=0, diversity=0, recency=0, size=_size(account.employees, cfg),
                              fit=_fit(account.vertical, cfg), total=0)
    parts = {
        "signal": max(s.confidence * cfg.type_weights.get(s.signal_type, 1.0) for s in signals),
        "diversity": len({s.signal_type for s in signals}) / max(1, len(cfg.type_weights)),
        "recency": _recency(signals, cfg.recency_weekly_decay),
        "size": _size(account.employees, cfg),
        "fit": _fit(account.vertical, cfg),
    }
    total = sum(cfg.weights.get(k, 0) * v for k, v in parts.items()) / max(1e-9, sum(cfg.weights.values()))
    return ScoreBreakdown(**{k: round(v, 3) for k, v in parts.items()}, total=round(100 * total, 1))
