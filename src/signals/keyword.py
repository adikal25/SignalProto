"""Keyword matching: the baseline the signal agent is measured against.

It doubles as the agent's high-recall candidate stage: anything a keyword catches is
sent to the LLM for verification, so the agent can only *remove* false positives.
Keywords come from each configured signal type (settings.signals.types).
"""
from __future__ import annotations

import re
from typing import Optional

from src.schemas import Candidate, Page, SignalType
from src.settings import DEFAULT_SIGNAL_TYPES, SignalTypeDef
from src.tools.web import sentences


class KeywordMatcher:
    def __init__(self, types: dict[str, SignalTypeDef]):
        self.patterns: dict[SignalType, re.Pattern[str]] = {
            name: re.compile(r"\b(?:" + "|".join(t.keywords) + r")\b", re.I) for name, t in types.items()
        }

    def classify(self, sentence: str) -> Optional[SignalType]:
        """The signal type with the most keyword hits; ties go to the type configured first."""
        hits = {t: len(p.findall(sentence)) for t, p in self.patterns.items()}
        best = max(hits, key=lambda t: hits[t])
        return best if hits[best] else None

    def candidates(self, pages: list[Page], limit: int) -> list[Candidate]:
        out: list[Candidate] = []
        seen: set[str] = set()
        for page in pages:
            sents = sentences(page.text)
            for i, s in enumerate(sents):
                kind = self.classify(s)
                if kind is None or s.lower() in seen:
                    continue
                seen.add(s.lower())
                context = " ".join(sents[max(0, i - 1): i + 2])
                out.append(Candidate(
                    url=page.url, sentence=s[:400], context=context[:800],
                    keyword_type=kind, published_at=page.published_at,
                ))
                if len(out) >= limit:
                    return out
        return out


DEFAULT_MATCHER = KeywordMatcher(DEFAULT_SIGNAL_TYPES)


def keyword_classify(sentence: str, matcher: KeywordMatcher = DEFAULT_MATCHER) -> Optional[SignalType]:
    return matcher.classify(sentence)


def candidates_from_pages(pages: list[Page], limit: int, matcher: KeywordMatcher = DEFAULT_MATCHER) -> list[Candidate]:
    return matcher.candidates(pages, limit)
