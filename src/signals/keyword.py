"""Keyword matching: the baseline the signal agent is measured against.

It doubles as the agent's high-recall candidate stage: anything a keyword catches is
sent to the LLM for verification, so the agent can only *remove* false positives.
"""
from __future__ import annotations

import re
from typing import Optional

from src.schemas import Candidate, Page, SignalType
from src.tools.web import sentences

KEYWORDS: dict[SignalType, list[str]] = {
    "hiring": [
        r"hiring", r"join (?:our|the) team", r"careers?", r"jobs?", r"positions?", r"(?:job )?openings",
        r"open roles", r"apply(?: now)?", r"applications?", r"recruit(?:ing|ment)?", r"seeking",
        r"looking for", r"help wanted",
    ],
    "expansion": [
        r"grand opening", r"(?:now|officially) open", r"opening (?:soon|our|a)", r"opened", r"coming soon",
        r"new (?:location|office|clinic|store|storefront|studio|showroom|facility|branch|shop|space|building)",
        r"(?:second|third|fourth|fifth) (?:location|office|clinic|store|studio|gym|branch|shop|salon)",
        r"expand(?:ing|ed|s)?", r"expansion", r"ribbon cutting", r"now serving", r"moved (?:into|to)", r"lease",
    ],
}

_PATTERNS: dict[SignalType, re.Pattern[str]] = {
    t: re.compile(r"\b(?:" + "|".join(words) + r")\b", re.I) for t, words in KEYWORDS.items()
}


def keyword_classify(sentence: str) -> Optional[SignalType]:
    """Baseline classifier: the first signal type whose keywords appear in the sentence."""
    hits = {t: len(p.findall(sentence)) for t, p in _PATTERNS.items()}
    best = max(hits, key=lambda t: hits[t])
    return best if hits[best] else None


def candidates_from_pages(pages: list[Page], limit: int) -> list[Candidate]:
    out: list[Candidate] = []
    seen: set[str] = set()
    for page in pages:
        sents = sentences(page.text)
        for i, s in enumerate(sents):
            kind = keyword_classify(s)
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
