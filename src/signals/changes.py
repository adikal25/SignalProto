"""Change detection helpers: page content hashes and stable signal fingerprints."""
from __future__ import annotations

import hashlib
import re

from src.schemas import EvidenceCard


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_snippet(snippet: str) -> str:
    """Lowercase, drop digits and punctuation, so 'opens June 14!' and 'opens June 21' collapse."""
    text = re.sub(r"[\d\W_]+", " ", snippet.lower())
    return " ".join(text.split())


def fingerprint(domain: str, card: EvidenceCard) -> str:
    key = f"{domain.lower()}|{card.signal_type}|{normalize_snippet(card.snippet)}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]
