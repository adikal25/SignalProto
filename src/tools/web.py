from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import dateparser
import requests
from bs4 import BeautifulSoup

__all__ = ["fetch", "visible_text", "sentences", "extract_date"]

DEFAULT_HEADERS = {
    "User-Agent": "SignalProto/1.0 (+https://github.com/adikal25/SignalProto)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.7",
}

_session = requests.Session()
_session.mount("http://", requests.adapters.HTTPAdapter(max_retries=2))
_session.mount("https://", requests.adapters.HTTPAdapter(max_retries=2))


def fetch(url: str, *, timeout: int = 15) -> str:
    """Fetch a URL and return decoded HTML. Raises requests.HTTPError on non-2xx."""
    resp = _session.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    if not resp.encoding:
        resp.encoding = resp.apparent_encoding
    return resp.text


BLOCK_TAGS = [
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "ul", "ol", "div", "br", "tr", "td", "th",
    "section", "article", "main", "aside", "nav", "header", "footer", "form", "label", "button",
]


def visible_text(html: str) -> str:
    """All human-visible text. Banners and job lists matter, so no readability-style pruning."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    # Block-level boundaries become sentence breaks so headings don't merge into the next line.
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after(". ")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.!?])", r"\1", text)  # "open ." -> "open."
    text = re.sub(r"([.!?])(?:\s*\.)+", r"\1", text)  # "May 2! ." / ". ." -> single mark
    return text.strip(" .")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def extract_date(html: str) -> Optional[datetime]:
    """Best-effort publish date for a page, always timezone-aware (UTC)."""
    soup = BeautifulSoup(html, "lxml")
    for name, attrs in [
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "date"}),
        ("time", {}),
    ]:
        el = soup.find(name, attrs)
        if el:
            val = el.get("content") or el.get("datetime") or el.get_text(strip=True)
            dt = dateparser.parse(str(val)) if val else None
            if dt:
                return _as_utc(dt)
    m = re.search(
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4}",
        soup.get_text(" ", strip=True),
    )
    if m:
        dt = dateparser.parse(m.group(0))
        if dt:
            return _as_utc(dt)
    return None
