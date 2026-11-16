"""Deterministic site crawler: fetch a fixed set of high-signal paths for a domain.

Fetching does not need an LLM; keeping it deterministic makes the pipeline reproducible
and leaves the model budget for the judgement calls (is this a real signal? what to write?).
"""
from __future__ import annotations

import logging
from typing import Callable
from urllib.parse import urljoin, urlparse

import requests

from src.schemas import Page
from src.settings import CrawlSettings
from src.tools.web import extract_date, fetch, visible_text

log = logging.getLogger(__name__)

Fetcher = Callable[[str], str]


def normalize_base_url(domain: str) -> str:
    domain = domain.strip()
    if not urlparse(domain).scheme:
        domain = f"https://{domain}"
    return domain.rstrip("/") + "/"


def crawl(domain: str, settings: CrawlSettings, fetcher: Fetcher | None = None) -> list[Page]:
    get = fetcher or (lambda url: fetch(url, timeout=settings.timeout_seconds))
    base = normalize_base_url(domain)
    pages: list[Page] = []
    seen_text: set[str] = set()
    for path in settings.paths:
        if len(pages) >= settings.max_pages:
            break
        url = urljoin(base, path.lstrip("/"))
        try:
            html = get(url)
        except requests.RequestException as e:
            log.debug("skip %s: %s", url, e)
            continue
        text = visible_text(html)
        if not text or text in seen_text:  # many sites serve the homepage for unknown paths
            continue
        seen_text.add(text)
        pages.append(Page(url=url, path=path, text=text, published_at=extract_date(html)))
    log.info("crawled %s: %d pages", domain, len(pages))
    return pages
