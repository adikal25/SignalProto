"""Deterministic site crawler: find a prospect's high-signal pages and fetch them.

Every site names its pages differently (/careers, /join-us, /work-with-us, /visit-our-new-office),
so with `crawl.discover` on the crawler ranks the site's OWN links: the homepage's <a> tags and
/sitemap.xml, scored by `crawl.link_hints` found in the URL path or the link text. Pages seen on
earlier runs are always re-checked (so change detection can notice a signal went away), and the
configured `paths` are tried as a fallback for sites that hide their links behind scripts.

Fetching does not need an LLM; keeping it deterministic makes the pipeline reproducible
and leaves the model budget for the judgement calls (is this a real signal? what to write?).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.schemas import Page
from src.settings import CrawlSettings
from src.tools.web import extract_date, fetch, visible_text

log = logging.getLogger(__name__)

Fetcher = Callable[[str], str]

SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".css", ".js",
                   ".zip", ".mp4", ".mp3", ".xml", ".json", ".doc", ".docx", ".xls", ".xlsx")
KNOWN_BONUS = 1.0     # pages read on earlier runs keep their place in the budget
FALLBACK_SCORE = 0.75  # configured paths: below a hint in the URL, above a hint only in link text
ANCHOR_WEIGHT = 0.5   # a hint only in the link text ("03. Small teams") is weaker evidence than one in the URL
PER_SECTION = 2       # pages per first path segment before other sections get a turn (/blog/tags/... floods)


def normalize_base_url(domain: str) -> str:
    domain = domain.strip()
    if not urlparse(domain).scheme:
        domain = f"https://{domain}"
    return domain.rstrip("/") + "/"


def _host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _canonical(url: str) -> str:
    """Drop query and fragment so /careers?utm=x and /careers#open are one page."""
    p = urlparse(url)
    return p._replace(query="", fragment="", params="").geturl()


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


@dataclass
class Link:
    url: str
    score: float = 0.0
    lastmod: str = ""  # ISO date from the sitemap; newer pages win ties


def hint_score(path: str, anchor: str, hints: Iterable[str]) -> float:
    """Distinct hint words that start a token of the URL path (1 each) or only of the link text (0.5 each)."""
    path_tokens, anchor_tokens = _tokens(path), _tokens(anchor)
    score = 0.0
    for h in hints:
        if any(t.startswith(h) for t in path_tokens):
            score += 1.0
        elif any(t.startswith(h) for t in anchor_tokens):
            score += ANCHOR_WEIGHT
    return score


def _section(url: str) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    return parts[0].lower() if parts else ""


def _spread_sections(urls: list[str]) -> list[str]:
    """Keep the ranking, but move a section's pages beyond the first PER_SECTION to the end."""
    head: list[str] = []
    tail: list[str] = []
    seen: dict[str, int] = {}
    for url in urls:
        section = _section(url)
        seen[section] = seen.get(section, 0) + 1
        (head if seen[section] <= PER_SECTION else tail).append(url)
    return head + tail


def links_from_html(html: str, page_url: str) -> list[tuple[str, str]]:
    """(absolute url, link text) for every same-site link on the page."""
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        url = _canonical(urljoin(page_url, href))
        if urlparse(url).scheme not in ("http", "https") or _host(url) != _host(page_url):
            continue
        if urlparse(url).path.lower().endswith(SKIP_EXTENSIONS):
            continue
        out.append((url, a.get_text(" ", strip=True)[:120]))
    return out


def urls_from_sitemap(xml: str, limit: int) -> tuple[list[tuple[str, str]], list[str]]:
    """(page url, lastmod) entries and nested sitemap urls from a sitemap or sitemap index."""
    soup = BeautifulSoup(xml, "xml")
    pages: list[tuple[str, str]] = []
    nested: list[str] = []
    for node in soup.find_all(["url", "sitemap"])[:limit]:
        loc = node.find("loc")
        if not loc or not loc.text.strip():
            continue
        if node.name == "sitemap":
            nested.append(loc.text.strip())
        else:
            lastmod = node.find("lastmod")
            pages.append((_canonical(loc.text.strip()), lastmod.text.strip() if lastmod else ""))
    return pages, nested


def _try(get: Fetcher, url: str) -> Optional[str]:
    try:
        return get(url)
    except requests.RequestException as e:
        log.debug("skip %s: %s", url, e)
        return None


def discover(base: str, homepage_html: Optional[str], settings: CrawlSettings, get: Fetcher,
             known_urls: Iterable[str] = ()) -> list[str]:
    """Candidate page URLs for a site, best first. The homepage itself is not included."""
    candidates: dict[str, Link] = {}

    def add(url: str, score: float, lastmod: str = "") -> None:
        if _host(url) != _host(base) or url.rstrip("/") == base.rstrip("/"):
            return
        link = candidates.setdefault(url, Link(url=url))
        link.score = max(link.score, score)
        link.lastmod = max(link.lastmod, lastmod)

    if settings.discover:
        if homepage_html:
            for url, text in links_from_html(homepage_html, base):
                score = hint_score(urlparse(url).path, text, settings.link_hints)
                if score:
                    add(url, score)
        sitemap = _try(get, urljoin(base, "sitemap.xml"))
        if sitemap:
            entries, nested = urls_from_sitemap(sitemap, settings.sitemap_max_urls)
            if not entries and nested:  # sitemap index: read the first child sitemap
                child = _try(get, nested[0])
                entries = urls_from_sitemap(child, settings.sitemap_max_urls)[0] if child else []
            for url, lastmod in entries:
                score = hint_score(urlparse(url).path, "", settings.link_hints)
                if score:
                    add(url, score, lastmod)

    for url in known_urls:
        add(url, candidates[url].score + KNOWN_BONUS if url in candidates else KNOWN_BONUS)
    for path in settings.paths:
        add(urljoin(base, path.lstrip("/")), FALLBACK_SCORE)

    # Python's sort is stable, so sorting by the least important key first gives:
    # higher score, then newer (sitemap lastmod), then shorter path (section pages over deep posts).
    ranked = sorted(candidates.values(), key=lambda link: len(link.url))
    ranked.sort(key=lambda link: link.lastmod, reverse=True)
    ranked.sort(key=lambda link: link.score, reverse=True)
    return _spread_sections([link.url for link in ranked])


def crawl(domain: str, settings: CrawlSettings, fetcher: Fetcher | None = None,
          known_urls: Iterable[str] = ()) -> list[Page]:
    get = fetcher or (lambda url: fetch(url, timeout=settings.timeout_seconds))
    base = normalize_base_url(domain)
    pages: list[Page] = []
    seen_text: set[str] = set()

    def read(url: str, html: str) -> None:
        text = visible_text(html)
        if not text or text in seen_text:  # many sites serve the homepage for unknown paths
            return
        seen_text.add(text)
        path = urlparse(url).path or "/"
        pages.append(Page(url=url, path=path, text=text, published_at=extract_date(html)))

    homepage = _try(get, base)
    if homepage is not None:
        read(base, homepage)

    attempts = 0
    for url in discover(base, homepage, settings, get, known_urls):
        if len(pages) >= settings.max_pages or attempts >= settings.max_pages * 2:
            break
        attempts += 1
        html = _try(get, url)
        if html is not None:
            read(url, html)
    log.info("crawled %s: %d pages", domain, len(pages))
    return pages
