"""Page discovery and config-defined signal types. All network calls are faked."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests
from pydantic import ValidationError

from src.agents.outbound import DraftCopy
from src.crawler import crawl, discover, hint_score
from src.graph import Deps, Pipeline
from src.integrations.booking import NoBooker
from src.integrations.crm import Account
from src.llm.base import LLMError
from src.llm.tracing import prompt_version
from src.schemas import Candidate, SignalVerdict
from src.settings import CrawlSettings, Settings, SignalSettings
from src.signals.agent import build_messages, system_prompt, verdict_schema, verify
from src.signals.keyword import KeywordMatcher
from src.store import Store


class Site:
    """Fake website keyed by full URL; records which URLs were requested."""

    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.requested: list[str] = []

    def __call__(self, url: str) -> str:
        self.requested.append(url)
        if url not in self.pages:
            raise requests.HTTPError("404")
        return self.pages[url]


HOME = """<html><body>
  <a href="/work-with-us">Join our team</a> <a href="/new-patients">New patients</a>
  <a href="https://www.smile.test/visit/holly-springs?utm=nav#map">Our locations</a>
  <a href="https://facebook.com/careers">Facebook careers</a> <a href="/menu.pdf">Careers PDF</a>
  <a href="mailto:hi@smile.test">Email us</a>
  <p>Welcome to Smile Dental.</p></body></html>"""
SITEMAP = """<?xml version="1.0"?><urlset>
  <url><loc>https://smile.test/news/old-post</loc><lastmod>2024-01-01</lastmod></url>
  <url><loc>https://smile.test/news/grand-opening</loc><lastmod>2026-11-01</lastmod></url>
  <url><loc>https://smile.test/privacy</loc></url></urlset>"""


def smile(extra: dict[str, str] | None = None) -> Site:
    pages = {
        "https://smile.test/": HOME,
        "https://smile.test/sitemap.xml": SITEMAP,
        "https://smile.test/work-with-us": "<p>We're hiring a hygienist.</p>",
        "https://www.smile.test/visit/holly-springs": "<p>Our Holly Springs office is now open.</p>",
        "https://smile.test/news/grand-opening": "<p>Grand opening this Saturday.</p>",
        "https://smile.test/news/old-post": "<p>Tips for brushing.</p>",
        "https://smile.test/new-patients": "<p>Forms for new patients.</p>",
        "https://smile.test/privacy": "<p>Privacy.</p>",
    }
    pages.update(extra or {})
    return Site(pages)


# --- discovery -----------------------------------------------------------------------

def test_hint_score_matches_word_prefixes_in_path_and_link_text():
    hints = ["career", "join", "location"]
    assert hint_score("/careers", "", hints) == 1
    assert hint_score("/work-with-us", "Join our team", hints) == 0.5  # link text only: half weight
    assert hint_score("/visit", "Our locations", hints) == 0.5
    assert hint_score("/new-patients", "New patients", hints) == 0
    assert hint_score("/rejoined", "", hints) == 0  # prefix of a token, not any substring


def test_crawl_finds_pages_by_the_sites_own_links_and_sitemap():
    site = smile()
    pages = crawl("smile.test", CrawlSettings(paths=["/"]), fetcher=site)
    urls = [p.url for p in pages]
    assert urls[0] == "https://smile.test/"
    assert {"https://smile.test/work-with-us", "https://www.smile.test/visit/holly-springs",
            "https://smile.test/news/grand-opening"} <= set(urls)
    # no hint word -> not read; other sites, files and mail links are never followed
    assert "https://smile.test/new-patients" not in site.requested
    assert "https://smile.test/privacy" not in site.requested
    assert not any("facebook" in u or u.endswith(".pdf") or "mailto" in u for u in site.requested)
    assert pages[urls.index("https://smile.test/work-with-us")].path == "/work-with-us"


def test_newer_sitemap_pages_rank_first_and_max_pages_holds():
    site = smile()
    ranked = discover("https://smile.test/", None, CrawlSettings(paths=[]), site)
    assert ranked.index("https://smile.test/news/grand-opening") < ranked.index("https://smile.test/news/old-post")
    pages = crawl("smile.test", CrawlSettings(paths=["/"], max_pages=2), fetcher=site)
    assert len(pages) == 2


def test_previously_seen_pages_are_rechecked_and_fallback_paths_still_work():
    site = smile({"https://smile.test/old-promo": "<p>Spring promo.</p>",
                  "https://smile.test/jobs": "<p>Jobs board.</p>"})
    pages = crawl("smile.test", CrawlSettings(paths=["/jobs"]),
                  fetcher=site, known_urls=["https://smile.test/old-promo"])
    urls = {p.url for p in pages}
    assert "https://smile.test/old-promo" in urls  # no longer linked, still checked: lets signals go stale
    assert "https://smile.test/jobs" in urls


def test_one_section_cannot_take_the_whole_budget():
    blog = {f"https://smile.test/blog/tags/news/{i}": f"<p>Post {i}.</p>" for i in range(10)}
    site = smile({**blog, "https://smile.test/sitemap.xml":
                  "<urlset>" + "".join(f"<url><loc>{u}</loc></url>" for u in blog) + "</urlset>",
                  "https://smile.test/careers": "<p>Careers page.</p>"})
    pages = crawl("smile.test", CrawlSettings(paths=["/careers"], max_pages=5), fetcher=site)
    paths = [p.path for p in pages]
    assert sum(p.startswith("/blog/") for p in paths) == 2 and "/careers" in paths


def test_discover_off_uses_only_configured_paths():
    site = smile()
    crawl("smile.test", CrawlSettings(paths=["/", "/careers"], discover=False), fetcher=site)
    assert site.requested == ["https://smile.test/", "https://smile.test/careers"]


def test_sitemap_index_is_followed_one_level():
    index = """<sitemapindex><sitemap><loc>https://smile.test/pages.xml</loc></sitemap></sitemapindex>"""
    child = """<urlset><url><loc>https://smile.test/careers/hygienist</loc></url></urlset>"""
    site = Site({"https://smile.test/": "<p>Hi there friend.</p>", "https://smile.test/sitemap.xml": index,
                 "https://smile.test/pages.xml": child,
                 "https://smile.test/careers/hygienist": "<p>Hygienist wanted.</p>"})
    pages = crawl("smile.test", CrawlSettings(paths=[]), fetcher=site)
    assert "https://smile.test/careers/hygienist" in {p.url for p in pages}


# --- signal types from config ----------------------------------------------------------

NEW_SERVICE = {
    "description": "the business itself now offers a new service or treatment.",
    "detail": "the new service",
    "strong_when": "a named service",
    "noticed": "Saw that you've added a new service",
    "keywords": [r"now offer(?:s|ing)?", r"introducing"],
}


def custom_settings(**signals) -> Settings:
    return Settings.model_validate({
        "crawl": {"paths": ["/"], "discover": False},
        "signals": {"types": {"new_service": NEW_SERVICE}, **signals},
        "scoring": {"signal_types": {"hiring": {"base_value": 0.6, "half_life_days": 30},
                                     "expansion": {"base_value": 0.75, "half_life_days": 60},
                                     "new_service": {"base_value": 0.5, "half_life_days": 45}}},
    })


def test_default_prompt_is_unchanged_so_recorded_evals_stay_valid():
    c = Candidate(url="u", sentence="s", context="c", keyword_type="hiring")
    assert prompt_version(build_messages(c)) == "81c5f2c9209c"


def test_custom_type_flows_into_keywords_prompt_and_schema():
    cfg = custom_settings().signals
    assert list(cfg.types) == ["hiring", "expansion", "new_service"]  # merged after the built-ins
    assert KeywordMatcher(cfg.types).classify("We now offer Invisalign.") == "new_service"
    prompt = system_prompt(cfg)
    assert "- new_service: the business itself now offers a new service or treatment." in prompt
    assert "(a named role, a named city or address, a named service, a date)" in prompt
    assert "detail is the role, the new location or the new service if" in prompt
    enum = verdict_schema(cfg).model_json_schema()["properties"]["signal_type"]["enum"]
    assert enum == ["hiring", "expansion", "new_service", "none"]


def test_builtin_can_be_turned_off_or_partly_overridden():
    cfg = SignalSettings.model_validate({
        "types": {"expansion": None, "hiring": {"keywords": ["help wanted"]}},
        "examples": [],
    })
    assert list(cfg.types) == ["hiring"]
    assert cfg.types["hiring"].keywords == ["help wanted"]
    assert cfg.types["hiring"].noticed == "Saw that you're hiring"  # untouched fields keep the built-in value


@pytest.mark.parametrize("raw,match", [
    ({"signals": {"types": {"new_service": NEW_SERVICE}}}, "scoring.signal_types.*new_service"),
    ({"signals": {"types": {"expansion": None}}}, "not in signals.types"),  # built-in examples use it
    ({"signals": {"types": {"New Service": NEW_SERVICE}}}, "lowercase"),
    ({"signals": {"types": {"none": NEW_SERVICE}}}, "lowercase"),
    ({"signals": {"types": {"hiring": {"keywords": ["(unclosed"]}}}}, "bad keyword pattern"),
])
def test_invalid_signal_config_fails_at_load_with_a_clear_message(raw, match):
    with pytest.raises(ValidationError, match=match):
        Settings.model_validate(raw)


class ServiceLLM:
    def __init__(self, signal_type: str = "new_service", fail: bool = False):
        self.signal_type, self.fail, self.schemas = signal_type, fail, []

    def chat_json(self, messages, schema):
        self.schemas.append(schema)
        if self.fail:
            raise LLMError("down")
        if schema is DraftCopy:
            return DraftCopy(subject="s", opener="o", value="v")
        return SignalVerdict(signal_type=self.signal_type, strength="strong", detail="Invisalign", reason="r")


def test_custom_type_runs_end_to_end(tmp_path):
    settings = custom_settings()
    site = Site({"https://smile.test/": "<p>We now offer Invisalign at both offices.</p>"})
    signal_llm = ServiceLLM()
    deps = Deps(settings=settings, store=Store(tmp_path / "t.db"), signal_llm=signal_llm,
                outbound_llm=ServiceLLM(fail=True), booker=NoBooker(), crm=MagicMock(), fetcher=site)
    r = Pipeline(deps).run(Account(domain="smile.test", company="Smile"))
    assert [d.card.signal_type for d in r.new_signals] == ["new_service"]
    assert "new_service" in signal_llm.schemas[0].model_json_schema()["properties"]["signal_type"]["enum"]
    assert "Saw that you've added a new service (Invisalign)." in r.email.body  # template fallback per type
    assert deps.store.signals("smile.test")[0].signal_type == "new_service"


def test_verdict_of_an_unconfigured_type_is_dropped():
    c = Candidate(url="u", sentence="We now offer Invisalign.", context="c", keyword_type="hiring")
    assert verify(ServiceLLM(signal_type="new_service"), c) is None  # default config has no new_service
