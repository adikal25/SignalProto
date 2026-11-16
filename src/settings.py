"""Typed configuration loaded from configs/config.yml.

Secrets (API tokens) never live in YAML; they are read from environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yml"


class CrawlSettings(BaseModel):
    """Which pages of a prospect's site to read.

    With `discover` on, the crawler reads the homepage's own links and /sitemap.xml and ranks
    them by `link_hints`, so a site's /join-us or /visit-our-new-office is found without
    configuration. `paths` are always tried as well, ranked below discovered matches.
    """
    paths: list[str] = ["/", "/careers", "/jobs", "/locations", "/news", "/press", "/blog", "/about"]
    discover: bool = True
    link_hints: list[str] = [
        "career", "job", "hiring", "join", "employment", "opportunit",
        "location", "office", "clinic", "store", "branch", "opening",
        "news", "press", "blog", "announce", "update", "about",
    ]
    sitemap_max_urls: int = 500  # only the first N <loc> entries are considered
    max_pages: int = 8
    timeout_seconds: int = 15


class GateSettings(BaseModel):
    """Minimum confidence a signal needs before the outbound agent acts on it."""
    min_confidence: float = Field(default=0.6, ge=0, le=1)


class FreshnessSettings(BaseModel):
    weekly_decay: float = Field(default=0.85, gt=0, le=1)
    floor: float = Field(default=0.3, ge=0, le=1)


ProviderName = Literal["ollama", "anthropic", "openai", "groq", "openrouter", "gemini", "openai_compatible"]


class LLMSettings(BaseModel):
    """One agent's model. `provider: ollama` runs locally; anything else is bring-your-own-key.

    Keys are never stored in YAML: each hosted provider reads its key from an environment
    variable (see src/llm/providers.py), or from `api_key_env` if you name one.
    """
    model_config = ConfigDict(protected_namespaces=())  # allow the `model_id` field name

    provider: ProviderName = "ollama"
    model_id: str = "llama3.2:3b"
    max_new_tokens: int = 300  # hosted models with built-in reasoning need room: use >= 4096 for anthropic
    temperature: Optional[float] = 0.0  # null to omit (some hosted reasoning models reject it); ignored by anthropic (current Claude models don't take sampling params)
    base_url: Optional[str] = None     # required for openai_compatible (vLLM, LM Studio, Together, ...)
    api_key_env: Optional[str] = None  # override the provider's default key variable
    effort: Optional[Literal["low", "medium", "high", "xhigh", "max"]] = None  # anthropic only
    timeout_seconds: int = 120


class LLMGroup(BaseModel):
    signal: LLMSettings = LLMSettings(max_new_tokens=200, temperature=0.0)
    outbound: LLMSettings = LLMSettings(max_new_tokens=400, temperature=0.3)


class SignalTypeDef(BaseModel):
    """One kind of buying signal. Everything type-specific in the pipeline is built from these:
    the keyword recall stage, the verification prompt and JSON schema, and the email fallback."""
    description: str                 # what counts, as one line of the verification prompt
    detail: str                      # what `detail` should hold, e.g. "the role"
    strong_when: str                 # what makes it a strong signal, e.g. "a named role"
    noticed: str                     # fallback email opener, e.g. "Saw that you're hiring"
    keywords: list[str] = Field(min_length=1)  # case-insensitive regex fragments, matched on word boundaries

    @field_validator("keywords")
    @classmethod
    def _valid_regex(cls, v: list[str]) -> list[str]:
        for k in v:
            try:
                re.compile(k)
            except re.error as e:
                raise ValueError(f"bad keyword pattern {k!r}: {e}") from e
        return v


class SignalExample(BaseModel):
    """A worked example shown to the verifier (few-shot). Order is preserved in the prompt."""
    sentence: str
    context: str
    signal_type: str  # one of signals.types, or "none"
    strength: Literal["strong", "weak"]
    detail: str = ""
    reason: str


DEFAULT_SIGNAL_TYPES: dict[str, SignalTypeDef] = {
    "hiring": SignalTypeDef(
        description='the business itself is actively recruiting now (open roles, "we\'re hiring", a job posting).',
        detail="the role",
        strong_when="a named role",
        noticed="Saw that you're hiring",
        keywords=[
            r"hiring", r"join (?:our|the) team", r"careers?", r"jobs?", r"positions?", r"(?:job )?openings",
            r"open roles", r"apply(?: now)?", r"applications?", r"recruit(?:ing|ment)?", r"seeking",
            r"looking for", r"help wanted",
        ],
    ),
    "expansion": SignalTypeDef(
        description="the business itself is opening, has just opened, or will open a new location, office, clinic, store or service area.",
        detail="the new location",
        strong_when="a named city or address",
        noticed="Saw that you're opening a new location",
        keywords=[
            r"grand opening", r"(?:now|officially) open", r"opening (?:soon|our|a)", r"opened", r"coming soon",
            r"new (?:location|office|clinic|store|storefront|studio|showroom|facility|branch|shop|space|building)",
            r"(?:second|third|fourth|fifth) (?:location|office|clinic|store|studio|gym|branch|shop|salon)",
            r"expand(?:ing|ed|s)?", r"expansion", r"ribbon cutting", r"now serving", r"moved (?:into|to)", r"lease",
        ],
    ),
}

DEFAULT_SIGNAL_EXAMPLES: list[SignalExample] = [
    SignalExample(
        sentence="Grand opening of our second clinic in Cary on March 3!",
        context="Grand opening of our second clinic in Cary on March 3! Book your first visit today.",
        signal_type="expansion", strength="strong", detail="Cary", reason="Announces a new clinic location with a date.",
    ),
    SignalExample(
        sentence="Home | Services | Careers | Contact",
        context="Home | Services | Careers | Contact",
        signal_type="none", strength="weak", detail="", reason="Navigation link, not a hiring announcement.",
    ),
    SignalExample(
        sentence="We're hiring a front desk coordinator and two dental hygienists.",
        context="Join our growing team. We're hiring a front desk coordinator and two dental hygienists. Apply below.",
        signal_type="hiring", strength="strong", detail="front desk coordinator, dental hygienists",
        reason="Names specific open roles.",
    ),
    SignalExample(
        sentence="Proudly serving Raleigh since we opened in 1998.",
        context="Family owned. Proudly serving Raleigh since we opened in 1998.",
        signal_type="none", strength="weak", detail="", reason="Historical opening date, not a current event.",
    ),
]

_TYPE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class SignalSettings(BaseModel):
    max_candidates_per_domain: int = 12
    # Merged onto DEFAULT_SIGNAL_TYPES (see _merge_with_builtins). Order matters: on a keyword tie the first type wins.
    types: dict[str, SignalTypeDef] = Field(default_factory=lambda: dict(DEFAULT_SIGNAL_TYPES), min_length=1)
    examples: list[SignalExample] = Field(default_factory=lambda: list(DEFAULT_SIGNAL_EXAMPLES))

    @field_validator("types", mode="before")
    @classmethod
    def _merge_with_builtins(cls, given: object) -> object:
        """Configured types merge onto the built-ins: add a type by naming only the new one,
        override some fields of a built-in, or set a built-in to null to turn it off."""
        if not isinstance(given, dict):
            return given
        merged: dict[str, object] = {name: t.model_dump() for name, t in DEFAULT_SIGNAL_TYPES.items()}
        for name, value in given.items():
            if value is None:
                merged.pop(name, None)
            elif isinstance(value, dict) and name in merged:
                merged[name] = {**merged[name], **value}
            else:
                merged[name] = value
        return merged

    @model_validator(mode="after")
    def _consistent(self) -> "SignalSettings":
        for name in self.types:
            if name == "none" or not _TYPE_NAME.match(name):
                raise ValueError(f"signal type name {name!r}: use lowercase letters, digits and _ (not 'none')")
        for ex in self.examples:
            if ex.signal_type != "none" and ex.signal_type not in self.types:
                raise ValueError(f"example {ex.sentence[:40]!r} uses signal type {ex.signal_type!r}, "
                                 f"which is not in signals.types; edit signals.examples too")
        return self


class SenderSettings(BaseModel):
    """Who the outreach is from and what they offer; injected into the outbound prompt."""
    name: str = "Alex"
    company: str = "Acme"
    value_prop: str = "helps growing teams turn new demand into booked appointments"


class BookingSettings(BaseModel):
    provider: Literal["none", "link", "calcom"] = "none"
    link_url: Optional[str] = None
    calcom_event_type_id: Optional[int] = None
    timezone: str = "America/New_York"
    slots_to_offer: int = 3
    lookahead_days: int = 7

    @property
    def calcom_api_key(self) -> Optional[str]:
        return os.getenv("CALCOM_API_KEY")


class CRMSettings(BaseModel):
    provider: Literal["none", "csv", "webhook", "hubspot"] = "none"
    csv_accounts_path: Optional[str] = None
    csv_output_path: str = "data/crm_records.jsonl"
    webhook_url: Optional[str] = None

    @property
    def hubspot_token(self) -> Optional[str]:
        return os.getenv("HUBSPOT_TOKEN")


class StorageSettings(BaseModel):
    db_path: str = "data/signalproto.db"


class AlertSettings(BaseModel):
    """New signals found by `watch` are POSTed here (Slack incoming webhook, Zapier, n8n...)."""
    webhook_url: Optional[str] = None


class SizeBand(BaseModel):
    min: int
    max: int
    points: float


class ICPSettings(BaseModel):
    """Ideal customer profile: drives the FIT score."""
    size_bands: list[SizeBand] = [
        SizeBand(min=1, max=4, points=10), SizeBand(min=5, max=49, points=60),
        SizeBand(min=50, max=200, points=45), SizeBand(min=201, max=1000, points=15),
    ]
    unknown_size_points: float = 30
    disqualify_above: Optional[int] = 5000
    verticals: list[str] = []
    excluded_verticals: list[str] = []
    vertical_points: float = 40


class SignalTypeScoring(BaseModel):
    base_value: float = Field(ge=0, le=1)   # what a fresh, certain, relevant signal is worth (0-1)
    half_life_days: float = Field(gt=0)     # intent halves every N days


class ScoringSettings(BaseModel):
    """Fit x intent account scoring. See src/scoring.py."""
    icp: ICPSettings = ICPSettings()
    signal_types: dict[str, SignalTypeScoring] = {
        "expansion": SignalTypeScoring(base_value=0.75, half_life_days=60),  # openings stay relevant for months
        "hiring": SignalTypeScoring(base_value=0.6, half_life_days=30),     # job posts go stale fast
    }
    # Signals mentioning these terms map directly to what you sell; others count at irrelevant_multiplier.
    relevant_keywords: dict[str, list[str]] = {}
    irrelevant_multiplier: float = Field(default=0.6, ge=0, le=1)
    surge_window_days: int = 30
    surge_bonus: float = Field(default=0.25, ge=0, le=1)  # share of the remaining gap a multi-signal surge closes
    contact_cooldown_days: int = 21
    cooldown_penalty: float = Field(default=0.5, ge=0, le=1)


class Settings(BaseModel):
    crawl: CrawlSettings = CrawlSettings()
    gate: GateSettings = GateSettings()
    freshness: FreshnessSettings = FreshnessSettings()
    llm: LLMGroup = LLMGroup()
    signals: SignalSettings = SignalSettings()
    sender: SenderSettings = SenderSettings()
    booking: BookingSettings = BookingSettings()
    crm: CRMSettings = CRMSettings()
    storage: StorageSettings = StorageSettings()
    alerts: AlertSettings = AlertSettings()
    scoring: ScoringSettings = ScoringSettings()

    @model_validator(mode="after")
    def _every_signal_type_is_scored(self) -> "Settings":
        missing = [t for t in self.signals.types if t not in self.scoring.signal_types]
        if missing:
            raise ValueError(f"add scoring.signal_types entries (base_value, half_life_days) for: {', '.join(missing)}")
        return self


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.getenv("SIGNALPROTO_CONFIG", DEFAULT_CONFIG_PATH))
    raw = {}
    if config_path.exists():
        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}
    settings = Settings.model_validate(raw)
    if os.getenv("SIGNALPROTO_DB"):  # e.g. point the API at the demo database
        settings.storage.db_path = os.environ["SIGNALPROTO_DB"]
    return settings
