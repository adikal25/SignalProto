"""Typed configuration loaded from configs/config.yml.

Secrets (API tokens) never live in YAML; they are read from environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yml"


class CrawlSettings(BaseModel):
    paths: list[str] = ["/", "/careers", "/jobs", "/locations", "/news", "/press", "/blog", "/about"]
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


class SignalSettings(BaseModel):
    max_candidates_per_domain: int = 12


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
