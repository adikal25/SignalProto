"""Typed configuration loaded from configs/config.yml.

Secrets (API tokens) never live in YAML; they are read from environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

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


class LLMSettings(BaseModel):
    model_id: str = "llama3.2:3b"
    max_new_tokens: int = 300
    temperature: float = 0.0


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


class ICPSettings(BaseModel):
    min_employees: Optional[int] = None
    max_employees: Optional[int] = None
    verticals: list[str] = []


class ScoringSettings(BaseModel):
    """Transparent weighted prospect score. Component weights should sum to 1."""
    type_weights: dict[str, float] = {"expansion": 1.0, "hiring": 0.8}
    weights: dict[str, float] = {"signal": 0.45, "diversity": 0.15, "recency": 0.2, "size": 0.1, "fit": 0.1}
    recency_weekly_decay: float = Field(default=0.85, gt=0, le=1)
    icp: ICPSettings = ICPSettings()


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
