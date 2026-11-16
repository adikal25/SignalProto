from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl

# Signal types are configured (settings.signals.types); "hiring" and "expansion" are the defaults.
SignalType = str


class Page(BaseModel):
    """One fetched page of a prospect's website."""
    url: str
    path: str
    text: str
    published_at: Optional[datetime] = None


class Candidate(BaseModel):
    """A sentence the keyword stage flagged as a possible signal (high recall, low precision)."""
    url: str
    sentence: str
    context: str
    keyword_type: SignalType
    published_at: Optional[datetime] = None


class SignalVerdict(BaseModel):
    """Structured output the signal agent's LLM must return for each candidate.

    The model is called with `verdict_schema(types)`, a subclass that narrows `signal_type`
    to the configured types plus "none", so the JSON schema constrains generation.
    """
    signal_type: str
    strength: Literal["strong", "weak"]
    # Required (not Optional): small models under a JSON-schema constraint almost always pick
    # null for optional fields. "" means "not stated".
    detail: str = Field(description='Role being hired for, or the new location/city; "" if not stated.')
    reason: str = Field(description="One short sentence justifying the decision.")


class EvidenceCard(BaseModel):
    signal_type: SignalType
    canonical_url: HttpUrl
    first_seen: datetime
    last_seen: datetime
    snippet: str = Field(max_length=320)
    detail: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    explain: str


class EmailDraft(BaseModel):
    subject: str = Field(max_length=120)
    body: str
    call_to_action: str


class Slot(BaseModel):
    start: datetime


class Booking(BaseModel):
    id: str
    start: datetime
    status: str
    attendee_email: str
