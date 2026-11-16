"""Meeting booking. The outbound agent offers real open slots; booking one creates the meeting.

Providers:
- calcom: live availability + booking via the Cal.com v2 API (CALCOM_API_KEY).
- link:   a static scheduling link (Calendly, Cal.com, SavvyCal...); the prospect books themselves.
- none:   plain-text CTA only.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

import requests

from src.schemas import Booking, Slot
from src.settings import BookingSettings


class BookingNotSupported(RuntimeError):
    pass


class Booker(Protocol):
    def available_slots(self) -> list[Slot]: ...
    def booking_link(self) -> Optional[str]: ...
    def book(self, start: datetime, name: str, email: str, timezone_name: str) -> Booking: ...


class NoBooker:
    def available_slots(self) -> list[Slot]:
        return []

    def booking_link(self) -> Optional[str]:
        return None

    def book(self, start: datetime, name: str, email: str, timezone_name: str) -> Booking:
        raise BookingNotSupported("booking provider is 'none'")


class LinkBooker(NoBooker):
    def __init__(self, link_url: str):
        self.link_url = link_url

    def booking_link(self) -> Optional[str]:
        return self.link_url

    def book(self, start: datetime, name: str, email: str, timezone_name: str) -> Booking:
        raise BookingNotSupported("provider 'link' books through the scheduling link, not the API")


class CalComBooker:
    BASE_URL = "https://api.cal.com/v2"
    SLOTS_API_VERSION = "2024-09-04"
    BOOKINGS_API_VERSION = "2026-02-25"

    def __init__(self, api_key: str, event_type_id: int, cfg: BookingSettings, link_url: Optional[str] = None):
        self.api_key = api_key
        self.event_type_id = event_type_id
        self.cfg = cfg
        self.link_url = link_url

    def _headers(self, version: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "cal-api-version": version}

    def booking_link(self) -> Optional[str]:
        return self.link_url

    def available_slots(self) -> list[Slot]:
        now = datetime.now(timezone.utc)
        params = {
            "eventTypeId": self.event_type_id,
            "start": (now + timedelta(days=1)).date().isoformat(),
            "end": (now + timedelta(days=self.cfg.lookahead_days)).date().isoformat(),
            "timeZone": self.cfg.timezone,
        }
        r = requests.get(f"{self.BASE_URL}/slots", params=params,
                         headers=self._headers(self.SLOTS_API_VERSION), timeout=15)
        r.raise_for_status()
        by_day: dict[str, list[dict]] = r.json().get("data", {})
        slots = [Slot(start=s["start"]) for day in sorted(by_day) for s in by_day[day]]
        return _spread(slots, self.cfg.slots_to_offer)

    def book(self, start: datetime, name: str, email: str, timezone_name: str) -> Booking:
        body = {
            "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "eventTypeId": self.event_type_id,
            "attendee": {"name": name, "email": email, "timeZone": timezone_name},
        }
        r = requests.post(f"{self.BASE_URL}/bookings", json=body,
                          headers=self._headers(self.BOOKINGS_API_VERSION), timeout=15)
        r.raise_for_status()
        data = r.json()["data"]
        return Booking(id=str(data.get("uid") or data.get("id")), start=data.get("start", start),
                       status=data.get("status", "accepted"), attendee_email=email)


def _spread(slots: list[Slot], n: int) -> list[Slot]:
    """Offer slots on different days where possible instead of three back-to-back times."""
    picked: list[Slot] = []
    days: set = set()
    for s in slots:
        if s.start.date() not in days:
            picked.append(s)
            days.add(s.start.date())
        if len(picked) == n:
            return picked
    rest = [s for s in slots if s not in picked]
    return sorted(picked + rest[: n - len(picked)], key=lambda s: s.start)


def make_booker(cfg: BookingSettings) -> Booker:
    if cfg.provider == "calcom":
        if not cfg.calcom_api_key or cfg.calcom_event_type_id is None:
            raise ValueError("booking.provider=calcom needs CALCOM_API_KEY and booking.calcom_event_type_id")
        return CalComBooker(cfg.calcom_api_key, cfg.calcom_event_type_id, cfg, link_url=cfg.link_url)
    if cfg.provider == "link":
        if not cfg.link_url:
            raise ValueError("booking.provider=link needs booking.link_url")
        return LinkBooker(cfg.link_url)
    return NoBooker()
