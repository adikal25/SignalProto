"""Outbound agent: turn a verified signal into a short, specific email that proposes real meeting times.

The LLM writes only the copy (subject, opener, value line). Scheduling facts (slots, link) and the
sign-off are assembled in code so the model can never invent a meeting time.
"""
from __future__ import annotations

import logging
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from src.llm.base import LLM, LLMError
from src.schemas import EmailDraft, EvidenceCard, Slot
from src.settings import SenderSettings

log = logging.getLogger(__name__)

SYSTEM = """You write the first lines of a cold email FROM a sales rep TO a small business they noticed online.
Write from the rep's point of view, addressing the prospect as "you". Never write as the prospect.
Return:
- subject: at most 60 characters, specific to the signal, no clickbait.
- opener: one sentence starting "Saw that you're" stating what you noticed, using ONLY facts from the evidence
  (the role they are hiring for, or the new location). Do not add facts the evidence does not state.
- value: one sentence on how the rep's company helps with exactly that situation.
No greetings, no sign-off, no meeting times, no links, no emojis, no placeholders like [Name]."""


class DraftCopy(BaseModel):
    subject: str = Field(max_length=120)
    opener: str
    value: str


def format_slots(slots: list[Slot], tz: str) -> list[str]:
    zone = ZoneInfo(tz)
    return [s.start.astimezone(zone).strftime("%a %b %-d, %-I:%M %p %Z") for s in slots]


def build_ask(slot_text: list[str], link: Optional[str]) -> str:
    if slot_text:
        ask = f"Would any of these work for a 15-minute call: {'; '.join(slot_text)}?"
        return f"{ask} Or pick another time: {link}" if link else ask
    if link:
        return f"Open to a 15-minute call? Grab any time here: {link}"
    return "Open to a 15-minute call next week?"


def build_messages(card: EvidenceCard, company: str, sender: SenderSettings) -> list[dict[str, str]]:
    user = "\n".join([
        f"Prospect company: {company}",
        f"Signal type: {card.signal_type}",
        f"What their website says: \"{card.snippet}\"",
        f"Detail: {card.detail or 'n/a'}",
        f"Rep: {sender.name} at {sender.company}, which {sender.value_prop}.",
    ])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def fallback_copy(card: EvidenceCard, company: str, sender: SenderSettings) -> DraftCopy:
    noticed = {"hiring": "Saw that you're hiring", "expansion": "Saw that you're opening a new location"}
    detail = f" ({card.detail})" if card.detail else ""
    return DraftCopy(subject=f"Quick idea for {company}", opener=f"{noticed[card.signal_type]}{detail}.",
                     value=f"{sender.company} {sender.value_prop}.")


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?" else text + "."


def draft_email(llm: LLM, card: EvidenceCard, company: str, sender: SenderSettings,
                slots: list[Slot], link: Optional[str], tz: str) -> EmailDraft:
    try:
        copy = llm.chat_json(build_messages(card, company, sender), DraftCopy)
    except LLMError as e:
        log.warning("outbound LLM failed, using template: %s", e)
        copy = fallback_copy(card, company, sender)
    ask = build_ask(format_slots(slots, tz), link)
    body = (f"Hi {company} team,\n\n{_sentence(copy.opener)} {_sentence(copy.value)}\n\n{ask}\n\n"
            f"{sender.name}\n{sender.company}")
    return EmailDraft(subject=copy.subject.strip()[:120], body=body, call_to_action=ask)
