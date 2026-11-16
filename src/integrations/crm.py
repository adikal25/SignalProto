"""CRM adapters. The pipeline only depends on the `CRMAdapter` protocol.

- hubspot: pull companies, write each signal + draft as a note on the company record.
- webhook: POST a JSON payload anywhere (Zapier / Make / n8n -> Salesforce, Pipedrive, Attio...).
- csv:     accounts from a CSV, records appended to a JSONL file (local dev, no CRM needed).
"""
from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import requests
from pydantic import BaseModel

from src.schemas import EmailDraft, EvidenceCard
from src.settings import CRMSettings


class Account(BaseModel):
    domain: str
    company: Optional[str] = None
    crm_id: Optional[str] = None
    employees: Optional[int] = None
    vertical: Optional[str] = None


class SignalRecord(BaseModel):
    account: Account
    card: EvidenceCard
    email: Optional[EmailDraft] = None


def account_from_row(row: dict[str, str]) -> Account:
    """CSV row with `domain` and optional `company`, `employees`, `vertical` columns."""
    def clean(key: str) -> Optional[str]:
        return (row.get(key) or "").strip() or None

    employees = clean("employees")
    return Account(domain=row["domain"].strip(), company=clean("company"),
                   employees=int(employees) if employees and employees.isdigit() else None,
                   vertical=clean("vertical"))


class CRMAdapter(Protocol):
    def list_accounts(self, limit: int) -> list[Account]: ...
    def record_signal(self, record: SignalRecord) -> Optional[str]:
        """Persist a signal (and draft) against the account. Returns the CRM object id if any."""
        ...


class NoCRM:
    def list_accounts(self, limit: int) -> list[Account]:
        return []

    def record_signal(self, record: SignalRecord) -> Optional[str]:
        return None


class CSVCRM:
    def __init__(self, accounts_path: Optional[str], output_path: str):
        self.accounts_path = accounts_path
        self.output_path = Path(output_path)

    def list_accounts(self, limit: int) -> list[Account]:
        if not self.accounts_path:
            return []
        with open(self.accounts_path, newline="") as f:
            rows = list(csv.DictReader(f))
        return [account_from_row(r) for r in rows[:limit]]

    def record_signal(self, record: SignalRecord) -> Optional[str]:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a") as f:
            f.write(record.model_dump_json() + "\n")
        return None


class WebhookCRM:
    def __init__(self, url: str):
        self.url = url

    def list_accounts(self, limit: int) -> list[Account]:
        return []  # push-only: accounts arrive through the API instead

    def record_signal(self, record: SignalRecord) -> Optional[str]:
        payload = {"event": "signal.detected", **json.loads(record.model_dump_json())}
        requests.post(self.url, json=payload, timeout=15).raise_for_status()
        return None


class HubSpotCRM:
    BASE_URL = "https://api.hubapi.com"
    NOTE_TO_COMPANY = 190  # HUBSPOT_DEFINED association type id

    def __init__(self, token: str):
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def list_accounts(self, limit: int) -> list[Account]:
        accounts: list[Account] = []
        after: Optional[str] = None
        while len(accounts) < limit:
            params = {"limit": min(100, limit - len(accounts)), "properties": "domain,name,numberofemployees,industry"}
            if after:
                params["after"] = after
            r = requests.get(f"{self.BASE_URL}/crm/v3/objects/companies", params=params,
                             headers=self.headers, timeout=15)
            r.raise_for_status()
            body = r.json()
            for obj in body.get("results", []):
                props = obj.get("properties", {})
                if props.get("domain"):
                    employees = props.get("numberofemployees")
                    accounts.append(Account(
                        domain=props["domain"], company=props.get("name"), crm_id=obj["id"],
                        employees=int(employees) if employees and str(employees).isdigit() else None,
                        vertical=props.get("industry"),
                    ))
            after = body.get("paging", {}).get("next", {}).get("after")
            if not after:
                break
        return accounts[:limit]

    def record_signal(self, record: SignalRecord) -> Optional[str]:
        if not record.account.crm_id:
            return None
        body = {
            "properties": {
                "hs_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "hs_note_body": _note_html(record),
            },
            "associations": [{
                "to": {"id": record.account.crm_id},
                "types": [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": self.NOTE_TO_COMPANY}],
            }],
        }
        r = requests.post(f"{self.BASE_URL}/crm/v3/objects/notes", json=body, headers=self.headers, timeout=15)
        r.raise_for_status()
        return r.json().get("id")


def _note_html(record: SignalRecord) -> str:
    c = record.card
    parts = [
        f"<p><strong>SignalProto: {c.signal_type} signal</strong> (confidence {c.confidence:.2f})</p>",
        f"<p>&ldquo;{html.escape(c.snippet)}&rdquo;<br/>Source: {html.escape(str(c.canonical_url))}</p>",
    ]
    if record.email:
        e = record.email
        parts.append(f"<p><strong>Draft:</strong> {html.escape(e.subject)}<br/>"
                     f"{html.escape(e.body).replace(chr(10), '<br/>')}</p>")
    return "".join(parts)


def make_crm(cfg: CRMSettings) -> CRMAdapter:
    if cfg.provider == "hubspot":
        if not cfg.hubspot_token:
            raise ValueError("crm.provider=hubspot needs HUBSPOT_TOKEN")
        return HubSpotCRM(cfg.hubspot_token)
    if cfg.provider == "webhook":
        if not cfg.webhook_url:
            raise ValueError("crm.provider=webhook needs crm.webhook_url")
        return WebhookCRM(cfg.webhook_url)
    if cfg.provider == "csv":
        return CSVCRM(cfg.csv_accounts_path, cfg.csv_output_path)
    return NoCRM()
