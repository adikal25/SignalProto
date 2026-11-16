"""SQLite persistence: accounts, page snapshots, signals, outreach drafts and LLM call traces.

One short-lived connection per operation keeps it safe under FastAPI's threadpool.
Schema changes go through MIGRATIONS (PRAGMA user_version), never by editing an applied step.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from pydantic import BaseModel

from src.schemas import EmailDraft, EvidenceCard, SignalType

MIGRATIONS: list[str] = [
    """
    CREATE TABLE accounts (
        domain      TEXT PRIMARY KEY,
        company     TEXT,
        crm_id      TEXT,
        employees   INTEGER,
        vertical    TEXT,
        updated_at  TEXT NOT NULL
    );
    CREATE TABLE page_snapshots (
        domain       TEXT NOT NULL,
        url          TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        fetched_at   TEXT NOT NULL,
        PRIMARY KEY (domain, url)
    );
    CREATE TABLE signals (
        fingerprint TEXT PRIMARY KEY,
        domain      TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        url         TEXT NOT NULL,
        snippet     TEXT NOT NULL,
        detail      TEXT,
        confidence  REAL NOT NULL,
        explain     TEXT NOT NULL,
        first_seen  TEXT NOT NULL,
        last_seen   TEXT NOT NULL,
        status      TEXT NOT NULL DEFAULT 'active'  -- active | stale
    );
    CREATE INDEX signals_domain ON signals (domain, status);
    CREATE TABLE outreach (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        domain         TEXT NOT NULL,
        fingerprint    TEXT NOT NULL,
        subject        TEXT NOT NULL,
        body           TEXT NOT NULL,
        call_to_action TEXT NOT NULL,
        status         TEXT NOT NULL DEFAULT 'drafted',
        created_at     TEXT NOT NULL
    );
    CREATE TABLE llm_calls (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id       TEXT,
        node           TEXT NOT NULL,
        model          TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        input_hash     TEXT NOT NULL,
        input          TEXT NOT NULL,
        output         TEXT,
        error          TEXT,
        latency_ms     INTEGER NOT NULL,
        created_at     TEXT NOT NULL
    );
    CREATE INDEX llm_calls_trace ON llm_calls (trace_id);
    """,
    # 2: track when a draft was actually sent; drafts alone are not contact.
    """
    ALTER TABLE outreach ADD COLUMN sent_at TEXT;
    """,
    # 3: bring-your-own-key providers: record who served each call and what it cost in tokens.
    """
    ALTER TABLE llm_calls ADD COLUMN provider TEXT NOT NULL DEFAULT 'ollama';
    ALTER TABLE llm_calls ADD COLUMN input_tokens INTEGER;
    ALTER TABLE llm_calls ADD COLUMN output_tokens INTEGER;
    """,
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class AccountRow(BaseModel):
    domain: str
    company: Optional[str] = None
    crm_id: Optional[str] = None
    employees: Optional[int] = None
    vertical: Optional[str] = None
    updated_at: Optional[datetime] = None


class SignalRow(BaseModel):
    fingerprint: str
    domain: str
    signal_type: SignalType
    url: str
    snippet: str
    detail: Optional[str]
    confidence: float
    explain: str
    first_seen: datetime
    last_seen: datetime
    status: str


class OutreachRow(BaseModel):
    id: int
    domain: str
    fingerprint: str
    subject: str
    body: str
    call_to_action: str
    status: str
    created_at: datetime
    sent_at: Optional[datetime] = None


class LLMCallRow(BaseModel):
    id: int
    trace_id: Optional[str]
    node: str
    model: str
    prompt_version: str
    input_hash: str
    input: str
    output: Optional[str]
    error: Optional[str]
    latency_ms: int
    created_at: datetime
    provider: str = "ollama"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class Store:
    def __init__(self, path: str | Path):
        # A file path, not ":memory:": every operation opens its own connection.
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _migrate(self) -> None:
        with self._conn() as c:
            c.execute("PRAGMA journal_mode = WAL")
            version = c.execute("PRAGMA user_version").fetchone()[0]
            for i, sql in enumerate(MIGRATIONS[version:], start=version + 1):
                c.executescript(sql)
                c.execute(f"PRAGMA user_version = {i}")

    # --- accounts ----------------------------------------------------------------

    def upsert_account(self, a: AccountRow) -> None:
        """Insert or update; fields passed as None keep their stored value."""
        with self._conn() as c:
            c.execute(
                """INSERT INTO accounts (domain, company, crm_id, employees, vertical, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(domain) DO UPDATE SET
                     company   = COALESCE(excluded.company, accounts.company),
                     crm_id    = COALESCE(excluded.crm_id, accounts.crm_id),
                     employees = COALESCE(excluded.employees, accounts.employees),
                     vertical  = COALESCE(excluded.vertical, accounts.vertical),
                     updated_at = excluded.updated_at""",
                (a.domain, a.company, a.crm_id, a.employees, a.vertical, utcnow()),
            )

    def get_account(self, domain: str) -> Optional[AccountRow]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM accounts WHERE domain = ?", (domain,)).fetchone()
        return AccountRow(**row) if row else None

    def list_accounts(self) -> list[AccountRow]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM accounts ORDER BY domain").fetchall()
        return [AccountRow(**r) for r in rows]

    # --- page snapshots ----------------------------------------------------------

    def snapshot_hashes(self, domain: str) -> dict[str, str]:
        with self._conn() as c:
            rows = c.execute("SELECT url, content_hash FROM page_snapshots WHERE domain = ?", (domain,)).fetchall()
        return {r["url"]: r["content_hash"] for r in rows}

    def save_snapshots(self, domain: str, hashes: dict[str, str]) -> None:
        now = utcnow()
        with self._conn() as c:
            c.executemany(
                """INSERT INTO page_snapshots (domain, url, content_hash, fetched_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT(domain, url) DO UPDATE SET content_hash = excluded.content_hash,
                                                          fetched_at = excluded.fetched_at""",
                [(domain, url, h, now) for url, h in hashes.items()],
            )

    # --- signals -----------------------------------------------------------------

    def signals(self, domain: Optional[str] = None, status: Optional[str] = "active") -> list[SignalRow]:
        sql, args = "SELECT * FROM signals WHERE 1=1", []
        if domain:
            sql += " AND domain = ?"
            args.append(domain)
        if status:
            sql += " AND status = ?"
            args.append(status)
        with self._conn() as c:
            rows = c.execute(sql + " ORDER BY confidence DESC", args).fetchall()
        return [SignalRow(**r) for r in rows]

    def known_fingerprints(self, domain: str) -> set[str]:
        with self._conn() as c:
            rows = c.execute("SELECT fingerprint FROM signals WHERE domain = ?", (domain,)).fetchall()
        return {r["fingerprint"] for r in rows}

    def upsert_signal(self, domain: str, fingerprint: str, card: EvidenceCard) -> None:
        now = utcnow()
        with self._conn() as c:
            c.execute(
                """INSERT INTO signals (fingerprint, domain, signal_type, url, snippet, detail, confidence,
                                        explain, first_seen, last_seen, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                   ON CONFLICT(fingerprint) DO UPDATE SET
                     confidence = excluded.confidence, explain = excluded.explain,
                     detail = COALESCE(excluded.detail, signals.detail),
                     last_seen = excluded.last_seen, status = 'active'""",
                (fingerprint, domain, card.signal_type, str(card.canonical_url), card.snippet, card.detail,
                 card.confidence, card.explain, now, now),
            )

    def touch_signals_on_pages(self, domain: str, urls: list[str]) -> None:
        """Pages unchanged since last crawl: their active signals are still live."""
        if not urls:
            return
        marks = ",".join("?" * len(urls))
        with self._conn() as c:
            c.execute(f"UPDATE signals SET last_seen = ? WHERE domain = ? AND status = 'active' AND url IN ({marks})",
                      [utcnow(), domain, *urls])

    def mark_stale(self, domain: str, changed_urls: list[str], still_present: set[str]) -> int:
        """Signals on re-read pages that were not detected again are no longer live."""
        if not changed_urls:
            return 0
        marks = ",".join("?" * len(changed_urls))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT fingerprint FROM signals WHERE domain = ? AND status = 'active' AND url IN ({marks})",
                [domain, *changed_urls]).fetchall()
            gone = [r["fingerprint"] for r in rows if r["fingerprint"] not in still_present]
            c.executemany("UPDATE signals SET status = 'stale' WHERE fingerprint = ?", [(f,) for f in gone])
        return len(gone)

    # --- outreach ----------------------------------------------------------------

    def add_outreach(self, domain: str, fingerprint: str, email: EmailDraft) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO outreach (domain, fingerprint, subject, body, call_to_action, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (domain, fingerprint, email.subject, email.body, email.call_to_action, utcnow()),
            )
            return int(cur.lastrowid or 0)

    def mark_outreach_sent(self, outreach_id: int) -> Optional[OutreachRow]:
        with self._conn() as c:
            c.execute("UPDATE outreach SET status = 'sent', sent_at = COALESCE(sent_at, ?) WHERE id = ?",
                      (utcnow(), outreach_id))
            row = c.execute("SELECT * FROM outreach WHERE id = ?", (outreach_id,)).fetchone()
        return OutreachRow(**row) if row else None

    def last_contacted(self, domain: str) -> Optional[datetime]:
        with self._conn() as c:
            row = c.execute("SELECT MAX(sent_at) AS t FROM outreach WHERE domain = ?", (domain,)).fetchone()
        return datetime.fromisoformat(row["t"]) if row and row["t"] else None

    def outreach(self, domain: str) -> list[OutreachRow]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM outreach WHERE domain = ? ORDER BY id DESC", (domain,)).fetchall()
        return [OutreachRow(**r) for r in rows]

    # --- llm calls ---------------------------------------------------------------

    def log_llm_call(self, *, trace_id: Optional[str], node: str, model: str, prompt_version: str,
                     input_hash: str, input: str, output: Optional[str], error: Optional[str],
                     latency_ms: int, provider: str = "ollama", input_tokens: Optional[int] = None,
                     output_tokens: Optional[int] = None) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO llm_calls (trace_id, node, model, prompt_version, input_hash, input, output,
                                          error, latency_ms, created_at, provider, input_tokens, output_tokens)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trace_id, node, model, prompt_version, input_hash, input, output, error, latency_ms, utcnow(),
                 provider, input_tokens, output_tokens),
            )

    def llm_calls(self, trace_id: Optional[str] = None, limit: int = 100) -> list[LLMCallRow]:
        sql, args = "SELECT * FROM llm_calls", []
        if trace_id:
            sql += " WHERE trace_id = ?"
            args.append(trace_id)
        with self._conn() as c:
            rows = c.execute(sql + " ORDER BY id DESC LIMIT ?", [*args, limit]).fetchall()
        return [LLMCallRow(**r) for r in rows]
