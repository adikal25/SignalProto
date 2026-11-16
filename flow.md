# SignalProto: code flow

```
            CRM (HubSpot / webhook / CSV)            FastAPI  (src/api/main.py)
                       │ list_accounts                    │ /pipeline/run, /crm/sync, /signals/detect
                       ▼                                  ▼
 ┌──────────────────────────── LangGraph  (src/graph.py) ─────────────────────────────┐
 │ crawl ──► signal_agent ──► gate(min_confidence) ──► outbound_agent ──► crm_sync    │
 │                                   └──► below_gate ─────────────────────► crm_sync  │
 └────────────────────────────────────────────────────────────────────────────────────┘
                                                          │ available_slots / booking_link
                                                          ▼
                                             Booker (Cal.com / link)  ◄── /meetings/book
```

## Nodes

| Node | Module | What it does |
|---|---|---|
| `crawl` | `src/crawler.py`, `src/graph.py` | Reads the homepage, then ranks candidate pages: the homepage's same-site links and `/sitemap.xml` entries scored by `crawl.link_hints` (1 per hint in the URL path, 0.5 if only in the link text), pages stored from earlier runs (+1, so vanished signals can go stale), and `crawl.paths` as a fallback (0.75). Ties go to newer sitemap `lastmod`, then shorter URLs; at most 2 pages per first path segment before other sections get a turn. Reads up to `max_pages`. Deterministic, no LLM. Skips 404s and pages that duplicate the homepage. Hashes each page against the stored snapshot; unchanged pages are skipped downstream, and their signals get a fresh `last_seen`. |
| `signal_agent` | `src/signals/agent.py` | Signal types come from `signals.types` (built-ins: hiring, expansion; config entries merge onto them). 1) **Recall:** each type's keyword patterns (`src/signals/keyword.py`) flag candidate sentences. 2) **Precision:** each candidate goes to the local model with its context and few-shot examples, under a JSON-schema constraint (`verdict_schema(cfg)`: `SignalVerdict` with `signal_type` narrowed to the configured types plus `none`). The system prompt lists each type's description; the default prompt is byte-identical to the one the eval cassettes were recorded with. The model rejects negations, history, navigation, other word senses and third-party mentions. 3) **Score:** strength + specificity × freshness decay produces an `EvidenceCard`. 4) **Diff:** each card is fingerprinted (`src/signals/changes.py`); unknown fingerprints are *new*, and known signals that vanished from a re-read page go *stale*. |
| gate | `src/graph.py` | Routes to outbound only if the top **new** signal meets `gate.min_confidence`. Known signals never trigger a second email. |
| `outbound_agent` | `src/agents/outbound.py` | The LLM writes the subject, the opener and the value line (`DraftCopy`). Code then appends the meeting ask, built from **real** booker slots or the booking link, so the model never invents a time. A template takes over if the LLM fails. The draft is stored in `outreach`. |
| `crm_sync` | `src/integrations/crm.py` | Runs when `sync_crm=true`: writes the signal and the draft to the CRM, e.g. a HubSpot note associated to the company. |

## Contracts (Pydantic, `src/schemas.py`)
`Page` → `Candidate` → `SignalVerdict` (LLM output) → `EvidenceCard` → `EmailDraft`; `Slot`, `Booking`.

## Adapters
- `CRMAdapter` protocol: `list_accounts(limit)`, `record_signal(record)`. Implementations: `HubSpotCRM`, `WebhookCRM`, `CSVCRM`, `NoCRM`.
- `Booker` protocol: `available_slots()`, `booking_link()`, `book(start, name, email, tz)`. Implementations: `CalComBooker`, `LinkBooker`, `NoBooker`.

Adding Salesforce or Pipedrive means writing one class with those two methods and adding it to `make_crm`.

## State and observability
- `src/store.py`: SQLite with `PRAGMA user_version` migrations. Tables: `accounts`, `page_snapshots`, `signals`, `outreach`, `llm_calls`.
- `src/llm/tracing.py`: `TracedLLM` logs every call (trace id from a context variable set per run, node, model, prompt version, latency, output or error). `RecordingLLM` / `ReplayLLM` back the CI eval gate.
- `src/watch.py`: sweeps all accounts, isolates per-account failures, posts new signals to `alerts.webhook_url`, and ranks prospects using `src/scoring.py`: fit (ICP) × intent (decaying, diminishing-returns signals) → grade, tier, priority, with a contact cooldown read from `outreach.sent_at`.
