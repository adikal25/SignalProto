# SignalProto

Open-source multi-agent prospecting on **local models**. A **signal agent** watches prospect websites for hiring and expansion triggers. An **outbound agent** drafts a grounded email that offers **real meeting slots**. Results are ranked, traced and written back to your **CRM**. Everything runs behind a **FastAPI** REST API on local Ollama models by default, so there is no per-call API cost. Or **bring your own key** for Claude, OpenAI, Groq, OpenRouter, Gemini or any OpenAI-compatible server, separately for each agent.

```
CRM accounts ─► watch (daily) ─► changed pages only ─► signal agent ─► new signals ─► gate ─► outbound agent (+ live slots) ─► CRM note / alert
                                                                         └─► fit × intent score ─► A1-D4 grade ─► ranked dashboard
```

![Prospects dashboard](docs/screenshots/dashboard.png)

<sub>Demo data from `python -m demo.seed`: fictional `.example` sites run through the real pipeline and a local `llama3.2:3b`. Two runs ten days apart: Bright Path Dental changed in between, so its new hiring post was picked up and the old announcement marked stale. Oak & Iron's draft was sent 8 days ago, so it sits in cooldown below accounts you can act on now. Summit Roofing grades D because 850 employees is outside the ICP. Copper Kettle Bakery is missing because all of its keyword hits ("all positions have been filled", "our client just opened a new location") were rejected.</sub>

![Agent activity](docs/screenshots/activity.png)

<sub>Every LLM call is traced with its agent, model, prompt version and latency, and shows whether the candidate was accepted or rejected.</sub>

## Signal agent vs. keyword matching

Both systems share the same keyword list, so they flag the same candidate sentences. The difference is that the agent asks a local model to verify each candidate in context.

Results on `eval/signals_labeled.jsonl`: 80 sentences, 40 of them real signals. A prediction counts only if the signal **type** is also right.

| | Precision | Recall | F1 | False positives |
|---|---|---|---|---|
| Keyword matching | 0.52 | 0.98 | 0.68 | 36 |
| Signal agent, `llama3.2:1b` | 0.84 | 0.80 | 0.82 | 6 |
| **Signal agent, `llama3.2:3b`** | **0.95** | **0.88** | **0.91** | **2** |

Mean verification latency is about 0.6 s per candidate on an Apple-silicon laptop. Per-example errors are in `eval/results/`.

> Caveat: the eval set is small and hand-written, and the keyword list was widened after seeing it (that helps the baseline's recall as much as the agent's). Treat these numbers as directional. Grow the set with labeled sentences from real crawls before quoting them anywhere.

## The product loop

**Change detection.** Each run hashes every page. Unchanged pages skip the LLM entirely, and their signals just get a fresh `last_seen` date. Signals are fingerprinted by domain, type and normalised snippet (dates and punctuation stripped), so only genuinely **new** signals reach the outbound agent. A signal that disappears from a re-read page is marked **stale**. `watch` sweeps every tracked account, pulls new ones from the CRM, and posts new signals to an alert webhook (Slack, Zapier, n8n).

```bash
# crontab: every weekday at 7am
0 7 * * 1-5  cd /path/to/SignalProto && .venv/bin/python -m src.app watch --sync-crm
```

**Account scoring.** Modelled on how B2B teams score accounts in practice (HubSpot / MadKudu / Marketo-style fit × intent grading), not one blended number:

| | What it measures | How |
|---|---|---|
| **Fit** (0-100, graded A-D) | *Who they are* vs. your ICP | employee size bands + target vertical. Hard disqualifiers: excluded verticals, companies above `disqualify_above` |
| **Intent** (0-100, graded 1-4) | *What they're doing now* | each signal = base value × confidence × relevance × decay. Decay uses a **per-type half-life** (hiring 30d, expansion 60d). Signals stack with **diminishing returns** (1 − ∏(1 − sᵢ)), and a **surge** bonus applies when different signal types land within 30 days. Hiring for roles you don't serve counts at 0.6× (`relevant_keywords`) |
| **Priority** (0-100) | Ranking | √(fit × intent): a great fit with no intent, or loud intent from a bad fit, both rank low. Halved during a **contact cooldown** (21d after a draft is marked sent) |

The grade maps to a tier and a next step: **hot** (A/B fit with intent 1-2) means reach out today, **warm** means add to this week's sequence, **nurture** means keep watching, and **disqualified** means skip. Accounts in cooldown drop below everything you can act on now. Every point added or removed is listed as a reason in the dashboard. All weights and the ICP live under `scoring:` in `configs/config.yml`. Learned weights are the next step once there's outcome data (replies, meetings).

**Observability and the CI eval gate.** Every LLM call is written to the `llm_calls` table and tagged with the run's trace id. The `prompt_version` is a hash of the system prompt plus few-shot examples, so editing a prompt changes it automatically. The eval runs in two modes:

| Workflow | When | How |
|---|---|---|
| `ci.yml` | every PR / push | replays recorded outputs (`eval/cassettes/`) with no model server and gates against `eval/baseline.json`. If a prompt changed without re-recording, it fails loudly. |
| `eval-live.yml` | nightly, on demand, and on prompt or eval changes | installs Ollama, runs `llama3.2:3b` live, gates with a wider tolerance |

```bash
python -m eval.run_eval --model llama3.2:3b --record          # after changing a prompt: re-record live
python -m eval.run_eval --model llama3.2:3b --replay --gate   # what CI runs
python -m eval.run_eval --model llama3.2:3b --write-baseline  # accept new scores as the bar
```

## Features
- **LangGraph pipeline:** crawl → signal agent → gate → outbound agent → CRM sync (see [flow.md](flow.md)).
- **Structured outputs on every provider:** the model is constrained to Pydantic JSON schemas (Ollama `format`, Claude structured outputs, OpenAI-style `response_format`), so there is no regex-parsing of model text.
- **Bring your own key:** per-agent provider choice; keys come from environment variables only (see below).
- **Meeting booking:** Cal.com v2 (live slots spread across days, plus `POST /meetings/book`) or any scheduling link. Slots are inserted by code, so the model can't invent a time.
- **CRM adapters:** HubSpot (pull companies with size and industry, write notes on the company record), a generic webhook (Zapier, Make or n8n into Salesforce, Pipedrive, Attio and others), and CSV. Adding a CRM means implementing two methods.
- **SQLite store** with versioned migrations: accounts, page snapshots, signals, outreach drafts, LLM calls.
- **Dashboard** served by the API at `/`, with no build step and no extra dependencies.
- **Tests:** 64 pytest tests, with every LLM and HTTP call faked.

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b          # Ollama must be running on :11434
```

**See it with demo data**
```bash
python -m demo.seed                                     # runs the real pipeline on fixture sites -> data/demo.db
SIGNALPROTO_DB=data/demo.db uvicorn src.api.main:app    # open http://localhost:8000
```

**Use it for real**
```bash
uvicorn src.api.main:app --reload                       # dashboard at /, API docs at /docs
python -m src.app run --csv data/companies.sample.csv   # columns: domain, company, [employees], [vertical]
python -m src.app watch --sync-crm                      # the scheduled loop
python -m src.app prospects                             # ranked list in the terminal
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/health` | Ollama reachability, models, CRM, booking and alert configuration |
| POST | `/pipeline/run` | Track an account and run it (`force`, `sync_crm` optional) |
| POST | `/watch/run` | Sweep all tracked accounts, alert on new signals |
| GET | `/prospects` | Ranked prospects with score breakdowns |
| GET | `/prospects/detail?domain=` | Signals, stale signals, outreach drafts |
| POST | `/outreach/{id}/sent` | Mark a draft as sent (starts the contact cooldown) |
| GET | `/traces?trace_id=` | Logged LLM calls |
| POST | `/signals/detect` | Stateless crawl + signal agent (nothing stored) |
| GET | `/meetings/slots` | Open slots from the booking provider |
| POST | `/meetings/book` | Book a slot (Cal.com) |

## Models: local or bring your own key

Each agent picks its own provider in `configs/config.yml`, so you can mix them. For example, keep the high-volume signal checks on a free local model and write emails with your own Claude key:

```yaml
llm:
  signal:   {provider: ollama, model_id: llama3.2:3b, max_new_tokens: 200}
  outbound: {provider: anthropic, model_id: claude-opus-5-5, max_new_tokens: 4096, effort: low}
```

| `provider` | Key variable | Notes |
|---|---|---|
| `ollama` (default) | none | Local, no per-call cost |
| `anthropic` | `ANTHROPIC_API_KEY` | Official `anthropic` SDK with structured outputs. `effort` controls reasoning depth. Refusals fall back to another model server-side on models that support it |
| `openai` | `OPENAI_API_KEY` | |
| `groq` | `GROQ_API_KEY` | |
| `openrouter` | `OPENROUTER_API_KEY` | |
| `gemini` | `GEMINI_API_KEY` | Google's OpenAI-compatible endpoint |
| `openai_compatible` | `LLM_API_KEY` or your `api_key_env` | Needs `base_url`: vLLM, LM Studio, Together, a corporate gateway, ... |

- **Keys:** read only from the environment. They are never written to config, the database, logs or API responses. `/health` and the dashboard show only whether each agent's key is set.
- **Misconfiguration:** a hosted agent without its key makes the API answer `503 configuration error: ... needs an API key in $ANTHROPIC_API_KEY` rather than failing mid-run.
- **Usage:** every call logs its provider plus input and output tokens (Agent activity tab, `GET /traces`), so hosted usage is visible per run.
- **Comparing models:** the eval runs against any provider, so you can compare your key's model with the local one on the same labeled set:

```bash
export ANTHROPIC_API_KEY=...
python -m eval.run_eval --provider anthropic --model claude-opus-5-5 --record --write-baseline
```

## Configuration
`configs/config.yml` covers crawl paths, gate, freshness, models, sender, booking, CRM, storage, alerts and scoring/ICP. Secrets come only from the environment:

| Env var | Used for |
|---|---|
| `OLLAMA_BASE_URL` | default `http://localhost:11434` |
| `CALCOM_API_KEY` | `booking.provider: calcom` (plus `calcom_event_type_id`) |
| `HUBSPOT_TOKEN` | `crm.provider: hubspot` (private-app token: companies read, notes write) |
| `SIGNALPROTO_CONFIG` | alternate config path |
| `SIGNALPROTO_DB` | alternate database path (e.g. `data/demo.db`) |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `LLM_API_KEY` | bring-your-own-key providers (see above) |

## Layout
```text
src/
  api/main.py              FastAPI app
  web/index.html           dashboard
  graph.py                 LangGraph pipeline, change detection, dependency wiring
  watch.py                 scheduled sweep, alerts, ranked prospects
  scoring.py               prospect score
  store.py                 SQLite store + migrations
  crawler.py               deterministic fetcher
  signals/keyword.py       keyword baseline / candidate recall stage
  signals/agent.py         signal agent (LLM verification + scoring)
  signals/changes.py       page hashes, signal fingerprints
  agents/outbound.py       outbound agent
  integrations/crm.py      HubSpot / webhook / CSV
  integrations/booking.py  Cal.com / link
  llm/base.py              LLM interface, errors, token usage
  llm/ollama_runtime.py    Ollama client with JSON-schema outputs
  llm/providers.py         bring-your-own-key providers (Anthropic SDK, OpenAI-compatible)
  llm/tracing.py           call tracing, record / replay
eval/                      labeled set, harness, cassettes, baseline, results
demo/                      fixture sites + seed script
tests/                     pytest suite
```
