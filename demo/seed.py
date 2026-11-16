"""Seed a demo database by running the REAL pipeline (local model included) on fixture sites.

The sites under demo/sites/ are fictional (.example domains). They mix real signals with
keyword traps ("not hiring", "apply sunscreen", "opened in 2009"), and Bright Path Dental
changes between two runs to exercise change detection.

    python -m demo.seed                                   # writes data/demo.db
    SIGNALPROTO_DB=data/demo.db uvicorn src.api.main:app  # browse it at http://localhost:8000
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.graph import Deps, Pipeline
from src.integrations.crm import account_from_row
from src.settings import load_settings

DEMO = Path(__file__).parent
SITES = DEMO / "sites"


def fixture_fetcher(version_suffix: dict[str, str]):
    def fetch(url: str) -> str:
        parsed = urlparse(url)
        host = parsed.netloc + version_suffix.get(parsed.netloc, "")
        name = parsed.path.strip("/") or "index"
        page = SITES / host / f"{name}.html"
        if not page.exists():
            raise requests.HTTPError(f"404 {url}")
        return page.read_text()
    return fetch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/demo.db")
    args = ap.parse_args()

    Path(args.db).unlink(missing_ok=True)
    settings = load_settings()
    settings.storage.db_path = args.db
    settings.crawl.paths = ["/", "/careers", "/news"]
    deps = Deps.from_settings(settings)
    with (DEMO / "accounts.csv").open() as f:
        accounts = [account_from_row(r) for r in csv.DictReader(f)]

    for label, versions in [("week 1", {}), ("week 2", {"brightpathdental.example": ".v2"})]:
        deps.fetcher = fixture_fetcher(versions)
        pipeline = Pipeline(deps)
        print(f"--- {label}")
        for account in accounts:
            r = pipeline.run(account)
            print(f"{account.company:<22} changed={len(r.changed_urls)} new={len(r.new_signals)} "
                  f"stale={r.stale_count} -> {'email drafted' if r.email else r.skipped_reason}")


if __name__ == "__main__":
    main()
