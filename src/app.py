"""CLI.

python -m src.app run --csv data/companies.sample.csv     # track + run these accounts once
python -m src.app watch [--sync-crm]                      # the scheduled loop (cron / launchd)
python -m src.app prospects                               # ranked list
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys

from src.graph import Deps, Pipeline
from src.integrations.crm import account_from_row
from src.settings import load_settings
from src.watch import ranked_prospects, run_watch


def cmd_run(pipeline: Pipeline, args: argparse.Namespace) -> None:
    with open(args.csv, newline="") as f:
        accounts = [account_from_row(r) for r in csv.DictReader(f)]
    for account in accounts[: args.limit]:
        r = pipeline.run(account, sync_crm=args.sync_crm, force=args.force)
        new = len(r.new_signals)
        status = "email drafted" if r.email else r.skipped_reason
        print(f"{account.domain}: {len(r.detected)} signal(s), {new} new, {status}  [trace {r.trace_id}]")


def cmd_watch(pipeline: Pipeline, args: argparse.Namespace) -> None:
    report = run_watch(pipeline, sync_crm=args.sync_crm)
    print(f"checked {report.accounts_checked} accounts, {report.pages_changed} changed pages, "
          f"{len(report.new_signals)} new signals, {report.stale_signals} went stale, "
          f"alert {'sent' if report.alert_sent else 'not sent'}")
    for n in report.new_signals:
        print(f"  + {n.company or n.domain}: {n.card.signal_type} {n.card.confidence:.2f}  {n.card.snippet[:100]}")
    for domain, err in report.errors.items():
        print(f"  ! {domain}: {err}", file=sys.stderr)


def cmd_prospects(pipeline: Pipeline, args: argparse.Namespace) -> None:
    rows = ranked_prospects(pipeline)
    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in rows], indent=2))
        return
    for p in rows:
        types = ", ".join(sorted({s.signal_type for s in p.signals}))
        print(f"{p.score.grade:<3} {p.score.tier:<12} {p.score.priority:5.1f}  "
              f"{p.account.company or p.account.domain:<28} {types:<20} {p.score.action}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.app")
    ap.add_argument("--config", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="track accounts from a CSV and run the pipeline once")
    run.add_argument("--csv", required=True, help="columns: domain, company, [employees], [vertical]")
    run.add_argument("--limit", type=int, default=100)
    run.add_argument("--sync-crm", action="store_true")
    run.add_argument("--force", action="store_true", help="re-read pages even if unchanged")

    watch = sub.add_parser("watch", help="re-check all tracked accounts; alert on new signals")
    watch.add_argument("--sync-crm", action="store_true")

    prospects = sub.add_parser("prospects", help="print the ranked prospect list")
    prospects.add_argument("--json", action="store_true")

    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    pipeline = Pipeline(Deps.from_settings(load_settings(args.config)))
    {"run": cmd_run, "watch": cmd_watch, "prospects": cmd_prospects}[args.command](pipeline, args)


if __name__ == "__main__":
    main()
