"""Measure the signal agent against the keyword baseline on a labeled sentence set.

Both systems share the same keyword candidate stage, so the comparison isolates what the
local-LLM verification adds. A prediction counts as correct only if the signal TYPE matches.

    python -m eval.run_eval --model llama3.2:3b --record          # live model, save outputs to a cassette
    python -m eval.run_eval --model llama3.2:3b --replay --gate   # CI: no model server, fail on regression
    python -m eval.run_eval --model llama3.2:3b --write-baseline  # accept current scores as the new bar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.llm.ollama_runtime import LLM, OllamaChat, OllamaConfig
from src.llm.tracing import RecordingLLM, ReplayLLM
from src.schemas import Candidate
from src.signals.agent import verify
from src.signals.keyword import keyword_classify

DATA = Path(__file__).parent / "signals_labeled.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
CASSETTES_DIR = Path(__file__).parent / "cassettes"
BASELINE = Path(__file__).parent / "baseline.json"


class Example(BaseModel):
    id: int
    label: str
    sentence: str


class Metrics(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def metrics(gold: list[str], pred: list[Optional[str]]) -> Metrics:
    tp = sum(1 for g, p in zip(gold, pred) if p is not None and p == g)
    fp = sum(1 for g, p in zip(gold, pred) if p is not None and p != g)
    fn = sum(1 for g, p in zip(gold, pred) if g != "none" and p != g)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(precision=round(precision, 3), recall=round(recall, 3), f1=round(f1, 3), tp=tp, fp=fp, fn=fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="call the live model and save a cassette")
    mode.add_argument("--replay", action="store_true", help="serve outputs from the cassette (no model server)")
    ap.add_argument("--gate", action="store_true", help="exit 1 if the agent regresses vs. eval/baseline.json")
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    examples = [Example.model_validate_json(line) for line in DATA.read_text().splitlines() if line.strip()]
    slug = args.model.replace(":", "_").replace("/", "_")
    cassette = CASSETTES_DIR / f"{slug}.jsonl"
    llm: LLM
    if args.replay:
        llm = ReplayLLM(cassette)
    else:
        llm = OllamaChat(OllamaConfig(model_id=args.model, max_new_tokens=200, temperature=0.0))
        if args.record:
            llm = RecordingLLM(llm, cassette)

    gold = [e.label for e in examples]
    baseline: list[Optional[str]] = []
    agent: list[Optional[str]] = []
    latencies: list[float] = []
    rows = []
    for e in examples:
        kw = keyword_classify(e.sentence)
        baseline.append(kw)
        pred: Optional[str] = None
        if kw is not None:
            c = Candidate(url="https://example.com/", sentence=e.sentence, context=e.sentence, keyword_type=kw)
            t0 = time.perf_counter()
            verdict = verify(llm, c)
            latencies.append(time.perf_counter() - t0)
            if verdict and verdict.signal_type != "none":
                pred = verdict.signal_type
        agent.append(pred)
        rows.append({"id": e.id, "gold": e.label, "keyword": kw, "agent": pred, "sentence": e.sentence})

    report = {
        "model": args.model,
        "n": len(examples),
        "n_positive": sum(1 for g in gold if g != "none"),
        "keyword_baseline": metrics(gold, baseline).model_dump(),
        "signal_agent": metrics(gold, agent).model_dump(),
        "mean_llm_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "errors": [r for r in rows if (r["agent"] or "none") != r["gold"]],
    }
    out = RESULTS_DIR / f"{slug}.json"
    if not args.replay:  # replay reproduces a recorded run; don't overwrite its latency numbers
        RESULTS_DIR.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2))

    kb, sa = report["keyword_baseline"], report["signal_agent"]
    print(f"model={args.model}  n={report['n']} ({report['n_positive']} positive)")
    print(f"{'':16}{'precision':>10}{'recall':>8}{'f1':>7}{'fp':>5}")
    print(f"{'keyword':16}{kb['precision']:>10}{kb['recall']:>8}{kb['f1']:>7}{kb['fp']:>5}")
    print(f"{'signal agent':16}{sa['precision']:>10}{sa['recall']:>8}{sa['f1']:>7}{sa['fp']:>5}")
    print(f"mean LLM latency: {report['mean_llm_latency_s']}s   details -> {out}")

    baselines = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    if args.write_baseline:
        baselines[args.model] = {k: sa[k] for k in ("precision", "recall", "f1")}
        BASELINE.write_text(json.dumps(baselines, indent=2) + "\n")
        print(f"baseline for {args.model} updated")
    if args.gate:
        sys.exit(gate(sa, baselines.get(args.model), args.tolerance))


def gate(current: dict, baseline: Optional[dict], tolerance: float) -> int:
    if baseline is None:
        print("GATE FAIL: no baseline for this model; run with --write-baseline")
        return 1
    failures = [f"{k} {current[k]} < baseline {baseline[k]} - {tolerance}"
                for k in ("precision", "recall", "f1") if current[k] < baseline[k] - tolerance]
    for f in failures:
        print(f"GATE FAIL: {f}")
    if not failures:
        print("GATE PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    main()
