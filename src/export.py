"""Excel export: the whole pipeline state as a workbook a team without a CRM can live in.

Sheets: Prospects (ranked, one row per account), Signals (every signal, active and stale),
Outreach (every draft and whether it was sent). Text scraped from websites is written as
plain strings, never formulas, so a page containing "=HYPERLINK(...)" can't execute in Excel.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from src.graph import Pipeline
from src.watch import ranked_prospects

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

HEADER_FONT = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="2F6FDE")
TIER_FILLS = {
    "hot": PatternFill("solid", fgColor="FDEEE5"),
    "warm": PatternFill("solid", fgColor="FBF3DC"),
    "disqualified": PatternFill("solid", fgColor="EEEEEE"),
}
FORMULA_PREFIXES = ("=", "+", "-", "@")


def _excel_time(dt: Optional[datetime]) -> Optional[datetime]:
    """Excel has no time zones: store naive UTC (headers say so)."""
    if dt is None:
        return None
    return (dt.astimezone(timezone.utc) if dt.tzinfo else dt).replace(tzinfo=None, microsecond=0)


def _write_row(ws: Worksheet, values: list[Any]) -> None:
    ws.append([None] * len(values))
    row = ws.max_row
    for col, value in enumerate(values, start=1):
        cell = ws.cell(row=row, column=col)
        cell.value = value
        if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
            cell.data_type = "s"  # keep as literal text, never a formula
            cell.quotePrefix = True


def _sheet(wb: Workbook, title: str, columns: list[tuple[str, int]], first: bool = False) -> Worksheet:
    ws = wb.active if first else wb.create_sheet()
    ws.title = title
    ws.append([name for name, _ in columns])
    for i, (_, width) in enumerate(columns, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width
        cell = ws.cell(row=1, column=i)
        cell.font, cell.fill = HEADER_FONT, HEADER_FILL
        cell.alignment = Alignment(vertical="center")
    ws.freeze_panes = "A2"
    return ws


def _finish(ws: Worksheet, date_cols: tuple[int, ...] = (), pct_cols: tuple[int, ...] = (),
            link_col: Optional[int] = None, wrap_cols: tuple[int, ...] = ()) -> None:
    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions
    for row in ws.iter_rows(min_row=2):
        for col in date_cols:
            row[col - 1].number_format = "yyyy-mm-dd hh:mm"
        for col in pct_cols:
            row[col - 1].number_format = "0%"
        for col in wrap_cols:
            row[col - 1].alignment = Alignment(wrap_text=True, vertical="top")
        if link_col and isinstance(row[link_col - 1].value, str) and row[link_col - 1].value.startswith("http"):
            row[link_col - 1].hyperlink = row[link_col - 1].value
            row[link_col - 1].style = "Hyperlink"


def build_workbook(pipeline: Pipeline) -> Workbook:
    store = pipeline.deps.store
    prospects = ranked_prospects(pipeline, include_empty=True)
    company_of = {p.account.domain: p.account.company or p.account.domain for p in prospects}
    wb = Workbook()

    ws = _sheet(wb, "Prospects", [
        ("Rank", 6), ("Company", 26), ("Domain", 28), ("Grade", 7), ("Tier", 12), ("Priority", 9),
        ("Fit", 6), ("Intent", 7), ("Next action", 34), ("Signals", 18), ("Top evidence", 60),
        ("Evidence URL", 36), ("Employees", 10), ("Vertical", 14), ("Last contacted (UTC)", 18),
        ("Latest draft", 12),
    ], first=True)
    for rank, p in enumerate(prospects, start=1):
        top = p.signals[0] if p.signals else None
        drafts = store.outreach(p.account.domain)
        _write_row(ws, [
            rank, p.account.company or "", p.account.domain, p.score.grade, p.score.tier,
            round(p.score.priority), round(p.score.fit), round(p.score.intent), p.score.action,
            ", ".join(sorted({s.signal_type for s in p.signals})),
            top.snippet if top else "", top.url if top else "", p.account.employees, p.account.vertical or "",
            _excel_time(store.last_contacted(p.account.domain)), drafts[0].status if drafts else "",
        ])
        fill = TIER_FILLS.get(p.score.tier)
        if fill:
            for cell in ws[ws.max_row][:9]:
                cell.fill = fill
    _finish(ws, date_cols=(15,), link_col=12)

    ws = _sheet(wb, "Signals", [
        ("Company", 26), ("Domain", 28), ("Type", 11), ("Status", 9), ("Confidence", 11), ("Detail", 28),
        ("Evidence", 70), ("URL", 36), ("First seen (UTC)", 18), ("Last seen (UTC)", 18),
    ])
    for s in sorted(store.signals(status=None), key=lambda s: (s.domain, s.status, -s.confidence)):
        _write_row(ws, [company_of.get(s.domain, s.domain), s.domain, s.signal_type, s.status, s.confidence,
                        s.detail or "", s.snippet, s.url, _excel_time(s.first_seen), _excel_time(s.last_seen)])
    _finish(ws, date_cols=(9, 10), pct_cols=(5,), link_col=8, wrap_cols=(7,))

    ws = _sheet(wb, "Outreach", [
        ("Company", 26), ("Domain", 28), ("Subject", 40), ("Body", 80), ("Call to action", 40),
        ("Status", 9), ("Drafted (UTC)", 18), ("Sent (UTC)", 18),
    ])
    for p in prospects:
        for o in store.outreach(p.account.domain):
            _write_row(ws, [company_of[o.domain], o.domain, o.subject, o.body, o.call_to_action, o.status,
                            _excel_time(o.created_at), _excel_time(o.sent_at)])
    _finish(ws, date_cols=(7, 8), wrap_cols=(4,))
    return wb


def export_bytes(pipeline: Pipeline) -> bytes:
    buffer = BytesIO()
    build_workbook(pipeline).save(buffer)
    return buffer.getvalue()


def export_file(pipeline: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.xlsx")
    build_workbook(pipeline).save(tmp)
    tmp.replace(path)  # atomic: a teammate with the old file open never sees a half-written one
    return path


def default_filename() -> str:
    return f"signalproto-{datetime.now(timezone.utc):%Y-%m-%d}.xlsx"
