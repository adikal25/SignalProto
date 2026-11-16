"""Excel export."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO

from fastapi.testclient import TestClient
from openpyxl import load_workbook

from src.api.main import create_app
from src.export import XLSX_MEDIA_TYPE, export_bytes, export_file
from src.integrations.crm import Account
from tests.test_product_loop import ACME, CAREERS, HOME, Site, make_pipeline, store  # noqa: F401 (fixture)


def _rows(ws) -> list[dict]:
    header = [c.value for c in ws[1]]
    return [dict(zip(header, (c.value for c in row))) for row in ws.iter_rows(min_row=2)]


def test_workbook_has_ranked_prospects_signals_and_outreach(store):
    site = Site({"/": HOME, "/careers": CAREERS})
    p = make_pipeline(store, site)
    p.run(ACME)
    [draft] = store.outreach("acme.test")
    store.mark_outreach_sent(draft.id)
    site.pages["/"] = "<h1>Welcome!</h1>"  # expansion goes stale
    p.run(ACME)

    wb = load_workbook(BytesIO(export_bytes(p)))
    assert wb.sheetnames == ["Prospects", "Signals", "Outreach"]

    [prospect] = _rows(wb["Prospects"])
    assert prospect["Company"] == "Acme" and prospect["Rank"] == 1
    assert prospect["Grade"] and prospect["Tier"] and prospect["Signals"] == "hiring"
    assert isinstance(prospect["Last contacted (UTC)"], datetime)
    assert wb["Prospects"].freeze_panes == "A2" and wb["Prospects"].auto_filter.ref

    signals = _rows(wb["Signals"])
    assert sorted((s["Type"], s["Status"]) for s in signals) == [("expansion", "stale"), ("hiring", "active")]
    assert wb["Signals"]["H2"].hyperlink is not None

    [outreach] = _rows(wb["Outreach"])
    assert outreach["Status"] == "sent" and isinstance(outreach["Sent (UTC)"], datetime)


def test_scraped_text_is_never_written_as_a_formula(store):
    evil = '<h1>=HYPERLINK("http://evil.example","We\'re hiring a nurse")</h1>'
    p = make_pipeline(store, Site({"/": evil}))
    p.run(ACME)
    wb = load_workbook(BytesIO(export_bytes(p)))
    cell = wb["Signals"]["G2"]
    assert cell.value.startswith("=HYPERLINK") and cell.data_type == "s"


def test_empty_database_exports_headers_only(store):
    p = make_pipeline(store, Site({}))
    wb = load_workbook(BytesIO(export_bytes(p)))
    assert all(wb[name].max_row == 1 for name in wb.sheetnames)


def test_export_file_is_atomic_and_overwrites(store, tmp_path):
    p = make_pipeline(store, Site({"/": HOME}))
    p.run(Account(domain="acme.test", company="Acme"))
    out = tmp_path / "reports" / "prospects.xlsx"
    export_file(p, out)
    export_file(p, out)  # second write replaces the first
    assert load_workbook(out).sheetnames[0] == "Prospects"
    assert [f.name for f in out.parent.iterdir()] == ["prospects.xlsx"]  # no temp file left behind


def test_api_download(store):
    p = make_pipeline(store, Site({"/": HOME}))
    p.run(ACME)
    r = TestClient(create_app(p.deps)).get("/export.xlsx")
    assert r.status_code == 200 and r.headers["content-type"] == XLSX_MEDIA_TYPE
    assert r.headers["content-disposition"].startswith('attachment; filename="signalproto-')
    assert load_workbook(BytesIO(r.content)).sheetnames == ["Prospects", "Signals", "Outreach"]
