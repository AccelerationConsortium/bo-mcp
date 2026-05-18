"""Streaming parsers for tabular result uploads (TODO 8.17).

These tests verify the two correctness properties the audit called
out:

1. ``openpyxl`` is invoked in *read-only* mode so the worksheet is
   iterated rather than materialised in full. Pinning the keyword
   argument with a spy means the test fails if a future refactor
   switches back to pandas / the default loader.
2. Library exceptions never leak through to the caller as exception
   strings — they are converted to a typed :class:`UploadParseError`
   carrying a sanitized, library-agnostic message.

Reference: openpyxl optimised reader
https://openpyxl.readthedocs.io/en/stable/optimized.html
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

import api.upload_parser as upload_parser
from api.upload_parser import UploadParseError, parse_csv_rows, parse_upload_rows


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _xlsx_bytes(rows: list[dict[str, object]]) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def test_parse_csv_returns_headers_and_rows() -> None:
    payload = _csv_bytes([{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}])
    headers, rows = parse_csv_rows(payload)
    assert headers == ["x", "y"]
    assert [r["x"] for r in rows] == ["0.1", "0.3"]


def test_parse_csv_skips_extra_unmapped_columns() -> None:
    # ``DictReader`` would otherwise leave ``None`` as a key for any
    # field past the header row; the parser drops it so downstream
    # consumers can rely on ``str`` keys.
    payload = b"x,y\n0.1,0.2,extra\n"
    _, rows = parse_csv_rows(payload)
    assert None not in rows[0]


def test_parse_xlsx_uses_read_only_mode(monkeypatch) -> None:
    """openpyxl is loaded with ``read_only=True`` — pin the keyword.

    Switching back to ``load_workbook(buffer)`` (without the flag)
    would silently re-introduce the O(cells) memory hit the audit
    flagged.
    """
    captured_kwargs: dict[str, object] = {}
    real_loader = upload_parser.load_workbook

    def spy(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(upload_parser, "load_workbook", spy)
    payload = _xlsx_bytes([{"x": 0.1, "y": 0.2}])
    headers, rows = parse_upload_rows("test.xlsx", payload)

    assert captured_kwargs.get("read_only") is True
    assert captured_kwargs.get("data_only") is True
    assert headers == ["x", "y"]
    assert rows[0]["x"] == pytest.approx(0.1)


def test_parse_unsupported_extension_raises() -> None:
    with pytest.raises(UploadParseError):
        parse_upload_rows("results.txt", b"not a csv")


def test_parse_corrupt_xlsx_raises_sanitized_error() -> None:
    """A corrupt workbook surfaces the typed error, not the library trace."""
    with pytest.raises(UploadParseError) as exc:
        parse_upload_rows("results.xlsx", b"not actually an xlsx file")
    assert "openpyxl" not in str(exc.value).lower()
    assert "Excel" in str(exc.value)


def test_parse_corrupt_csv_raises_sanitized_error() -> None:
    """Invalid UTF-8 should be rejected with a sanitized message."""
    # 0xff is invalid UTF-8 (and not a valid utf-8-sig BOM byte).
    with pytest.raises(UploadParseError) as exc:
        parse_csv_rows(b"\xffnot,utf8\n1,2\n")
    assert "csv" in str(exc.value).lower() or "CSV" in str(exc.value)
