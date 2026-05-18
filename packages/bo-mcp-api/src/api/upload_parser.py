"""Streaming parsers for tabular uploads (CSV / XLSX).

The REST upload route previously read the whole file into memory and
handed it to pandas, which materialises the entire workbook before
producing rows. For workbooks that comfortably exceed the streaming
cap (:data:`api.limits.MAX_UPLOAD_FILE_SIZE_BYTES`) that is still
acceptable, but for the XLSX path it doubles the peak resident set:
once for the bytes buffer, once for the parsed cell graph. Switching
to ``openpyxl(read_only=True, data_only=True)`` and the stdlib ``csv``
module keeps the iteration lazy — no DataFrame is built — so peak
memory tracks the bytes buffer plus the per-row dict rather than the
whole sheet.

Library exceptions raised during parsing are converted to a typed
``UploadParseError`` so the route can return a sanitized message
without leaking ``openpyxl`` / ``csv`` stack frames or version
fingerprints back to the client.

References
----------
* openpyxl "Optimised reader" documentation —
  https://openpyxl.readthedocs.io/en/stable/optimized.html
* Python ``csv`` module —
  https://docs.python.org/3/library/csv.html
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from typing import Any

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

# Exception types raised by ``openpyxl`` / stdlib ``csv`` when a file is
# malformed. Centralised so the upload route can sanitize the message
# without depending on the concrete library types at every call site.
# ``BadZipFile`` is the openpyxl-via-zipfile failure mode when the
# bytes are not actually an XLSX container, and inherits from
# ``Exception`` (not ``OSError``), so it must be listed explicitly.
_PARSE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    InvalidFileException,
    zipfile.BadZipFile,
    csv.Error,
    UnicodeDecodeError,
    ValueError,
    OSError,
    KeyError,
)


class UploadParseError(Exception):
    """Raised when an uploaded file cannot be parsed.

    Carries a sanitized, library-agnostic message so the REST layer
    can surface it directly without further filtering. The original
    exception (if any) is attached via ``__cause__`` so server-side
    logs still preserve the diagnostic detail.
    """


def parse_csv_rows(content: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse CSV bytes into headers and a list of row dicts.

    Uses :mod:`csv` (stdlib) so no DataFrame is constructed —
    iterating row-by-row keeps peak memory proportional to the
    largest row, not the whole file.
    """
    try:
        text_stream = io.TextIOWrapper(io.BytesIO(content), encoding="utf-8-sig", newline="")
        reader = csv.DictReader(text_stream)
        headers = list(reader.fieldnames or ())
        rows = [_normalize_csv_row(row) for row in reader]
    except _PARSE_EXCEPTIONS as exc:
        raise UploadParseError("Invalid CSV file: could not parse contents.") from exc
    return headers, rows


def parse_xlsx_rows(content: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse XLSX bytes into headers and a list of row dicts.

    Uses ``openpyxl.load_workbook(..., read_only=True, data_only=True)``
    so the worksheet is iterated rather than fully materialised.
    ``data_only=True`` makes openpyxl emit cached cell *values* (not
    formulas), which is what the result-submission flow expects.
    """
    try:
        workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except _PARSE_EXCEPTIONS as exc:
        raise UploadParseError("Invalid Excel file: could not parse contents.") from exc
    try:
        worksheet = workbook.active
        if worksheet is None:
            raise UploadParseError("Excel file does not contain a readable sheet.")
        row_iter: Iterator[tuple[Any, ...]] = worksheet.iter_rows(values_only=True)
        try:
            header_row = next(row_iter)
        except StopIteration:
            return [], []
        headers = [str(cell) if cell is not None else "" for cell in header_row]
        rows = [dict(zip(headers, row, strict=False)) for row in row_iter]
    except _PARSE_EXCEPTIONS as exc:
        raise UploadParseError("Invalid Excel file: could not parse contents.") from exc
    finally:
        workbook.close()
    return headers, rows


def parse_upload_rows(filename: str, content: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    """Dispatch to the CSV or XLSX parser based on the filename extension.

    Returns ``(headers, rows)``. Raises :class:`UploadParseError` for
    unsupported extensions or malformed contents.
    """
    lowered = filename.lower()
    if lowered.endswith(".csv"):
        return parse_csv_rows(content)
    if lowered.endswith((".xlsx", ".xls")):
        return parse_xlsx_rows(content)
    raise UploadParseError("Unsupported file format. Use CSV or Excel.")


def _normalize_csv_row(row: dict[str | None, str | None]) -> dict[str, Any]:
    """Drop ``None`` keys (extra columns past the header) and coerce values.

    ``DictReader`` emits ``None`` as the key for trailing fields that
    have no matching header column. Those would otherwise crash the
    downstream parser, which assumes every key is a ``str``.
    """
    return {key: value for key, value in row.items() if key is not None}
