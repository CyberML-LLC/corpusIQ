# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""
ingestion/local_files.py
========================

Text extraction for locally-available files: PDF, DOCX, TXT, and CSV.

Each ``extract_text_from_*`` function accepts EITHER a filesystem path OR an
in-memory bytes/BytesIO object, so the Azure/AWS connectors can reuse the same
parsers on downloaded bytes without writing temp files to disk.

CSV is handled differently from the others: instead of one document per file,
each ROW becomes its own statement/document (the natural pairing unit for
conflict detection - e.g. a CSV of policy statements, one per row). See
``extract_rows_from_csv`` and ``load_docs_from_source``.
"""

from __future__ import annotations

import io
import os
from typing import List, Optional, Tuple, Union

from ingestion.base import IngestedDoc

SourceLike = Union[str, bytes, io.BytesIO]


def _as_file_like(source: SourceLike):
    """Normalize bytes into a BytesIO; paths and existing file-likes pass through."""
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    return source


def extract_text_from_pdf(source: SourceLike) -> str:
    """Extract text from a PDF (path, bytes, or file-like)."""
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise ImportError(
            "PDF ingestion requires 'pypdf'. Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(_as_file_like(source))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue  # a single malformed page shouldn't sink the whole doc
    return "\n\n".join(pages).strip()


def extract_text_from_docx(source: SourceLike) -> str:
    """Extract text from a Word .docx (path, bytes, or file-like)."""
    try:
        import docx
    except Exception as exc:
        raise ImportError(
            "DOCX ingestion requires 'python-docx'. Install it with: "
            "pip install python-docx"
        ) from exc

    document = docx.Document(_as_file_like(source))
    parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    # Also pull table cell text - policy/spec documents often keep key facts in tables.
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text and cell.text.strip():
                    parts.append(cell.text.strip())
    return "\n".join(parts).strip()


def extract_text_from_txt(source: SourceLike, encoding: str = "utf-8") -> str:
    """Read plain text (path, bytes, or file-like), tolerating bad bytes."""
    if isinstance(source, (bytes, bytearray)):
        return source.decode(encoding, errors="replace").strip()
    if isinstance(source, io.BytesIO):
        return source.read().decode(encoding, errors="replace").strip()
    with open(source, "r", encoding=encoding, errors="replace") as f:
        return f.read().strip()


_TEXT_COLUMN_CANDIDATES = ["text", "statement", "content", "body", "description", "summary", "value"]
_TITLE_COLUMN_CANDIDATES = ["title", "name", "id", "doc_id", "heading", "label", "key"]


def _pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Case-insensitive exact match first, then substring match."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for cand in candidates:
        for col_lower, col_orig in lower.items():
            if cand in col_lower:
                return col_orig
    return None


def extract_rows_from_csv(source: SourceLike) -> List[Tuple[str, str]]:
    """
    Parse a CSV into (title, text) tuples, one per row.

    Looks for a text-bearing column by common name (text/statement/content/
    body/description/summary/value); falls back to concatenating all columns
    as "col: value | col: value" if none match. Looks for a title-bearing
    column similarly (title/name/id/...); falls back to a row index.
    Empty/blank text rows are skipped.
    """
    import pandas as pd

    src = _as_file_like(source)
    try:
        df = pd.read_csv(src, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(src, "seek"):
            src.seek(0)
        df = pd.read_csv(src, dtype=str, keep_default_na=False, encoding="latin-1")

    text_col = _pick_column(list(df.columns), _TEXT_COLUMN_CANDIDATES)
    title_col = _pick_column(list(df.columns), _TITLE_COLUMN_CANDIDATES)

    rows: List[Tuple[str, str]] = []
    for i, row in df.iterrows():
        if text_col:
            text = str(row[text_col]).strip()
        else:
            text = " | ".join(
                f"{c}: {row[c]}" for c in df.columns if str(row[c]).strip()
            )
        if not text:
            continue
        title = str(row[title_col]).strip() if title_col else ""
        title = title or f"row{i}"
        rows.append((title, text))
    return rows


_EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
    ".md": extract_text_from_txt,
}


_MULTI_DOC_EXTENSIONS = {".csv"}
_SUPPORTED_EXTENSIONS = sorted(set(_EXTRACTORS) | _MULTI_DOC_EXTENSIONS)


def extract_text(source: SourceLike, filename: str) -> str:
    """
    Dispatch to the right single-string extractor based on filename extension.
    NOTE: does not handle .csv (which yields multiple documents, one per row) -
    use extract_docs / load_local_file(s) for a source that may be CSV.
    """
    ext = os.path.splitext(filename)[1].lower()
    fn = _EXTRACTORS.get(ext)
    if fn is None:
        raise ValueError(
            f"Unsupported file type {ext!r} for {filename!r}. "
            f"Supported: {_SUPPORTED_EXTENSIONS}"
        )
    return fn(source)


def extract_docs(source: SourceLike, filename: str, origin: str) -> List[IngestedDoc]:
    """
    Dispatch to the right extractor and return a LIST of IngestedDoc.

    Most formats (PDF/DOCX/TXT/MD) produce exactly one document per file. CSV
    is the exception: each ROW becomes its own IngestedDoc (the natural
    pairing unit for conflict detection - e.g. a CSV of policy statements,
    one per row), with origin ``{origin}#row{i}`` and title taken from a
    title-like column if present, else the base filename + row index.

    This is the single dispatch point local files AND the Azure/AWS
    connectors funnel through, so CSV support is automatically consistent
    across every data source.
    """
    ext = os.path.splitext(filename)[1].lower()
    base = os.path.basename(filename)

    if ext == ".csv":
        rows = extract_rows_from_csv(source)
        docs = []
        for i, (title, text) in enumerate(rows):
            # extract_rows_from_csv already guarantees a non-empty title
            # (falling back to "rowN" itself), so just prefix with the filename.
            doc_title = f"{base}:{title}" if not title.startswith("row") else f"{base}#{title}"
            docs.append(IngestedDoc(title=doc_title, text=text, origin=f"{origin}#row{i}"))
        return docs

    if ext not in _EXTRACTORS:
        raise ValueError(
            f"Unsupported file type {ext!r} for {filename!r}. "
            f"Supported: {_SUPPORTED_EXTENSIONS}"
        )
    text = extract_text(source, filename)
    return [IngestedDoc(title=base, text=text, origin=origin)]


def load_local_file(path: str) -> List[IngestedDoc]:
    """Load a single local file into one or more IngestedDoc (CSV -> many)."""
    filename = os.path.basename(path)
    return extract_docs(path, filename, origin=f"local:{filename}")


def load_local_files(paths: List[str]) -> List[IngestedDoc]:
    """
    Load multiple local files, skipping ones that fail with a printed warning.
    A single CSV file can expand into many IngestedDoc (one per row).
    """
    docs: List[IngestedDoc] = []
    for p in paths:
        try:
            docs.extend(load_local_file(p))
        except Exception as exc:
            print(f"[ingestion] WARNING: could not load {p!r}: {exc}")
    return docs
