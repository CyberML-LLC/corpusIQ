# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""
ingestion
=========

Data-source connectors for corpusIQ. Every connector
produces ``IngestedDoc`` objects (title, text, origin), which
``base.docs_to_corpus`` turns into a ``main.Corpus`` ready for the pipeline.

Modules
-------
- base.py         : IngestedDoc, docs_to_corpus, summarize_docs, sample_texts
- local_files.py  : PDF / DOCX / TXT extraction (path or in-memory bytes)
The web UI's S3, Azure and Google Cloud sources are in server/sources.py.
"""

from ingestion.base import IngestedDoc, docs_to_corpus, summarize_docs, sample_texts
from ingestion.local_files import (
    load_local_file, load_local_files, extract_text, extract_docs,
    extract_rows_from_csv,
)

__all__ = [
    "IngestedDoc",
    "docs_to_corpus",
    "summarize_docs",
    "sample_texts",
    "load_local_file",
    "load_local_files",
    "extract_text",
    "extract_docs",
    "extract_rows_from_csv",
]
