# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""
ingestion/base.py
==================

Common types shared by every data-source connector (local files, Azure Blob
Storage, AWS S3, ...), and the conversion from ingested documents into the
pipeline's ``main.Corpus`` shape.

Every connector in this package ultimately produces a list of ``IngestedDoc``
- (title, full text, origin) - regardless of where the bytes came from. That
lets the rest of the app (domain tuning, main.run_pipeline) stay completely
source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IngestedDoc:
    """One ingested document, already decoded to plain text."""
    title: str            # human-readable name (filename, blob name, S3 key)
    text: str              # full extracted text
    origin: str            # provenance string, e.g. "local:report.pdf",
                            # "azure://container/blob.pdf", "s3://bucket/key.txt"
    n_chars: int = field(init=False)

    def __post_init__(self):
        self.n_chars = len(self.text or "")


def docs_to_corpus(
    docs: List[IngestedDoc],
    chunk_size: int = 300,
    overlap: int = 20,
):
    """
    Convert ingested documents into a ``main.Corpus`` by chunking each
    document's text (reusing ``preprocess.chunk_text``, the same chunker the
    rest of the pipeline already uses) and giving each chunk a unique
    per-chunk URL so pairing/URL-differ logic behaves correctly (see the
    WikiContradiction loader for the same pattern and why it matters).

    Import of ``main`` and ``preprocess`` is local to avoid a circular import
    (main.py imports from this ingestion package for its own default loader
    wiring is NOT required - the app calls this directly).
    """
    from preprocess import chunk_text
    from data_loader import clean_wiki_text
    import main as M

    chunks: List[str] = []
    urls: List[str] = []
    titles: List[str] = []
    scopes: List[str] = []

    for doc in docs:
        if not doc.text or not doc.text.strip():
            continue
        # BUGFIX: chunk_text used to run directly on doc.text, which for any
        # wikitext-sourced corpus (WikiContradiction CSVs, or anything else
        # carrying raw MediaWiki markup) means chunks were being cut out of
        # {{templates}}, [[wikilinks]], citation refs, and category tags
        # instead of prose - candidate pairs ended up comparing things like
        # a bare "[[Category:X]]" tag against an unrelated references list,
        # rather than actual sentences. clean_wiki_text (already used by
        # data_loader.load_wiki_contradiction, just never wired into THIS
        # ingestion path before) strips markup into prose first; it's a
        # near-no-op on already-clean text (PDF/DOCX/TXT extraction, or a
        # CSV whose text column never had markup), so applying it
        # unconditionally here is safe for every source, not just wikitext.
        # NOTE: cleaning happens on a local copy for chunking ONLY -
        # doc.text itself is left untouched, since ground-truth detection
        # (eval_metrics.detect_ground_truth) needs the raw
        # {{Self-contradictory}} template markup to still be there to find.
        clean_text = clean_wiki_text(doc.text)
        if not clean_text.strip():
            continue
        pieces = chunk_text(clean_text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(piece)
            urls.append(f"{doc.origin}#chunk{i}")
            titles.append(doc.title)
            scopes.append("")

    return M.Corpus(chunks=chunks, urls=urls, titles=titles, scope_categories=scopes)


def summarize_docs(docs: List[IngestedDoc]) -> List[dict]:
    """Small table-friendly summary of ingested docs, for UI display."""
    return [
        {"title": d.title, "origin": d.origin, "characters": d.n_chars}
        for d in docs
    ]


def sample_texts(docs: List[IngestedDoc], max_chars: int = 400, max_docs: int = 24,
                 total_chars: int = 9000, seed: int = 0) -> List[str]:
    """
    A small, cheap but REPRESENTATIVE sample of the corpus for the domain-
    tuning meta-prompt (we do NOT send the whole corpus just to derive
    guidance bullets).

    Takes a short excerpt (<= max_chars) from each of up to ``max_docs``
    documents spread evenly across the corpus, each labelled with its title,
    within a ``total_chars`` budget; deterministic for a given ``seed``. The
    previous version took the first 5 documents and the caller then cut the
    joined text to 3,000 characters - in practice only the opening of the
    FIRST document, so guidance was fitted to one document (confdoc003:
    fossil-primate rules applied to a corpus of films, battles and sports).

    Each excerpt starts at a random sentence boundary rather than always at
    the top, so the sample isn't dominated by lead paragraphs.
    """
    import random
    rng = random.Random(seed)
    usable = [d for d in docs if (d.text or "").strip()]
    if not usable:
        return []
    n = min(max_docs, len(usable))
    # Evenly spaced picks so the sample spans the whole corpus, not a prefix.
    picks = [usable[int(i * len(usable) / n)] for i in range(n)]
    per_doc = max(150, min(max_chars, total_chars // n))
    out = []
    for d in picks:
        t = " ".join(d.text.split())
        if len(t) > per_doc:
            starts = [0] + [m + 2 for m in range(len(t) - per_doc)
                            if t[m] in ".!?" and t[m + 1] == " "][:200]
            st = rng.choice(starts)
            t = t[st: st + per_doc]
        out.append(f"[{d.title}] {t}")
    return out
