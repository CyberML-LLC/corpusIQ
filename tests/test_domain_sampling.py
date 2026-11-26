# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
from ingestion.base import IngestedDoc, sample_texts


def _docs(n=100):
    return [IngestedDoc(title=f"doc{i}", text=f"Doc {i} opens here. " + "".join(f"Fact {i}.{k} sentence. " for k in range(80)), origin="x")
            for i in range(n)]


def test_spans_corpus_not_prefix():
    out = sample_texts(_docs())
    titles = [s.split("]")[0][1:] for s in out]
    assert len(out) == 24 and titles[0] == "doc0" and titles[-1] in ("doc95", "doc96")
    assert sum(len(s) for s in out) < 11000          # fits the meta-prompt budget
    assert sample_texts(_docs(), seed=1) != out       # excerpts vary with seed...
    assert sample_texts(_docs()) == out               # ...but are deterministic per seed


def test_small_corpus_and_empty_docs():
    docs = _docs(3) + [IngestedDoc(title="empty", text="   ", origin="x")]
    out = sample_texts(docs)
    assert len(out) == 3 and all(len(s) <= 400 + 12 for s in out)
    assert sample_texts([]) == []
