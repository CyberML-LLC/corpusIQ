# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""The corpusIQ v4 pipeline behind the web UI.

  documents -> clean + split into sentences -> all-mpnet-base-v2 retrieval (top 20 neighbours per sentence)
  -> v4 student filter (fine-tuned DeBERTa-v3, max conflict over both orders) -> LLM judges -> labels

The v4 cutoff of 0.003 is the one that kept all 16 confirmed gold pairs on confdoc004 at about 7,400 of 26,411
candidate pairs. Pairs below it are reported as
"Filtered" (no LLM call). Pairs the student scores as likely duplicates are judged too.
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from data_loader import _is_usable_sentence, clean_wiki_text, split_sentences  # noqa: E402
from ingestion.base import IngestedDoc  # noqa: E402

V4_MODEL = os.environ.get('CORPUSIQ_V4_MODEL', os.path.join(ROOT, 'models/v4_filter'))
EMBED_MODEL = 'sentence-transformers/all-mpnet-base-v2'
_lock = threading.Lock(); _models: Dict = {}


@dataclass
class Options:
    top_k: int = 20                 # neighbours per sentence
    scope: str = 'all'              # 'all' = within and across documents, 'within' = same document only
    v4_cutoff: float = 0.003        # send pairs with v4 conflict score >= cutoff to the judges
    dup_cutoff: float = 0.5         # also send pairs with v4 duplicate score >= this
    max_calls: int = 10000          # cap on judged pairs (highest v4 score first)
    judge_mode: str = 'both'        # both | flash | pro
    domain: str = ''                # optional description of what counts as a conflict in this corpus
    min_words: int = 6
    max_words: int = 80


def device() -> str:
    import torch
    if torch.cuda.is_available(): return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): return 'mps'
    return 'cpu'


def models():
    with _lock:
        if not _models:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            if not os.path.isdir(V4_MODEL):
                raise RuntimeError(f'v4 model not found at {V4_MODEL}. Run `git lfs pull` '
                                   'or set CORPUSIQ_V4_MODEL.')
            dev = device()
            _models['embed'] = SentenceTransformer(EMBED_MODEL, device=dev)
            _models['tok'] = AutoTokenizer.from_pretrained(V4_MODEL)
            _models['v4'] = AutoModelForSequenceClassification.from_pretrained(V4_MODEL).to(dev).eval()
            _models['dev'] = dev
        return _models


def sentences(docs: List[IngestedDoc], o: Options):
    out = []  # (doc index, sentence index within doc, text)
    per_doc = []
    for d, doc in enumerate(docs):
        ss = [s for s in split_sentences(clean_wiki_text(doc.text or '')) if _is_usable_sentence(s, o.min_words, o.max_words)]
        per_doc.append(ss)
        out += [(d, k, s) for k, s in enumerate(ss)]
    return out, per_doc


def retrieve(emb: np.ndarray, doc_of: np.ndarray, o: Options, block: int = 2048):
    """Unique (i, j, cosine) pairs, i < j, from each sentence's top-k neighbours."""
    import torch
    E = torch.from_numpy(emb); n = len(E); k = min(o.top_k, n - 1); pairs = {}
    D = torch.from_numpy(doc_of)
    for s in range(0, n, block):
        sim = E[s:s + block] @ E.T
        rows = torch.arange(s, min(s + block, n))
        sim[torch.arange(len(rows)), rows] = -2  # no self pairs
        if o.scope == 'within': sim[D[rows][:, None] != D[None, :]] = -2
        val, idx = sim.topk(k, dim=1)
        for r, (vs, js) in enumerate(zip(val.tolist(), idx.tolist())):
            i = s + r
            for v, j in zip(vs, js):
                if v <= -1: continue
                key = (i, j) if i < j else (j, i)
                if key not in pairs: pairs[key] = v
    return [(i, j, c) for (i, j), c in pairs.items()]


def v4_scores(a: List[str], b: List[str], progress: Callable[[int, int], None], bs: int = 64):
    import torch
    m = models(); tok, model, dev = m['tok'], m['v4'], m['dev']
    X = list(zip(a, b)) + list(zip(b, a)); P = np.zeros((len(X), 3), dtype=np.float32)
    order = np.argsort([len(x) + len(y) for x, y in X])
    with torch.no_grad():
        for s in range(0, len(X), bs):
            idx = order[s:s + bs]
            enc = tok([X[i][0] for i in idx], [X[i][1] for i in idx], truncation=True, max_length=256,
                      padding=True, return_tensors='pt').to(dev)
            P[idx] = model(**enc).logits.float().softmax(-1).cpu().numpy()
            if (s // bs) % 20 == 0: progress(min(s + bs, len(X)), len(X))
    n = len(a)
    # head order: [contradiction, entailment, neutral]; conflict = max over orders, duplicate = mutual entailment
    return np.maximum(P[:n, 0], P[n:, 0]), np.minimum(P[:n, 1], P[n:, 1])


def run(docs: List[IngestedDoc], o: Options, judge, report: Callable[[str, int, int], None]) -> Dict:
    report('Splitting sentences', 0, 0)
    S, per_doc = sentences(docs, o)
    summary = {'documents': len(docs), 'documents_used': sum(1 for p in per_doc if p), 'sentences': len(S)}
    doc_list = [{'id': d, 'title': doc.title, 'origin': doc.origin, 'sentences': len(per_doc[d])} for d, doc in enumerate(docs)]
    if len(S) < 2:
        return {'summary': {**summary, 'candidate_pairs': 0, 'judged': 0,
                            'counts': {'Conflict': 0, 'Unsure': 0, 'Duplicate': 0, 'No Conflict': 0, 'Filtered': 0}},
                'documents': doc_list, 'items': [], 'options': o.__dict__}
    m = models()
    report('Embedding sentences', 0, len(S))
    emb = m['embed'].encode([s for _, _, s in S], batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)
    report('Finding candidate pairs', 0, 0)
    pairs = retrieve(emb, np.array([d for d, _, _ in S]), o)
    summary['candidate_pairs'] = len(pairs)
    report('Scoring pairs with v4', 0, 2 * len(pairs))
    con, dup = v4_scores([S[i][2] for i, _, _ in pairs], [S[j][2] for _, j, _ in pairs],
                         lambda d, t: report('Scoring pairs with v4', d, t))
    sel = np.where((con >= o.v4_cutoff) | (dup >= o.dup_cutoff))[0]
    sel = sel[np.argsort(-np.maximum(con[sel], dup[sel]))][:o.max_calls]
    summary['judged'] = int(len(sel))
    title = lambda d: docs[d].title

    def ctx(d, k):
        ss = per_doc[d]; return [ss[k - 1] if k > 0 else '', ss[k + 1] if k + 1 < len(ss) else '']

    items = []
    for n in sel:
        i, j, c = pairs[n]; (da, ka, a), (db, kb, b) = S[i], S[j]
        items.append({'id': int(n), 'doc_a': da, 'doc_b': db, 'title_a': title(da), 'title_b': title(db), 'a': a, 'b': b,
                      'sent_a': ka, 'sent_b': kb, 'ctx_a': ctx(da, ka), 'ctx_b': ctx(db, kb),
                      'cos': round(float(c), 4), 'v4': round(float(con[n]), 5), 'v4_dup': round(float(dup[n]), 5)})
    report('Asking the LLM judges', 0, len(items) * len(judge.names))
    verdicts = judge.run(items, lambda d, t: report('Asking the LLM judges', d, t))
    for it, v in zip(items, verdicts): it.update(v)
    counts = {k: sum(1 for it in items if it['label'] == k) for k in ('Conflict', 'Unsure', 'Duplicate', 'No Conflict')}
    counts['Filtered'] = len(pairs) - len(items)
    summary['counts'] = counts; summary['judge_errors'] = sum(1 for it in items for j in it['judges'].values() if 'error' in j)
    rank = {'Conflict': 0, 'Unsure': 1, 'Duplicate': 2, 'No Conflict': 3}
    items.sort(key=lambda it: (rank[it['label']], -it['v4']))
    return {'summary': summary, 'documents': doc_list, 'items': items, 'options': o.__dict__,
            'pipeline': {'retrieval': f'{EMBED_MODEL}, top {o.top_k}', 'filter': f'v4 student >= {o.v4_cutoff}',
                         'judges': [judge_model for judge_model in (j for j in judge.names)]}}
