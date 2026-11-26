# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""
score_recall.py
===============

Recall of a pipeline run against KNOWN contradictions, stage by stage.

The gold set (eval/recall_set/wikicontradiction_gold_pairs.csv) lists, for
the WikiContradiction revisions that still carry a {{Self-contradictory}}
tag, the two sentences that contradict each other (see
eval/recall_set/README.md for how they were annotated). For every gold pair
this script asks how far the run got:

  extracted   (claims mode) each sentence produced at least one claim
  paired      some candidate pair has one unit from each sentence
  sent_to_llm that pair survived the NLI filter
  stage1_flag Stage 1 called it Conflict or Unclear
  stage2_conflict Stage 2 labelled it Conflict
  final       the final label is Conflict

and reports the share of gold pairs (and of articles: at least one gold pair)
that reach each stage, overall and by annotation confidence. It also reports
the share of located articles with ANY final conflict, which credits catches
made through a different pair (e.g. a statement the article repeats). The first stage
where recall drops is the part of the pipeline to fix.

Matching units to gold sentences (word overlap, not exact text, because
cleaning leaves stray caption/heading text and claims are rewrites):
  - the unit must come from the same article and, for app-ingested CSVs,
    the same revision (URL '#row<N>' == gold app_row);
  - chunk mode: the unit text contains >= 80% of the gold sentence's words;
  - claims mode: the claim's source chunk contains >= 80% of the sentence's
    words AND >= 60% of the claim's words appear in the sentence (the claim
    was extracted from that sentence, not from a neighbour).

Usage
-----
    python score_recall.py ../projects/confdoc004/outputs
    python score_recall.py RUN_DIR --min-confidence medium --out recall.json

Works with new runs (pair_trace.parquet) and older ones (pairs.csv +
pairs_nli_filtered.csv + final_labeled.parquet).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Set

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

GOLD_PATH = os.path.join(_HERE, "recall_set", "wikicontradiction_gold_pairs.csv")
STAGES = ["extracted", "paired", "sent_to_llm", "stage1_flag", "stage2_conflict", "final"]
_CONF_RANK = {"high": 0, "medium": 1, "low": 2}
_STOP = set("""a an the of in on at to for from by with and or but is are was were be been being
as that this these those it its he she they them his her their which who whom whose not no
than then there here into over under about after before during while also has have had do
does did can could may might will would shall should""".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")


def tokens(text: str) -> Set[str]:
    return {t for t in _TOKEN.findall(str(text or "").lower()) if t not in _STOP and len(t) > 1}


def coverage(container: str, part: str) -> float:
    """Share of `part`'s content words that appear in `container`."""
    p = tokens(part)
    return len(p & tokens(container)) / len(p) if p else 0.0


def _row_of(url: str) -> Optional[int]:
    m = re.search(r"#row(\d+)", str(url))
    return int(m.group(1)) if m else None


def _same_article(title: str, url: str, gold) -> bool:
    t = str(title)
    if not (t == gold.title or t.endswith(":" + gold.title)):
        return False
    row = _row_of(url)
    return row is None or pd.isna(gold.app_row) or row == int(gold.app_row)


# ---------------------------------------------------------------------------
def load_run(run_dir: str) -> pd.DataFrame:
    """One row per candidate pair with the columns the scorer needs."""
    trace_path = os.path.join(run_dir, "pair_trace.parquet")
    if os.path.exists(trace_path):
        t = pd.read_parquet(trace_path)
        if "s1_decision" in t.columns:
            return t
    pairs = pd.read_csv(os.path.join(run_dir, "pairs.csv"))
    key = ["statement1", "statement2", "statement1URL", "statement2URL"]
    nli_path = os.path.join(run_dir, "pairs_nli_filtered.csv")
    kept = (pd.read_csv(nli_path, usecols=key) if os.path.exists(nli_path) else pairs[key]).drop_duplicates()
    kept["sent_to_llm"] = True
    pairs = pairs.merge(kept, on=key, how="left")
    pairs["sent_to_llm"] = pairs["sent_to_llm"].astype("boolean").fillna(False).astype(bool)
    lab_path = os.path.join(run_dir, "final_labeled.parquet")
    if os.path.exists(lab_path):
        from run_log import stage_verdicts
        lab = pd.read_parquet(lab_path)
        verd = stage_verdicts(lab)
        lab = pd.concat([lab[key + ["final_result"]].reset_index(drop=True),
                         verd.reset_index(drop=True)], axis=1).drop_duplicates(key)
        pairs = pairs.merge(lab, on=key, how="left")
    return pairs


def _claims_index(run_dir: str) -> Optional[pd.DataFrame]:
    p = os.path.join(run_dir, "claims.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else None


def match_side(text: str, source: Optional[str], gold_sentence: str, claims_mode: bool) -> bool:
    if claims_mode:
        src = source if isinstance(source, str) and source else text
        return coverage(src, gold_sentence) >= 0.8 and coverage(gold_sentence, text) >= 0.6
    return coverage(text, gold_sentence) >= 0.8


def score(run_dir: str, gold_path: str = GOLD_PATH, min_confidence: str = "low") -> Dict:
    gold_all = pd.read_csv(gold_path)
    gold = gold_all[(gold_all["status"] == "located") &
                    (gold_all["confidence"].map(_CONF_RANK) <= _CONF_RANK[min_confidence])].copy()
    pairs = load_run(run_dir)
    claims_mode = "statement1Source" in pairs.columns and pairs["statement1Source"].notna().any()
    claims = _claims_index(run_dir) if claims_mode else None
    results = []
    for g in gold.itertuples():
        in_doc = pairs[[_same_article(t, u, g) or _same_article(t2, u2, g) for t, u, t2, u2 in
                        zip(pairs["statement1Title"], pairs["statement1URL"],
                            pairs["statement2Title"], pairs["statement2URL"])]]
        rec = {"title": g.title, "confidence": g.confidence, "note": g.note,
               "sentence_a": g.sentence_a, "sentence_b": g.sentence_b}
        if claims_mode and claims is not None:
            doc_claims = claims[[_same_article(t, u, g) for t, u in zip(claims["title"], claims["url"])]]
            rec["extracted"] = bool(
                any(match_side(c, s, g.sentence_a, True) for c, s in zip(doc_claims["statement"], doc_claims["source_text"]))
                and any(match_side(c, s, g.sentence_b, True) for c, s in zip(doc_claims["statement"], doc_claims["source_text"])))
        else:
            rec["extracted"] = None  # chunk mode: no extraction stage
        src1 = in_doc["statement1Source"] if claims_mode else [None] * len(in_doc)
        src2 = in_doc["statement2Source"] if claims_mode else [None] * len(in_doc)
        hits = []
        for (i, r), s1, s2 in zip(in_doc.iterrows(), src1, src2):
            a1 = match_side(r["statement1"], s1, g.sentence_a, claims_mode)
            b2 = match_side(r["statement2"], s2, g.sentence_b, claims_mode)
            b1 = match_side(r["statement1"], s1, g.sentence_b, claims_mode)
            a2 = match_side(r["statement2"], s2, g.sentence_a, claims_mode)
            if (a1 and b2) or (b1 and a2):
                hits.append(r)
        hits = pd.DataFrame(hits)
        rec["paired"] = len(hits) > 0
        rec["n_matching_pairs"] = len(hits)
        def any_(col, ok):
            return bool(len(hits) and col in hits and hits[col].map(ok).any())
        rec["sent_to_llm"] = any_("sent_to_llm", lambda v: bool(v) if pd.notna(v) else False)
        rec["stage1_flag"] = any_("s1_decision", lambda v: v in ("Conflict", "Unclear"))
        rec["stage2_conflict"] = any_("s2_label", lambda v: v == "Conflict")
        rec["final"] = any_("final_result", lambda v: v == "Conflict")
        if len(hits) and "nli_contradiction" in hits:
            rec["best_nli"] = float(pd.to_numeric(hits["nli_contradiction"], errors="coerce").max())
        if len(hits) and "drop_reason" in hits:
            rec["drop_reasons"] = sorted(set(map(str, hits["drop_reason"])))
        results.append(rec)
    res = pd.DataFrame(results)

    # Article-level: did the run flag ANY final conflict in each tagged,
    # located article - even via a pair other than the annotated one (e.g.
    # a repeated statement). This is the article-level view the dataset's
    # labels support; "final" above is the stricter "caught the annotated
    # pair".
    flagged_titles = set()
    if "final_result" in pairs.columns:
        conf = pairs[pairs["final_result"] == "Conflict"]
        for g in gold.drop_duplicates("title").itertuples():
            if any(_same_article(t, u, g) for t, u in zip(conf["statement1Title"], conf["statement1URL"])):
                flagged_titles.add(g.title)
    res["article_flagged_any"] = res["title"].isin(flagged_titles)

    def funnel(df: pd.DataFrame, by_article: bool) -> Dict[str, float]:
        if df.empty:
            return {}
        out = {}
        stages = [s for s in STAGES if not (s == "extracted" and not claims_mode)]
        for s in stages:
            hit = df.groupby("title")[s].any() if by_article else df[s]
            out[s] = round(float(hit.mean()), 3)
        out["n"] = int(df["title"].nunique() if by_article else len(df))
        return out

    summary = {
        "run_dir": os.path.abspath(run_dir), "mode": "claims" if claims_mode else "chunks",
        "min_confidence": min_confidence,
        "gold_articles_total": int(gold_all["title"].nunique()),
        "gold_articles_unreachable": gold_all[gold_all["status"] != "located"]["status"].value_counts().to_dict(),
        "pairs": funnel(res, False), "articles": funnel(res, True),
        "articles_flagged_any_pair": round(float(res.groupby("title")["article_flagged_any"].any().mean()), 3)
        if len(res) else None,
        "pairs_by_confidence": {c: funnel(res[res.confidence == c], False)
                                for c in ("high", "medium", "low") if (res.confidence == c).any()},
    }
    return {"summary": summary, "per_pair": res}


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Stage-by-stage recall against known contradictions.")
    p.add_argument("run_dir")
    p.add_argument("--gold", default=GOLD_PATH)
    p.add_argument("--min-confidence", choices=["high", "medium", "low"], default="low")
    p.add_argument("--out", default=None, help="JSON summary path (default: RUN_DIR/recall_report.json)")
    a = p.parse_args(argv)
    r = score(a.run_dir, a.gold, a.min_confidence)
    s, per = r["summary"], r["per_pair"]
    print(f"\nRecall vs {s['gold_articles_total']} tagged articles ({s['mode']} mode, "
          f"gold confidence >= {s['min_confidence']}); unreachable/unlocated: {s['gold_articles_unreachable']}\n")
    print(f"{'stage':<16}{'gold pairs':>12}{'articles':>10}")
    for st in [k for k in STAGES if k in s["pairs"]]:
        print(f"{st:<16}{s['pairs'][st]:>12.0%}{s['articles'][st]:>10.0%}")
    print(f"{'n':<16}{s['pairs']['n']:>12}{s['articles']['n']:>10}")
    print(f"\narticles with ANY final conflict (not necessarily the annotated pair): "
          f"{s['articles_flagged_any_pair']:.0%}")
    cols = [c for c in ["title", "confidence", *[k for k in STAGES if k in s["pairs"]], "best_nli", "drop_reasons"] if c in per]
    print("\n" + per[cols].to_string(index=False))
    out = a.out or os.path.join(a.run_dir, "recall_report.json")
    with open(out, "w") as f:
        json.dump({"summary": s, "per_pair": per.to_dict(orient="records")}, f, indent=2, default=str)
    per.to_csv(os.path.splitext(out)[0] + "_pairs.csv", index=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
