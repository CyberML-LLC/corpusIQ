# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
import json

import pandas as pd

import score_recall as S


def _gold(tmp_path):
    g = pd.DataFrame([
        dict(csv_index=1, app_row=7, title="Lithgow", revision_id=1, about="", status="located",
             confidence="high", sentence_a="In 1901 a goods engine burst through the buffer stops.",
             sentence_b="In 1904 a goods engine burst through the buffer stops.",
             sent_idx_a=0, sent_idx_b=1, note=""),
        dict(csv_index=2, app_row=9, title="Opera", revision_id=2, about="", status="located",
             confidence="medium", sentence_a="Opera had an IPO in 2004.",
             sentence_b="Opera went public on NASDAQ in 2018.", sent_idx_a=0, sent_idx_b=1, note=""),
        dict(csv_index=3, app_row=11, title="Heligoland", revision_id=3, about="", status="not_in_text",
             confidence="", sentence_a="", sentence_b="", sent_idx_a=None, sent_idx_b=None, note=""),
    ])
    p = tmp_path / "gold.csv"
    g.to_csv(p, index=False)
    return str(p)


def test_chunk_mode_funnel(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    t = pd.DataFrame([
        # Lithgow gold pair, judged Conflict end to end (reversed orientation on purpose)
        dict(statement1="In 1904 a goods engine burst through the buffer stops at the top.",
             statement2="Earlier text. In 1901 a goods engine burst through the buffer stops.",
             statement1Title="f.csv:Lithgow", statement2Title="f.csv:Lithgow",
             statement1URL="local:f.csv#row7#chunk1", statement2URL="local:f.csv#row7#chunk0",
             sent_to_llm=True, s1_decision="Conflict", s2_label="Conflict", final_result="Conflict"),
        # same sentences but from ANOTHER revision of the article: must not count
        dict(statement1="Opera had an IPO in 2004.", statement2="Opera went public on NASDAQ in 2018.",
             statement1Title="f.csv:Opera", statement2Title="f.csv:Opera",
             statement1URL="local:f.csv#row10#chunk0", statement2URL="local:f.csv#row10#chunk1",
             sent_to_llm=True, s1_decision="Conflict", s2_label="Conflict", final_result="Conflict"),
        # right revision, but dropped by NLI
        dict(statement1="Opera had an IPO in 2004.", statement2="Opera went public on NASDAQ in 2018.",
             statement1Title="f.csv:Opera", statement2Title="f.csv:Opera",
             statement1URL="local:f.csv#row9#chunk0", statement2URL="local:f.csv#row9#chunk1",
             sent_to_llm=False, s1_decision=None, s2_label=None, final_result=None),
    ])
    t.to_parquet(run / "pair_trace.parquet")
    r = S.score(str(run), _gold(tmp_path))
    per = r["per_pair"].set_index("title")
    assert per.loc["Lithgow", "final"] and per.loc["Opera", "paired"]
    assert not per.loc["Opera", "sent_to_llm"]
    s = r["summary"]
    assert s["mode"] == "chunks" and "extracted" not in s["pairs"]
    assert s["pairs"]["paired"] == 1.0 and s["pairs"]["sent_to_llm"] == 0.5 and s["pairs"]["final"] == 0.5
    assert s["gold_articles_unreachable"] == {"not_in_text": 1}
    assert S.score(str(run), _gold(tmp_path), "high")["summary"]["pairs"]["n"] == 1


def test_claims_mode_requires_claim_from_that_sentence(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    src = "In 1901 a goods engine burst through the buffer stops. In 1904 a goods engine burst through the buffer stops."
    pd.DataFrame([dict(statement1="A goods engine burst through the buffer stops in 1901.",
                       statement2="The zig zag line opened in 1869.",   # neighbour claim, wrong sentence
                       statement1Title="f.csv:Lithgow", statement2Title="f.csv:Lithgow",
                       statement1URL="local:f.csv#row7#chunk0#s1", statement2URL="local:f.csv#row7#chunk0#s2",
                       statement1Source=src, statement2Source=src + " The zig zag line opened in 1869.",
                       sent_to_llm=True, s1_decision="Conflict", s2_label="Conflict", final_result="Conflict")]
                 ).to_parquet(run / "pair_trace.parquet")
    pd.DataFrame([dict(title="f.csv:Lithgow", url="local:f.csv#row7#chunk0#s1", source_text=src,
                       statement="A goods engine burst through the buffer stops in 1901."),
                  dict(title="f.csv:Lithgow", url="local:f.csv#row7#chunk0#s3", source_text=src,
                       statement="A goods engine burst through the buffer stops in 1904.")]
                 ).to_parquet(run / "claims.parquet")
    r = S.score(str(run), _gold(tmp_path), "high")
    row = r["per_pair"].iloc[0]
    assert row["extracted"] and not row["paired"]   # both claims exist, but this pair isn't them


def test_shipped_gold_file_is_consistent():
    g = pd.read_csv(S.GOLD_PATH)
    assert g["title"].nunique() == 44
    loc = g[g.status == "located"]
    assert loc["sentence_a"].str.len().gt(10).all() and loc["sentence_b"].str.len().gt(10).all()
    assert set(loc.confidence) <= {"high", "medium", "low"} and g.app_row.notna().all()


def test_article_flagged_any_credits_other_pairs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame([dict(statement1="Some other sentence about Lithgow.", statement2="Yet another one.",
                       statement1Title="f.csv:Lithgow", statement2Title="f.csv:Lithgow",
                       statement1URL="local:f.csv#row7#chunk3", statement2URL="local:f.csv#row7#chunk4",
                       sent_to_llm=True, s1_decision="Conflict", s2_label="Conflict", final_result="Conflict")]
                 ).to_parquet(run / "pair_trace.parquet")
    r = S.score(str(run), _gold(tmp_path), "high")
    assert not r["per_pair"].iloc[0]["final"]
    assert r["summary"]["articles_flagged_any_pair"] == 1.0
