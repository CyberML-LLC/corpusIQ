# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""Exact annotation choices behind wikicontradiction_gold_pairs.csv (see
README.md). Sentence indices refer to the cleaned text split into sentences
the way this file's helper did at annotation time; the CSV holds the verbatim
sentences, so it does not need to be regenerated."""
import pandas as pd
# (row, a_idx, b_idx, confidence, note). Annotated from the tag's about=/reason
# text and its position in the article. a/b index sentences of the cleaned text.
LOCATED = [
 (540, 0, 5, "high", "battle 'fought on 31 May 1918' vs counter-offensive starting 18 July 1918"),
 (540, 0, 3, "medium", "single date 31 May vs 31 May to 22 July"),
 (4451, 1, 5, "high", "history dating back to 1766 vs first created in 1909"),
 (3059, 3, 26, "high", "game dates back to the 7th century vs around 6000 BCE"),
 (3059, 3, 41, "medium", "7th century vs pieces dated 6th century B.C."),
 (4579, 0, 2, "medium", "Indian film actress vs categories Russian models/actresses (category text)"),
 (2976, 1, 7, "low", "killed at least fourteen vs remains of at least 11 additional people"),
 (5176, 185, 197, "high", "only four legal wives vs 'He married eighth'"),
 (5020, 54, 60, "high", "goods engine burst through buffer stops in 1901 vs 1904"),
 (2141, 6, 10, "high", "IPO July 2018 vs IPO February 2004"),
 (2141, 9, 10, "high", "first product released 1996 vs first released March 2004"),
 (1261, 24, 27, "high", "no evidence absorbed through skin vs could be absorbed through skin"),
 (1261, 25, 27, "medium", "not absorbed through skin (mouse study) vs absorbed through skin"),
 (1687, 97, 98, "medium", "charcoal is the treatment vs benefit unproven"),
 (3320, 34, 36, "high", "aberration grows with second power vs fourth power of pupil size"),
 (5274, 59, 114, "high", "narrow gauge allows smaller-radius curves vs no meaningful relationship"),
 (5274, 47, 114, "medium", "narrow gauge enabled tighter turns vs no meaningful relationship"),
 (3258, 3, 12, "medium", "low extreme A2 vs range C3-C5"),
 (4311, 3, 11, "high", "several agencies at the war's outbreak vs no agencies at the outbreak"),
 (5010, 4, 7, "medium", "'sweet water' etymology vs name borrowed from the Sawokli"),
 (5061, 0, 23, "medium", "aesthetic movement vs derogatory term for social conservatives"),
 (5257, 22, 54, "high", "'Mr Tembo' recorded for a baby elephant vs written for Albarn's daughter"),
 (2615, 1, 8, "high", "sea glass and beach glass differ vs terms used interchangeably"),
 (2615, 5, 8, "medium", "beach glass comes from fresh water vs terms interchangeable"),
 (4598, 1, 9, "medium", "rank equated to six-star vs set at five-star level"),
 (4598, 1, 8, "low", "six-star vs precedence still four-star"),
 (870, 114, 122, "low", "revealed to be gay vs Lynch claimed he was not gay (attributed dispute)"),
 (3465, 13, 14, "low", "preteen is 12 and under vs restricted to those close to 12"),
 (4176, 47, 50, "low", "name applied in 1533 vs question whether Alarcon 1540 is the first use"),
 (4866, 11, 12, "low", "India conviction rate around 46% vs 46.8% (tag: first vs second paragraph)"),
]
UNREACHABLE = [  # (row, status, note)
 (601, "not_in_text", "infobox population vs text; infobox is stripped by cleaning"),
 (2104, "not_in_text", "second death date not in cleaned text (likely infobox)"),
 (1597, "not_in_text", "conflict with an unsourced graphic, not in text"),
 (5268, "not_in_text", "photo date in a footnote, stripped by cleaning"),
 (1293, "not_in_text", "values in a table rendered without row structure"),
 (419, "single_sentence", "table row: 'Spain has the most elevators' next to China 4,000,000"),
 (1315, "not_located", "cause of death; second account not found"),
 (5204, "not_located", "start-of-game sequence; only one description found"),
 (4350, "not_located", "crater sizes in tables"),
 (2248, "not_located", "lead vs subsections; too diffuse for a sentence pair"),
 (2620, "not_located", "section-level tag on electromagnetic thrusters"),
 (5052, "not_located", "TV-adaptation section; ship names not found in cleaned text"),
 (4999, "not_located", "card-hand tables"),
 (1123, "not_located", "titles/styles section, no topic given"),
 (1721, "not_located", "titles/styles section, no topic given"),
 (2309, "not_located", "article-level tag, no topic given"),
 (4452, "not_located", "article-level tag, no topic given; vote figures consistent"),
 (624, "not_located", "article-level tag, no topic given"),
 (2480, "not_located", "article-level tag, no topic given"),
 (3412, "not_located", "tag variant not found in cleaned text, no topic given"),
 (4868, "not_located", "article-level tag, no topic given"),
]
