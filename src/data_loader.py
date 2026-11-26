# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""
data_loader.py
==============

Loads and prepares the public **WikiContradiction** dataset for the conflict-
detection pipeline.

Dataset
-------
WikiContradiction (Hsu et al., IEEE BigData 2021) is a collection of English
Wikipedia articles that editors flagged with the ``{{self-contradictory}}``
maintenance template. Each row is one article revision; the contradiction lives
*inside* the article, between two (or more) of its own sentences.

    Source : https://github.com/Wiki-Contradictory/Wiki-Self-Contradictory
    File   : solvedSelfC.csv
    Columns: revision_parent_id, revision_id, page_title, revision_text
    License: data derived from Wikipedia (CC BY-SA).

This makes it a strong, fully public showcase for the pipeline: the raw articles
contain real, human-identified self-contradictions, and the pipeline's job is to
surface the specific contradicting sentence pairs.

What this module does
---------------------
1. Fetches the CSV from GitHub (cached locally on first download).
2. Parses MediaWiki markup into clean prose (``mwparserfromhell`` if available,
   regex fallback otherwise).
3. Splits each article into sentences.
4. Emits candidate sentence pairs for the cascade to judge. By default it pairs
   sentences *within the same article* (where the known contradictions live).

Public API (kept stable for the rest of the pipeline)
----------------------------------------------------
- ``load_wiki_contradiction(...) -> pd.DataFrame``
- ``get_statement_pairs(...) -> list[tuple[str, str]]``   (drop-in replacement)
- ``get_article_sentences(...) -> dict[str, list[str]]``
- ``get_stage_examples(numbers, stage) -> str``           (few-shot examples)
"""

from __future__ import annotations

import os
import re
import urllib.request
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import mwparserfromhell  # type: ignore
    _HAS_MWP = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_MWP = False


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/Wiki-Contradictory/"
    "Wiki-Self-Contradictory/main/solvedSelfC.csv"
)

# Local cache lives next to this file, in a sibling ``data/`` directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(_THIS_DIR, "..", "data")
DEFAULT_CACHE_PATH = os.path.join(DEFAULT_CACHE_DIR, "wiki_self_contradictory.csv")

TEXT_COLUMN = "revision_text"
TITLE_COLUMN = "page_title"


# ----------------------------------------------------------------------------
# Download / cache
# ----------------------------------------------------------------------------

def download_dataset(
    url: str = DATA_URL,
    cache_path: str = DEFAULT_CACHE_PATH,
    force: bool = False,
) -> str:
    """
    Download the WikiContradiction CSV to ``cache_path`` (once).

    Returns the local path. Set ``force=True`` to re-download.
    """
    cache_path = os.path.abspath(cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if force or not os.path.exists(cache_path):
        print(f"[data_loader] Downloading dataset from {url}")
        urllib.request.urlretrieve(url, cache_path)
        print(f"[data_loader] Saved to {cache_path}")
    return cache_path


def load_wiki_contradiction(
    source: Optional[str] = None,
    cache_path: str = DEFAULT_CACHE_PATH,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load the raw dataset as a DataFrame.

    Parameters
    ----------
    source : str, optional
        Local path or URL to a CSV. If None, the canonical dataset is fetched
        (and cached). A local path that exists is read directly.
    cache_path : str
        Where to cache the canonical download.
    force_download : bool
        Re-download even if a cache exists.

    Returns
    -------
    pd.DataFrame with at least ``page_title`` and ``revision_text`` columns.
    """
    if source is not None and os.path.exists(source):
        path = source
    elif source is not None and source.startswith("http"):
        path = download_dataset(url=source, cache_path=cache_path, force=force_download)
    else:
        path = download_dataset(cache_path=cache_path, force=force_download)

    df = pd.read_csv(path)

    # Drop the unnamed pandas index column if present.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # Keep only rows that actually have article text.
    df = df.dropna(subset=[TEXT_COLUMN]).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Wiki-markup cleaning
# ----------------------------------------------------------------------------

# Block-level markup we want to drop entirely before extracting prose.
_REF_RE = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_FILE_LINK_RE = re.compile(r"\[\[(?:File|Image):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]",
                           re.IGNORECASE)
_HEADING_RE = re.compile(r"^=+\s*.*?\s*=+\s*$", re.MULTILINE)
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{2,}")

# --- Value-bearing template expansion -------------------------------------
# ``mwparserfromhell.strip_code()`` deletes template output entirely, which
# silently destroys the very facts a contradiction hinges on:
#
#   {{Track gauge|1435mm}}             -> ""   (loses "1435mm")
#   {{convert|1600|mm|ftin|abbr=on}}   -> ""   (loses "1600 mm")
#   {{start date|1874|3|2}}            -> ""   (loses the date)
#
# WikiContradiction is overwhelmingly about numbers, dates and units, so
# cleaning a revision with those templates removed leaves sentences that
# cannot contradict each other (e.g. "the gauge has been standardized at .").
# Render the positional arguments of these templates as plain text *before*
# stripping markup, so the figures survive into the cleaned prose.
_VALUE_TEMPLATES = {
    "convert", "cvt", "convert abbreviated", "convert to",
    "track gauge", "trackgauge", "rail gauge", "railtrackgauge",
    "val", "formatnum", "frac", "sub", "sup", "nowrap", "plainlist",
    "height", "weight", "age in years and days",
}
_DATE_TEMPLATES = {
    "start date", "end date", "start date and age", "end date and age",
    "birth date", "birth date and age", "death date", "death date and age",
    "film date", "dts", "birth year and age", "death year and age",
}

# Regex fallback (used when mwparserfromhell is unavailable): same allowlist,
# single-level, non-nested templates only.
_VALUE_TEMPLATE_RE = re.compile(
    r"\{\{\s*(?:" + "|".join(
        re.escape(n) for n in sorted(_VALUE_TEMPLATES | _DATE_TEMPLATES,
                                     key=len, reverse=True)
    ) + r")\s*\|([^{}]*)\}\}",
    re.IGNORECASE,
)


def _join_template_args(raw_args: str, is_date: bool) -> str:
    """Turn a template's raw ``|``-separated args into plain text."""
    vals = []
    for part in raw_args.split("|"):
        part = part.strip()
        if not part:
            continue
        # Drop named params (``abbr=on``, ``disp=table``, ``lk=in`` ...).
        if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_ -]*=", part):
            continue
        vals.append(part)
    if not vals:
        return ""
    return ("-" if is_date else " ").join(vals)


def _expand_value_templates(text: str) -> str:
    """
    Render numeric/date templates (``convert``, ``Track gauge``, ``start date``,
    ``val`` ...) into plain text so their figures survive markup stripping.

    Named parameters are discarded; positional ones are joined with a space
    (or a hyphen for date templates). Unknown templates are left untouched and
    handled by the normal stripping path.
    """
    if not _HAS_MWP:
        def _sub(m: "re.Match") -> str:
            name = m.group(0).split("|", 1)[0].strip("{ ").strip().lower()
            rendered = _join_template_args(m.group(1), name in _DATE_TEMPLATES)
            return f" {rendered} " if rendered else " "

        return _VALUE_TEMPLATE_RE.sub(_sub, text)

    try:
        code = mwparserfromhell.parse(text)
    except Exception:
        return text

    for tpl in list(code.filter_templates(recursive=True)):
        name = str(tpl.name).strip().lower()
        is_date = name in _DATE_TEMPLATES
        if not is_date and name not in _VALUE_TEMPLATES:
            continue
        vals = []
        for param in tpl.params:
            pname = str(param.name).strip()
            if pname and not pname.isdigit():
                continue  # skip named params like abbr=on
            value = str(param.value).strip()
            if value:
                vals.append(value)
        if not vals:
            continue
        rendered = ("-" if is_date else " ").join(vals)
        try:
            code.replace(tpl, f" {rendered} ")
        except Exception:
            continue
    return str(code)


def _regex_strip_markup(text: str) -> str:
    """Best-effort wiki-markup removal without external dependencies."""
    text = _COMMENT_RE.sub(" ", text)
    text = _REF_RE.sub(" ", text)
    text = _TABLE_RE.sub(" ", text)
    text = _FILE_LINK_RE.sub(" ", text)
    # Keep figures from value/date templates before templates get deleted.
    text = _expand_value_templates(text)

    # Remove {{templates}} (handle a couple of nesting levels).
    for _ in range(5):
        new = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        if new == text:
            break
        text = new

    # [[link|label]] -> label ; [[link]] -> link
    text = re.sub(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]+)\]\]", r"\1", text)
    # [http://url label] -> label ; bare [http://url] -> ""
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\]", " ", text)

    text = _HEADING_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)

    # Bold/italic apostrophes, leftover braces/brackets.
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"[\{\}\[\]]", " ", text)
    return text


def clean_wiki_text(text: str) -> str:
    """
    Convert raw MediaWiki ``revision_text`` into clean prose.

    Uses ``mwparserfromhell`` when installed (more accurate), otherwise falls
    back to the regex stripper above.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Expand figure-bearing templates first so strip_code() cannot delete the
    # numbers/dates that contradictions are made of.
    text = _expand_value_templates(text)

    if _HAS_MWP:
        text = _COMMENT_RE.sub(" ", text)
        text = _REF_RE.sub(" ", text)
        try:
            stripped = mwparserfromhell.parse(text).strip_code(normalize=True,
                                                               collapse=True)
        except Exception:
            stripped = _regex_strip_markup(text)
        text = stripped
    else:
        text = _regex_strip_markup(text)

    # Remove leftover image/caption scaffolding such as "thumb|", "left|250px|",
    # "upright=1.5|", "alt=...|", including options glued to punctuation or to
    # each other ("3.|left|thumb|120px|The emblem"), then a stray pipe left
    # between a sentence end and the caption text.
    text = _IMAGE_OPTION_RE.sub(" ", text)
    text = re.sub(r"(?<=[.!?\"'])\s*\|\s*(?=[A-Z])", " ", text)

    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n", text)
    return text.strip()


# ----------------------------------------------------------------------------
# Sentence splitting
# ----------------------------------------------------------------------------

_IMAGE_OPTION_RE = re.compile(
    r"(?:^|(?<=[\s|.,;:]))"
    r"(?:thumb|thumbnail|left|right|center|centre|frame|frameless|framed|border|none|"
    r"upright(?:\s*=\s*[\d.]+)?|\d+(?:x\d+)?px|(?:alt|link|class|lang|page)\s*=[^|\n]*)"
    r"\s*\|",
    re.IGNORECASE,
)

_ABBREV = {
    # titles and ranks
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon", "pres",
    "gov", "sen", "rep", "capt", "col", "gen", "lt", "maj", "sgt", "cpl", "adm",
    # months
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    # places and organisations
    "mt", "ft", "ave", "blvd", "cor", "inc", "ltd", "co", "corp", "bros", "dept",
    # references and approximations
    "vs", "etc", "e.g", "i.e", "no", "nos", "fig", "al", "vol", "pp", "op", "cf",
    "ca", "c", "approx", "est", "u.s", "u.k",
}
# A personal initial or run of initials ("T.", "J.R.R.") ends a name, not a sentence.
# Single Roman numerals ("World War I.", "Henry V.") are excluded: they end sentences.
_INITIALS_RE = re.compile(r"^\(?(?!(?:I|V|X)\.$)(?:[A-Z]\.)+$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# Table rows and markup that survive cleaning; not prose.
_NON_PROSE_RE = re.compile(
    r"\||!!|^\s*!|\b(?:alt|link|rowspan|colspan|width|style|class)\s*=", re.IGNORECASE)


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter (no heavy NLP dependency).

    Splits on sentence-final punctuation followed by whitespace and a capital,
    except after a known abbreviation ("Capt.", "Feb.", "ca.") or a personal
    initial ("James T. Sutherland", "J.R.R. Tolkien"). A sentence that really
    ends in a capital initial ("vitamin C.") is merged with the next one; that
    is rarer than names and costs less than a split name.
    """
    if not text:
        return []

    raw = _SENT_SPLIT_RE.split(text.replace("\n", " "))
    sentences: List[str] = []
    buffer = ""

    for piece in raw:
        candidate = (buffer + " " + piece).strip() if buffer else piece.strip()
        last_token = candidate.split(" ")[-1]
        last_word = re.sub(r"[^A-Za-z.]", "", last_token).lower().rstrip(".")
        if last_word in _ABBREV or _INITIALS_RE.match(last_token):
            buffer = candidate  # keep accumulating; likely a false split
        else:
            sentences.append(candidate)
            buffer = ""
    if buffer:
        sentences.append(buffer)

    return [s.strip() for s in sentences if s.strip()]


def _is_usable_sentence(s: str, min_words: int, max_words: int) -> bool:
    words = s.split()
    if not (min_words <= len(words) <= max_words):
        return False
    # Skip table rows and markup leftovers.
    if _NON_PROSE_RE.search(s):
        return False
    # Skip lines that are mostly markup leftovers or list scaffolding.
    alpha = sum(c.isalpha() for c in s)
    if alpha < 0.5 * max(len(s), 1):
        return False
    return True


# ----------------------------------------------------------------------------
# Article -> sentences -> candidate pairs
# ----------------------------------------------------------------------------

def get_article_sentences(
    source: Optional[str] = None,
    min_words: int = 6,
    max_words: int = 80,
    max_sentences_per_article: Optional[int] = None,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> Dict[str, List[str]]:
    """
    Return ``{article_title: [clean_sentence, ...]}`` for every article.

    Sentences are cleaned of wiki markup and filtered to a sensible length band.
    ``max_sentences_per_article`` truncates each article; it is off by default
    because contradictions can sit anywhere in an article (a cap of 60 dropped
    4 of 29 gold pairs on confdoc004).
    """
    df = load_wiki_contradiction(source=source, cache_path=cache_path)

    out: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        title = str(row.get(TITLE_COLUMN, "")).strip() or f"article_{len(out)}"
        cleaned = clean_wiki_text(row[TEXT_COLUMN])
        sentences = [
            s for s in split_sentences(cleaned)
            if _is_usable_sentence(s, min_words, max_words)
        ]
        if max_sentences_per_article:
            sentences = sentences[:max_sentences_per_article]
        if len(sentences) >= 2:
            out[title] = sentences
    return out


def get_statement_pairs(
    source: Optional[str] = None,
    max_articles: Optional[int] = None,
    max_pairs_per_article: Optional[int] = 50,
    min_words: int = 6,
    max_words: int = 80,
    return_titles: bool = False,
    cache_path: str = DEFAULT_CACHE_PATH,
) -> List[Tuple[str, str]]:
    """
    Drop-in replacement for the original ``get_statement_pairs``.

    Builds candidate (statement_a, statement_b) pairs *within each article*
    (i.e. the same-document setting where WikiContradiction's contradictions
    live), so the cascade can flag which pairs actually contradict.

    Parameters
    ----------
    source : str, optional
        CSV path/URL. None -> canonical dataset (cached).
    max_articles : int, optional
        Limit number of articles (useful for quick demos).
    max_pairs_per_article : int, optional
        Cap intra-article pairs to avoid combinatorial blow-up.
    return_titles : bool
        If True, returns 3-tuples ``(text_a, text_b, title)`` instead of pairs.

    Returns
    -------
    list of (text_a, text_b) tuples — or (text_a, text_b, title) if requested.
    """
    article_sentences = get_article_sentences(
        source=source, min_words=min_words, max_words=max_words,
        cache_path=cache_path,
    )

    titles = list(article_sentences.keys())
    if max_articles is not None:
        titles = titles[:max_articles]

    pairs: List[Tuple[str, str]] = []
    for title in titles:
        sents = article_sentences[title]
        article_pairs = list(combinations(sents, 2))
        if max_pairs_per_article is not None:
            article_pairs = article_pairs[:max_pairs_per_article]
        for a, b in article_pairs:
            pairs.append((a, b, title) if return_titles else (a, b))

    return pairs


# ----------------------------------------------------------------------------
# Few-shot examples for the cascade prompts
# ----------------------------------------------------------------------------

# A small, self-contained set of illustrative examples so the pipeline can run
# out-of-the-box with no proprietary example file. Stage 2 wants a label;
# stage 3 wants a judge decision. Edit/extend freely.
_BUILTIN_EXAMPLES = {
    "s2": [
        {
            "text_a": "The bridge was completed in 1932 and opened to traffic that same year.",
            "text_b": "Construction of the bridge finished in 1932, allowing vehicles to cross by year's end.",
            "rationale": "Both statements describe the same completion year and opening; "
                         "they are paraphrases of one another.",
            "label": "No Conflict",
        },
        {
            "text_a": "The species is found only in the highlands of Peru.",
            "text_b": "The species is native to lowland rainforests across Brazil.",
            "rationale": "Same entity (the species), but mutually exclusive ranges "
                         "(Peruvian highlands vs Brazilian lowlands).",
            "label": "Conflict",
        },
        {
            "text_a": "The festival is held annually in June.",
            "text_b": "Vendors at the festival sell handmade crafts and local food.",
            "rationale": "The two sentences discuss different aspects (timing vs vendors) "
                         "and make no competing claim.",
            "label": "No Conflict",
        },
    ],
    "s3": [
        {
            "text_a": "The population was recorded as 12,000 in the 2010 census.",
            "text_b": "According to the 2010 census, the town had 8,500 residents.",
            "stage1_output": "Alignment Same for entity/time; spans give 12,000 vs 8,500 "
                             "for the same census year.",
            "stage2_label": "Conflict",
            "decision": "Decision: Accept",
        },
        {
            "text_a": "The album was released in March.",
            "text_b": "The record came out in the spring.",
            "stage1_output": "Spans are compatible (March is in spring); no contradiction.",
            "stage2_label": "Conflict",
            "decision": "Decision: Reject (No Conflict)",
        },
    ],
}


def get_stage_examples(numbers: List[str], stage: int) -> str:
    """
    Return formatted few-shot examples for a given stage.

    Mirrors the original signature so ``cascade.py`` keeps working unchanged.
    ``numbers`` selects which built-in examples to include (1-indexed); an empty
    list means "use all available for this stage".
    """
    stage_name = f"s{stage}"
    pool = _BUILTIN_EXAMPLES.get(stage_name, [])
    if not pool:
        return ""

    if numbers:
        idxs = []
        for n in numbers:
            try:
                i = int(n) - 1
                if 0 <= i < len(pool):
                    idxs.append(i)
            except (TypeError, ValueError):
                continue
        selected = [pool[i] for i in idxs] or pool
    else:
        selected = pool

    blocks: List[str] = []
    for ex in selected:
        if stage_name == "s2":
            blocks.append(
                f"Statement 1: {ex['text_a']}\n"
                f"Statement 2: {ex['text_b']}\n"
                f"stage1_output: {ex['rationale']}\n"
                f"label: {ex['label']}"
            )
        elif stage_name == "s3":
            blocks.append(
                f"Statement 1: {ex['text_a']}\n"
                f"Statement 2: {ex['text_b']}\n"
                f"stage1_output: {ex['stage1_output']}\n"
                f"Stage-2 Label: {ex['stage2_label']}\n"
                f"{ex['decision']}"
            )
        else:  # stage 1 has no few-shot block by default
            continue

    return "\n---\n".join(blocks)


# ----------------------------------------------------------------------------
# Manual smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_wiki_contradiction()
    print(f"Loaded {len(df)} article rows "
          f"({df[TITLE_COLUMN].nunique()} unique titles).")

    sentences = get_article_sentences(max_sentences_per_article=20)
    print(f"Parsed sentences for {len(sentences)} articles.")

    sample_title = next(iter(sentences))
    print(f"\nExample article: {sample_title!r}")
    for s in sentences[sample_title][:3]:
        print("  -", s[:120])

    pairs = get_statement_pairs(max_articles=3, max_pairs_per_article=5)
    print(f"\nGenerated {len(pairs)} candidate pairs from 3 articles.")
    if pairs:
        print("First pair:")
        print("  A:", pairs[0][0][:120])
        print("  B:", pairs[0][1][:120])

    print("\nStage-2 few-shot examples:\n")
    print(get_stage_examples([], stage=2)[:400])
