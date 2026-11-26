# Recall set: known contradictions in WikiContradiction

`wikicontradiction_gold_pairs.csv` lists, for each of the **44 revisions in
`solvedSelfC.csv` that still carry a `{{Self-contradictory}}` tag**, the pair
of sentences that contradict each other, so a run's recall can be measured
stage by stage with `../score_recall.py`.

## How it was built

- **Source of truth:** the tag itself. Wikipedia editors place
  `{{Self-contradictory}}` on an article (or section) and usually say what the
  contradiction is about (`about=the date of the battle`, or a free-text
  reason). The tag's position marks the section.
- **Annotation:** for each tagged revision the article was cleaned exactly as
  the app ingests it (`data_loader.clean_wiki_text`), split into sentences,
  and the two sentences that make the contradiction described by the tag were
  chosen. Sentences are copied verbatim (some carry leftover caption/heading
  text from cleaning); `sent_idx_a/b` record their positions.
  `annotation_record.py` is the exact list of choices with a one-line reason
  for each.
- **Annotator:** Claude (an AI assistant), not a human. Every pair has a
  `confidence`:
  - `high` (13 pairs, 12 articles): the two sentences directly assert
    incompatible facts on the tag's stated topic.
  - `medium` (11 pairs): incompatible on the stated topic, but one side is
    hedged, a range, or category text.
  - `low` (6 pairs): plausible reading of a vague or missing topic.
  Spot-check before relying on medium/low pairs; `--min-confidence high` uses
  only the clearest ones.
- **Unusable articles (21) are kept, with a reason (`status`):**
  - `not_in_text` (5): the contradiction involves an infobox, footnote,
    graphic or table that cleaning removes, so no text pipeline can see it.
  - `single_sentence` (1): both sides are in one table row; a pair-based
    method cannot catch it by design.
  - `not_located` (15): no topic given, or the second statement could not be
    found with confidence.

## Columns

`csv_index` (the CSV's unnamed index column), `app_row` (the app's
`#row<N>` document number: position among non-empty CSV rows), `title`,
`revision_id`, `about` (tag text), `status`, `confidence`, `sentence_a`,
`sentence_b`, `sent_idx_a`, `sent_idx_b`, `note`.

## Caveats

- These are the contradictions editors **tagged**; an article may contain
  others. This set measures recall, not precision.
- 23 of 44 articles have a located pair, so recall is measured on a small
  set: one article is ~4 percentage points.
- The other 194 rows of the dataset are revisions where the tag had been
  removed; they are not evidence of "no contradiction" at sentence level.
