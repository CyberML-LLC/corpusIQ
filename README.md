# corpusIQ

**A product of [CyberML](https://cyberml.ai).**

Finds contradictions between sentences, within one document or across many. You give it a set of documents. It
returns the sentence pairs that conflict, the pairs it is unsure about, and the pairs that repeat each other, with
each judge's reasoning, in a web dashboard.

Web UI: <https://cyberml-llc.github.io/corpusIQ/> (it needs the API server below running on your computer).

## How it works

```
documents ─► sentences ─► retrieval ─► v4 filter ─► two LLM judges ─► Conflict / Unsure / Duplicate / No conflict
```

1. **Ingest** files (CSV, PDF, DOCX, TXT, MD, HTML, JSON), links, Amazon S3, Azure Blob Storage or Google Cloud
   Storage. Each CSV row becomes its own document.
2. **Split** each document into sentences of 6 to 80 words.
3. **Retrieve** the 20 nearest sentences for each sentence with `all-mpnet-base-v2`, within and across documents.
4. **Filter** with the v4 student: `nli-deberta-v3-base` fine-tuned for this task. A pair goes on if its conflict
   score is at least 0.003, or its duplicate score at least 0.5. Everything else is reported as Filtered and costs
   nothing.
5. **Judge** each remaining pair with DeepSeek V4 Flash and V4 Pro, one call each, using a single prompt that asks for
   a short reason and then a decision. The two verdicts combine as:
   - **Conflict**: both say Conflict.
   - **Unsure**: one says Conflict, or either says Unclear.
   - **Duplicate**: either says Duplicate, and neither says Conflict.
   - **No conflict**: everything else.

On the WikiContradiction gold set (23 articles, 16 confirmed contradictions), the filter keeps all 16 while sending
28% of the 26,400 candidate pairs to the judges, and the judges surface 11 of the 16 as Conflict or Unsure.
These figures come from CyberML's internal evaluation.

## Run it

```bash
python -m venv cIQenv && cIQenv/bin/pip install -r requirements.txt
cp .env.example .env                # then set DEEPSEEK_API_KEY
git lfs pull                        # the v4 model, 738 MB
cIQenv/bin/python -m uvicorn --app-dir server app:app --port 8000
```

Open <http://localhost:8000>. Settings, cloud credentials, browser notes and cost are in
[server/README.md](server/README.md).

## Repository

| Path | What it is |
|---|---|
| `server/` | API server: pipeline, judges, data sources |
| `web/` | Web UI (static; published to GitHub Pages by `.github/workflows/pages.yml`) |
| `src/` | Shared text cleaning and sentence splitting (`data_loader.py`) and file ingestion (`ingestion/`) |
| `models/v4_filter/` | The v4 filter model (Git LFS) |
| `eval/` | WikiContradiction gold pairs (`recall_set/`) and the scorer used by the experiments |
| `tests/` | Tests for the shared modules (`pytest tests`) |

## Dataset

WikiContradiction: Wikipedia articles that editors tagged `{{self-contradictory}}`. See `eval/recall_set/README.md`
for how the gold pairs were located.

## License

Copyright 2026 CyberML ([cyberml.ai](https://cyberml.ai)).

corpusIQ is licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for attributions.
The WikiContradiction gold pairs in `eval/recall_set/` quote Wikipedia text, which is available under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Contact

Questions, partnerships and commercial support: [cyberml.ai](https://cyberml.ai).
