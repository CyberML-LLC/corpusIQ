# corpusIQ web UI and API server

corpusIQ is a product of [CyberML](https://cyberml.ai).

The web UI in `web/` is a static site, published to GitHub Pages by `.github/workflows/pages.yml`. All processing
happens in the API server in this folder, which you run yourself. Your DeepSeek key, cloud credentials and documents
never go to GitHub.

## Pipeline

1. **Ingest** files (CSV, PDF, DOCX, TXT, MD, HTML, JSON), links, Amazon S3, Azure Blob Storage or Google Cloud Storage.
   Each CSV row becomes its own document.
2. **Split** each document into sentences (6 to 80 words) with the project's cleaner and splitter.
3. **Retrieve** the 20 nearest sentences for each sentence with `all-mpnet-base-v2`, within and across documents.
4. **Filter** with the v4 student (fine-tuned `nli-deberta-v3-base`). Pairs with a conflict score of at least 0.003,
   or a duplicate score of at least 0.5, go on to the judges. On the WikiContradiction gold set this kept all 16
   confirmed contradictions while sending 28% of candidate pairs.
5. **Judge** each pair with DeepSeek V4 Flash and V4 Pro (prompt C):
   - **Conflict**: both judges say Conflict.
   - **Unsure**: one judge says Conflict, or either says Unclear.
   - **Duplicate**: either says Duplicate, and neither says Conflict.
   - **No conflict**: everything else.

   On the gold set this surfaces 11 of 16 confirmed contradictions as Conflict or Unsure. The best single judge finds 9.

These figures come from CyberML's internal evaluation on the WikiContradiction gold set.

## Run it

From the repository root:

```bash
cIQenv/bin/pip install -r requirements.txt                 # plus boto3 / azure-storage-blob / google-cloud-storage if needed
git lfs pull                                               # fetches the v4 model (738 MB, stored with Git LFS)
cIQenv/bin/python -m uvicorn --app-dir server app:app --port 8000
```

Then open <http://localhost:8000>; the server also serves the UI. Alternatively, open the GitHub Pages site and set
the server address with the status button in the top right. Chrome asks permission before a public site may reach
your computer: allow **Local network access** for the site (icon left of the address bar, then Site settings), then
reload. Safari blocks this outright, so in Safari use <http://localhost:8000> directly.

Settings:

| Variable | Default | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` (in `.env`) | none | LLM judges |
| `CORPUSIQ_V4_MODEL` | `models/v4_filter` | v4 filter model |
| `CORPUSIQ_JOBS_DIR` | `server/jobs` | saved results (git-ignored) |
| `CORPUSIQ_CORS_REGEX` | `*.github.io` and localhost | which sites may call the server |
| `CORPUSIQ_MAX_OBJECTS` | 500 | most files read from one bucket or link list |
| `AZURE_STORAGE_CONNECTION_STRING` | none | Azure default when the form leaves it empty |

Cloud credentials typed into the form go only to your API server. Leave them empty to use each SDK's default
credentials: `~/.aws` or an instance role for S3, the connection string above for Azure, and
`gcloud auth application-default login` for Google Cloud.

The server binds to `127.0.0.1` by default. If you expose it on a network, put it behind authentication: anyone who
can reach it can run jobs on your DeepSeek account.

## Cost

Each judged pair costs one call per judge, about 1,000 input tokens. A corpus the size of the WikiContradiction gold
articles (23 articles, 26,000 candidate pairs) sends about 7,400 pairs, so about 14,800 calls with both judges. Lower
**Most pairs sent to the judges** in Settings to cap this.
