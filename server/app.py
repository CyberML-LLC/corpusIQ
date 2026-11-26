# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""corpusIQ API server for the web UI (web/, also published on GitHub Pages).

Run from the repository root:
    cIQenv/bin/python -m uvicorn --app-dir server app:app --port 8000
then open http://localhost:8000 (the server also serves the UI), or open the GitHub Pages site and point it at
http://localhost:8000. The DeepSeek key is read from .env (DEEPSEEK_API_KEY) and never sent to the browser.

Endpoints
  GET  /api/health            model, judge and connector readiness
  POST /api/jobs              multipart: source=upload|s3|azure|gcs|url, config=<json>, options=<json>, files=<files>
  GET  /api/jobs/{id}         status and progress
  GET  /api/jobs/{id}/results results JSON (the dashboard's input)
"""
from __future__ import annotations

import importlib.util
import json
import os
import threading
import time
import traceback
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline as P
import sources
from judge import MODES, Judge

ROOT = P.ROOT
JOBS_DIR = os.environ.get('CORPUSIQ_JOBS_DIR', os.path.join(ROOT, 'server', 'jobs'))
WEB_DIR = os.path.join(ROOT, 'web')
os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title='corpusIQ API', version='1.0')
# The GitHub Pages site (https://<user>.github.io) and local pages may call this server.
app.add_middleware(CORSMiddleware, allow_origin_regex=os.environ.get(
    'CORPUSIQ_CORS_REGEX', r'https://[a-z0-9-]+\.github\.io|http://(localhost|127\.0\.0\.1)(:\d+)?'),
    allow_methods=['GET', 'POST'], allow_headers=['*'],
    # Chrome asks before a public https page may call a server on localhost (Private Network Access);
    # without this the pre-check is rejected with "Disallowed CORS private-network".
    allow_private_network=True)


def _dotenv(path: str) -> Dict[str, str]:
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line: continue
            k, v = line.split('=', 1)
            out[k.strip().removeprefix('export ').strip().upper()] = v.strip().strip('"').strip("'")
    return out


def settings():
    """DeepSeek key and base URL: environment variables first, then .env in the repository root."""
    env = _dotenv(os.path.join(ROOT, '.env'))
    get = lambda k, d='': os.environ.get(k) or env.get(k) or d
    return get('DEEPSEEK_API_KEY'), get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')


JOBS: Dict[str, Dict] = {}
RUN_LOCK = threading.Lock()  # one pipeline at a time: the models share one GPU


def _set(job_id: str, **kw):
    JOBS[job_id].update(kw, updated=time.time())


def _work(job_id: str, kind: str, cfg: Dict, files: List, opts: P.Options):
    try:
        _set(job_id, status='queued', stage='Waiting for the previous job')
        with RUN_LOCK:
            _set(job_id, status='running', stage='Reading documents', done=0, total=0)
            docs = sources.load(kind, cfg, files)
            if not docs: raise ValueError('No readable documents found in this source.')
            _set(job_id, stage=f'Read {len(docs)} documents')
            key, base = settings()
            if not key: raise RuntimeError('DEEPSEEK_API_KEY is not set in .env on the API server.')
            judge = Judge(key, base, mode=opts.judge_mode, domain=opts.domain)
            res = P.run(docs, opts, judge, lambda stage, d, t: _set(job_id, stage=stage, done=d, total=t))
        res['job'] = {'id': job_id, 'name': JOBS[job_id]['name'], 'source': kind, 'created': JOBS[job_id]['created'],
                      'finished': time.time()}
        with open(os.path.join(JOBS_DIR, f'{job_id}.json'), 'w') as f: json.dump(res, f, ensure_ascii=False)
        _set(job_id, status='done', stage='Done', summary=res['summary'])
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status='error', stage='Failed', error=f'{type(e).__name__}: {e}')


@app.get('/api/health')
def health():
    try: key, _ = settings(); judge_ready = bool(key)
    except Exception: judge_ready = False
    has = lambda m: importlib.util.find_spec(m) is not None
    gcs = False
    try: gcs = has('google.cloud.storage')
    except ModuleNotFoundError: pass
    return {'ok': True, 'v4_model': os.path.isdir(P.V4_MODEL), 'judge_ready': judge_ready, 'judge_modes': list(MODES),
            'connectors': {'upload': True, 'url': True, 's3': has('boto3'), 'azure': has('azure.storage.blob') if has('azure') else False, 'gcs': gcs},
            'defaults': P.Options().__dict__, 'file_types': sorted(sources.TEXT_EXT)}


@app.post('/api/jobs')
async def create_job(source: str = Form(...), config: str = Form('{}'), options: str = Form('{}'), name: str = Form(''),
                     files: Optional[List[UploadFile]] = File(None)):
    try: cfg = json.loads(config or '{}'); raw = json.loads(options or '{}')
    except json.JSONDecodeError: raise HTTPException(400, 'config and options must be JSON')
    fields = P.Options().__dict__
    opts = P.Options(**{k: type(fields[k])(v) for k, v in raw.items() if k in fields and v is not None and v != ''})
    if opts.judge_mode not in MODES: raise HTTPException(400, f'judge_mode must be one of {list(MODES)}')
    if opts.scope not in ('all', 'within'): raise HTTPException(400, "scope must be 'all' or 'within'")
    blobs = []
    if source == 'upload':
        if not files: raise HTTPException(400, 'Add at least one file.')
        for f in files:
            data = await f.read()
            if len(data) > sources.MAX_BYTES: raise HTTPException(413, f'{f.filename} is too large')
            blobs.append((f.filename or 'upload.txt', data))
    job_id = uuid.uuid4().hex[:12]
    label = name.strip() or (', '.join(n for n, _ in blobs[:3]) if blobs else f'{source}: ' + (cfg.get('bucket') or cfg.get('container') or 'links'))
    JOBS[job_id] = {'id': job_id, 'name': label[:120], 'status': 'queued', 'stage': 'Queued', 'done': 0, 'total': 0,
                    'created': time.time(), 'updated': time.time()}
    threading.Thread(target=_work, args=(job_id, source, cfg, blobs, opts), daemon=True).start()
    return JOBS[job_id]


@app.get('/api/jobs')
def list_jobs():
    saved = []
    for fn in sorted(os.listdir(JOBS_DIR), key=lambda f: -os.path.getmtime(os.path.join(JOBS_DIR, f)))[:30]:
        if fn.endswith('.json') and fn[:-5] not in JOBS:
            saved.append({'id': fn[:-5], 'status': 'done', 'name': fn[:-5], 'updated': os.path.getmtime(os.path.join(JOBS_DIR, fn))})
    return sorted(JOBS.values(), key=lambda j: -j['created']) + saved


@app.get('/api/jobs/{job_id}')
def job(job_id: str):
    if job_id in JOBS: return JOBS[job_id]
    if os.path.exists(os.path.join(JOBS_DIR, f'{job_id}.json')): return {'id': job_id, 'status': 'done', 'stage': 'Done'}
    raise HTTPException(404, 'No such job')


@app.get('/api/jobs/{job_id}/results')
def results(job_id: str):
    if not job_id.isalnum(): raise HTTPException(400, 'bad job id')
    path = os.path.join(JOBS_DIR, f'{job_id}.json')
    if not os.path.exists(path): raise HTTPException(404, 'Results are not ready')
    return FileResponse(path, media_type='application/json')


if os.path.isdir(WEB_DIR):
    app.mount('/', StaticFiles(directory=WEB_DIR, html=True), name='web')
