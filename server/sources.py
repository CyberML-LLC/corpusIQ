# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""Turn the web UI's data sources into IngestedDoc lists, reusing src/ingestion (PDF, DOCX, TXT, MD, CSV) and adding
HTML and JSON. Cloud SDKs are optional and imported only when that source is used:
  S3: boto3        Azure Blob: azure-storage-blob        Google Cloud Storage: google-cloud-storage
Credentials typed in the UI are optional; without them each SDK uses its default chain (environment variables,
shared config files, instance roles, gcloud application-default credentials)."""
from __future__ import annotations

import io
import json
import os
import re
import urllib.request
from html.parser import HTMLParser
from typing import Dict, List
from urllib.parse import unquote, urlparse

from pipeline import ROOT  # noqa: F401  (puts src/ on sys.path)
from ingestion.base import IngestedDoc
from ingestion.local_files import extract_docs

TEXT_EXT = {'.pdf', '.docx', '.txt', '.md', '.csv', '.html', '.htm', '.json'}
MAX_OBJECTS = int(os.environ.get('CORPUSIQ_MAX_OBJECTS', '500'))
MAX_BYTES = int(os.environ.get('CORPUSIQ_MAX_BYTES', str(200 * 1024 * 1024)))


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript'): self.skip += 1
        if tag in ('p', 'br', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'tr'): self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript') and self.skip: self.skip -= 1

    def handle_data(self, data):
        if not self.skip: self.parts.append(data)


def _json_texts(x) -> List[str]:
    """Strings from a JSON document: a list of strings, a list of objects with a text-like field, or nested."""
    if isinstance(x, str): return [x]
    if isinstance(x, list): return [t for v in x for t in _json_texts(v)]
    if isinstance(x, dict):
        for k in ('text', 'content', 'body', 'statement', 'sentence', 'description'):
            if isinstance(x.get(k), str): return [x[k]]
        return [t for v in x.values() for t in _json_texts(v)]
    return []


def parse_bytes(data: bytes, name: str, origin: str) -> List[IngestedDoc]:
    ext = os.path.splitext(name.lower())[1]
    if ext in ('.html', '.htm'):
        p = _Text(); p.feed(data.decode('utf-8', errors='replace'))
        text = re.sub(r'[ \t]+', ' ', ''.join(p.parts))
        return [IngestedDoc(title=os.path.basename(name), text=text, origin=origin)]
    if ext == '.json':
        texts = _json_texts(json.loads(data.decode('utf-8', errors='replace')))
        return [IngestedDoc(title=f'{os.path.basename(name)}#{k}', text=t, origin=f'{origin}#{k}') for k, t in enumerate(texts) if t.strip()]
    if ext not in TEXT_EXT:
        raise ValueError(f'unsupported file type {ext or "(none)"}; use one of {", ".join(sorted(TEXT_EXT))}')
    return extract_docs(io.BytesIO(data), name, origin=origin)


def from_uploads(files: List) -> List[IngestedDoc]:
    docs = []
    for name, data in files: docs += parse_bytes(data, name, f'upload:{name}')
    return docs


def _keep(key: str) -> bool:
    return not key.endswith('/') and os.path.splitext(key.lower())[1] in TEXT_EXT


def from_s3(cfg: Dict) -> List[IngestedDoc]:
    try: import boto3
    except ImportError: raise RuntimeError('S3 needs boto3 on the API server: pip install boto3')
    kw = {}
    if cfg.get('region'): kw['region_name'] = cfg['region']
    if cfg.get('access_key_id') and cfg.get('secret_access_key'):
        kw.update(aws_access_key_id=cfg['access_key_id'], aws_secret_access_key=cfg['secret_access_key'])
        if cfg.get('session_token'): kw['aws_session_token'] = cfg['session_token']
    if cfg.get('endpoint_url'): kw['endpoint_url'] = cfg['endpoint_url']   # S3-compatible stores (MinIO, R2, Wasabi)
    s3 = boto3.client('s3', **kw); bucket = cfg['bucket']; keys = []
    for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=cfg.get('prefix', '')):
        keys += [o['Key'] for o in page.get('Contents', []) if _keep(o['Key'])]
        if len(keys) >= MAX_OBJECTS: break
    docs = []
    for k in keys[:MAX_OBJECTS]:
        docs += parse_bytes(s3.get_object(Bucket=bucket, Key=k)['Body'].read(), k, f's3://{bucket}/{k}')
    return docs


def from_azure(cfg: Dict) -> List[IngestedDoc]:
    try: from azure.storage.blob import BlobServiceClient
    except ImportError: raise RuntimeError('Azure Blob needs azure-storage-blob on the API server: pip install azure-storage-blob')
    conn = cfg.get('connection_string') or os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if conn: svc = BlobServiceClient.from_connection_string(conn)
    elif cfg.get('account_url'): svc = BlobServiceClient(cfg['account_url'], credential=cfg.get('sas_token') or None)
    else: raise ValueError('Azure needs a connection string, or an account URL (with a SAS token for private containers).')
    cc = svc.get_container_client(cfg['container']); docs = []
    names = [b.name for b in cc.list_blobs(name_starts_with=cfg.get('prefix') or None) if _keep(b.name)][:MAX_OBJECTS]
    for n in names: docs += parse_bytes(cc.download_blob(n).readall(), n, f'azure://{cfg["container"]}/{n}')
    return docs


def from_gcs(cfg: Dict) -> List[IngestedDoc]:
    try: from google.cloud import storage
    except ImportError: raise RuntimeError('Google Cloud Storage needs google-cloud-storage on the API server: pip install google-cloud-storage')
    client = storage.Client(project=cfg.get('project') or None) if not cfg.get('anonymous') else storage.Client.create_anonymous_client()
    blobs = [b for b in client.list_blobs(cfg['bucket'], prefix=cfg.get('prefix') or None) if _keep(b.name)][:MAX_OBJECTS]
    docs = []
    for b in blobs: docs += parse_bytes(b.download_as_bytes(), b.name, f'gs://{cfg["bucket"]}/{b.name}')
    return docs


def from_urls(urls: List[str]) -> List[IngestedDoc]:
    """Public or pre-signed http(s) links (S3/GCS/Azure SAS links, Dropbox/OneDrive direct links, web pages)."""
    docs = []
    for u in urls[:MAX_OBJECTS]:
        u = u.strip()
        if not u: continue
        if urlparse(u).scheme not in ('http', 'https'): raise ValueError(f'only http(s) links are supported: {u[:80]}')
        req = urllib.request.Request(u, headers={'User-Agent': 'corpusIQ/1.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(MAX_BYTES + 1); ctype = r.headers.get('Content-Type', '')
        if len(data) > MAX_BYTES: raise ValueError(f'{u[:80]} is larger than {MAX_BYTES // 2**20} MB')
        name = unquote(os.path.basename(urlparse(u).path)) or 'page.html'
        if os.path.splitext(name.lower())[1] not in TEXT_EXT:
            name += '.pdf' if 'pdf' in ctype else '.html' if 'html' in ctype else '.csv' if 'csv' in ctype else '.txt'
        docs += parse_bytes(data, name, u.split('?')[0])
    return docs


def load(kind: str, cfg: Dict, files: List) -> List[IngestedDoc]:
    if kind == 'upload': return from_uploads(files)
    if kind == 's3': return from_s3(cfg)
    if kind == 'azure': return from_azure(cfg)
    if kind == 'gcs': return from_gcs(cfg)
    if kind == 'url': return from_urls(cfg.get('urls') or [])
    raise ValueError(f'unknown source {kind!r}')
