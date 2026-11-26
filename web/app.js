// Copyright 2026 CyberML (https://cyberml.ai)
// SPDX-License-Identifier: Apache-2.0
/* corpusIQ web UI: talks to the API server in server/app.py. No build step, no dependencies. */
(() => {
'use strict';

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
}
const store = {
  get(k, d) { try { const v = localStorage.getItem('ciq.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('ciq.' + k, JSON.stringify(v)); } catch { /* storage blocked */ } },
};
const fmt = (n) => (n ?? 0).toLocaleString('en-US');
const pct = (a, b) => (b ? (100 * a / b) : 0);
const LABELS = ['Conflict', 'Unsure', 'Duplicate', 'No Conflict'];
const LABEL_COLOR = { Conflict: 'var(--conflict)', Unsure: 'var(--unsure)', Duplicate: 'var(--dup)', 'No Conflict': 'var(--ok)', Filtered: 'var(--filtered)' };
const JUDGE_NAME = { flash: 'DeepSeek V4 Flash', pro: 'DeepSeek V4 Pro' };
const STAGES = ['Reading documents', 'Splitting sentences', 'Embedding sentences', 'Finding candidate pairs', 'Scoring pairs with v4', 'Asking the LLM judges'];

/* ---------- API server ---------- */
const sameOrigin = /^https?:$/.test(location.protocol) && !location.hostname.endsWith('github.io');
let api = store.get('api', sameOrigin ? location.origin : 'http://localhost:8000');
let health = null;

async function call(path, opts) {
  const r = await fetch(api.replace(/\/$/, '') + path, opts);
  if (!r.ok) {
    let msg = r.status + ' ' + r.statusText;
    try { const j = await r.json(); if (j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}

async function checkServer() {
  const dot = $('#server-dot'), label = $('#server-label'), detail = $('#server-detail');
  dot.className = 'dot'; label.textContent = 'Checking server…';
  try {
    health = await call('/api/health');
    const problems = [];
    if (!health.v4_model) problems.push('the v4 model is missing on the server');
    if (!health.judge_ready) problems.push('DEEPSEEK_API_KEY is not set on the server');
    dot.className = 'dot ' + (problems.length ? 'warn' : 'ok');
    label.textContent = problems.length ? 'Server needs setup' : 'Server ready';
    const conn = Object.entries(health.connectors).filter(([, v]) => !v).map(([k]) => k.toUpperCase());
    detail.textContent = (problems.length ? 'Connected, but ' + problems.join(' and ') + '. ' : 'Connected to ' + api + '. ') +
      (conn.length ? 'Not installed on the server: ' + conn.join(', ') + ' (see server/README.md).' : '');
  } catch (e) {
    health = null; dot.className = 'dot bad'; label.textContent = 'Server offline';
    const local = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(api);
    const fromPublicSite = location.protocol === 'https:' && !/^(localhost|127\.0\.0\.1)$/.test(location.hostname);
    if (local && fromPublicSite) {
      // Chrome asks before a public site may reach this computer (Local Network Access); Safari blocks it outright.
      detail.replaceChildren('Could not reach ' + api + '. If the server is running, your browser is blocking this site from reaching your computer. ' +
        'In Chrome, click the icon left of the address bar, open Site settings, set Local network access to Allow, and reload. ' +
        'Or open ', el('a', { href: api, text: api }), ' directly, where the server shows this same page (this also works in Safari).');
      $('#server-panel').hidden = false;
    } else {
      detail.textContent = 'Could not reach ' + api + '. Start the API server, then press Connect.';
    }
  }
}

$('#api-url').value = api;
$('#server-btn').addEventListener('click', () => { $('#server-panel').hidden = !$('#server-panel').hidden; });
$('#api-save').addEventListener('click', () => {
  api = $('#api-url').value.trim().replace(/\/$/, '') || 'http://localhost:8000'; store.set('api', api); checkServer();
});

/* ---------- routing ---------- */
let results = null; let pollTimer = null;
function show(view) {
  for (const v of ['new', 'progress', 'runs', 'dash']) $('#view-' + v).hidden = v !== view;
  $('#nav-new').toggleAttribute('aria-current', view === 'new'); if (view === 'new') $('#nav-new').setAttribute('aria-current', 'page');
  $('#nav-runs').toggleAttribute('aria-current', view === 'runs'); if (view === 'runs') $('#nav-runs').setAttribute('aria-current', 'page');
  window.scrollTo(0, 0);
}
async function route() {
  clearTimeout(pollTimer);
  const h = location.hash.slice(1);
  if (h.startsWith('job-')) return watchJob(h.slice(4));
  if (h === 'runs') { show('runs'); return loadRuns(); }
  if (h === 'local' && results) { show('dash'); return renderDash(); }
  show('new');
}
window.addEventListener('hashchange', route);

/* ---------- new analysis form ---------- */
let source = 'upload'; let files = [];
$$('#run-form .tabs button').forEach((b) => b.addEventListener('click', () => {
  source = b.dataset.src;
  $$('#run-form .tabs button').forEach((x) => x.setAttribute('aria-selected', String(x === b)));
  $$('#run-form .src').forEach((s) => { s.hidden = s.dataset.src !== source; });
}));

function renderFiles() {
  $('#file-list').replaceChildren(...files.map((f, i) => el('li', {},
    el('span', { text: f.name }), el('span', { class: 'size', text: (f.size / 1024 < 1024 ? Math.max(1, Math.round(f.size / 1024)) + ' KB' : (f.size / 1048576).toFixed(1) + ' MB') }),
    el('button', { type: 'button', 'aria-label': 'Remove ' + f.name, text: '×', onclick: () => { files.splice(i, 1); renderFiles(); } }))));
}
function addFiles(list) {
  for (const f of list) if (!files.some((x) => x.name === f.name && x.size === f.size)) files.push(f);
  renderFiles();
}
$('#files').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
const drop = $('#drop');
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => addFiles(e.dataTransfer.files));

const v = (id) => $(id).value.trim();
function sourceConfig() {
  switch (source) {
    case 'url': return { urls: v('#urls').split(/\s+/).filter(Boolean) };
    case 's3': return { bucket: v('#s3-bucket'), prefix: v('#s3-prefix'), region: v('#s3-region'), endpoint_url: v('#s3-endpoint'),
      access_key_id: v('#s3-key'), secret_access_key: v('#s3-secret') };
    case 'azure': return { container: v('#az-container'), prefix: v('#az-prefix'), connection_string: v('#az-conn'), account_url: v('#az-url'), sas_token: v('#az-sas') };
    case 'gcs': return { bucket: v('#gcs-bucket'), prefix: v('#gcs-prefix'), project: v('#gcs-project'), anonymous: $('#gcs-public').checked };
    default: return {};
  }
}
function validate(cfg) {
  if (source === 'upload' && !files.length) return 'Add at least one file.';
  if (source === 'url' && !cfg.urls.length) return 'Add at least one link.';
  if ((source === 's3' || source === 'gcs') && !cfg.bucket) return 'Enter the bucket name.';
  if (source === 'azure' && !cfg.container) return 'Enter the container name.';
  if (health && source !== 'upload' && source !== 'url' && !health.connectors[source])
    return 'This connector is not installed on the API server. See server/README.md.';
  return '';
}

$('#run-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = $('#form-msg'); const cfg = sourceConfig(); const err = validate(cfg);
  if (err) { msg.textContent = err; return; }
  if (!health) { await checkServer(); if (!health) { msg.textContent = 'The API server is offline. Start it, then try again.'; return; } }
  msg.textContent = ''; const btn = $('#run-btn'); btn.disabled = true; btn.textContent = 'Starting…';
  const fd = new FormData();
  fd.append('source', source); fd.append('config', JSON.stringify(cfg)); fd.append('name', v('#opt-name'));
  fd.append('options', JSON.stringify({ judge_mode: v('#opt-judge'), scope: v('#opt-scope'), max_calls: v('#opt-max'),
    v4_cutoff: v('#opt-cutoff'), domain: v('#domain') }));
  if (source === 'upload') files.forEach((f) => fd.append('files', f, f.name));
  try {
    const job = await call('/api/jobs', { method: 'POST', body: fd });
    location.hash = 'job-' + job.id;
  } catch (ex) { msg.textContent = 'Could not start: ' + ex.message; }
  finally { btn.disabled = false; btn.textContent = 'Analyze'; }
});

$('#open-file').addEventListener('change', async (e) => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    const r = JSON.parse(await f.text());
    if (!r || !r.summary || !Array.isArray(r.items)) throw new Error('not a corpusIQ results file');
    results = r; location.hash = 'local'; if (location.hash === '#local') { show('dash'); renderDash(); }
  } catch (ex) { $('#form-msg').textContent = 'Could not open ' + f.name + ': ' + ex.message; }
});

/* ---------- progress ---------- */
async function watchJob(id) {
  let job;
  try { job = await call('/api/jobs/' + id); }
  catch (e) { show('new'); $('#form-msg').textContent = 'Could not load that run: ' + e.message; return; }
  if (job.status === 'done') {
    try { results = await call('/api/jobs/' + id + '/results'); show('dash'); renderDash(); }
    catch (e) { show('new'); $('#form-msg').textContent = 'Could not load results: ' + e.message; }
    return;
  }
  show('progress');
  $('#prog-name').textContent = job.name || 'Analysis';
  const at = STAGES.findIndex((s) => (job.stage || '').startsWith(s));
  $('#stages').replaceChildren(...STAGES.map((s, i) => el('li', { class: job.status === 'error' ? (i < at ? 'done' : '') : i < at ? 'done' : i === at ? 'now' : '', text: s })));
  const frac = job.total ? job.done / job.total : 0;
  $('#prog-bar').style.width = (100 * Math.max(0, at) / STAGES.length + 100 * frac / STAGES.length).toFixed(1) + '%';
  if (job.status === 'error') {
    $('#prog-detail').textContent = 'The run failed: ' + (job.error || 'unknown error');
    return;
  }
  $('#prog-detail').textContent = job.status === 'queued' ? 'Waiting for the previous run to finish.' :
    (job.total ? job.stage + ': ' + fmt(job.done) + ' of ' + fmt(job.total) : job.stage);
  pollTimer = setTimeout(() => { if (location.hash === '#job-' + id) watchJob(id); }, 1500);
}

/* ---------- runs ---------- */
async function loadRuns() {
  const ul = $('#runs'); ul.replaceChildren(el('li', { class: 'empty', text: 'Loading…' }));
  try {
    const jobs = await call('/api/jobs');
    if (!jobs.length) { ul.replaceChildren(el('li', { class: 'empty', text: 'No runs yet. Start one from New analysis.' })); return; }
    ul.replaceChildren(...jobs.map((j) => el('li', {},
      el('a', { href: '#job-' + j.id, text: j.name || j.id }),
      el('span', { class: 'meta', text: (j.status === 'done' && j.summary ? fmt(j.summary.counts.Conflict) + ' conflicts · ' : j.status + ' · ') +
        new Date(1000 * (j.updated || j.created)).toLocaleString() }))));
  } catch (e) { ul.replaceChildren(el('li', { class: 'empty', text: 'Could not reach the API server: ' + e.message })); }
}

/* ---------- dashboard ---------- */
let tab = 'Conflict'; let query = ''; let docFilter = ''; let shown = 50; let visible = [];

function renderDash() {
  const r = results; const s = r.summary; const c = s.counts;
  $('#dash-title').textContent = r.job?.name || 'Results';
  $('#dash-when').textContent = r.job?.finished ? new Date(1000 * r.job.finished).toLocaleString() : 'Results file';
  const n = (x, word) => fmt(x) + ' ' + word + (x === 1 ? '' : 's');
  $('#funnel').textContent = n(s.documents_used ?? s.documents, 'document') + ' · ' + n(s.sentences, 'sentence') + ' · ' +
    n(s.candidate_pairs, 'candidate pair') + ' · ' + fmt(s.judged) + ' sent to the judges' + (s.judge_errors ? ' · ' + n(s.judge_errors, 'judge error') : '');
  const total = s.candidate_pairs || 1;
  const tiles = [
    ['Conflict', 'Conflict', 'Both judges agree'], ['Unsure', 'Unsure', 'Judges disagree or unclear'], ['Duplicate', 'Duplicate', 'Same fact twice'],
    ['No Conflict', 'No conflict', 'Judged compatible'], ['Filtered', 'Filtered out', 'Screened by v4, no LLM call'],
  ];
  $('#tiles').replaceChildren(...tiles.map(([k, name, sub]) => {
    const pressable = k !== 'Filtered';
    return el('button', { type: 'button', class: 'tile' + (pressable ? '' : ' static'), style: '--c:' + LABEL_COLOR[k],
      'aria-pressed': pressable ? String(tab === k) : null, disabled: !pressable || null,
      onclick: pressable ? () => { tab = k; shown = 50; renderDash(); } : null },
      el('span', { class: 'k', text: name }), el('span', { class: 'v', text: fmt(c[k]) }),
      el('span', { class: 'p', text: k === 'Filtered' ? pct(c[k], total).toFixed(0) + '% of pairs' : sub }));
  }));
  const judged = LABELS.reduce((a, k) => a + (c[k] || 0), 0) || 1;
  $('#stack').replaceChildren(...LABELS.filter((k) => c[k]).map((k) => el('span', { style: `width:${pct(c[k], judged)}%;background:${LABEL_COLOR[k]}`, title: k })));

  $('#label-tabs').replaceChildren(...LABELS.map((k) => el('button', { type: 'button', role: 'tab', 'aria-selected': String(tab === k),
    onclick: () => { tab = k; shown = 50; renderDash(); } }, k === 'No Conflict' ? 'No conflict' : k, el('span', { class: 'n', text: fmt(c[k]) }))));

  const docs = r.documents || []; const per = new Map();
  for (const it of r.items) for (const d of new Set([it.doc_a, it.doc_b])) {
    const p = per.get(d) || { Conflict: 0, Unsure: 0 }; if (it.label in p) p[it.label]++; per.set(d, p);
  }
  const sel = $('#doc-filter');
  sel.replaceChildren(el('option', { value: '', text: 'All documents' }),
    ...docs.filter((d) => per.has(d.id)).map((d) => el('option', { value: String(d.id), text: d.title })));
  sel.value = docFilter;
  const ranked = [...per.entries()].filter(([, p]) => p.Conflict + p.Unsure).sort((a, b) => (b[1].Conflict - a[1].Conflict) || (b[1].Unsure - a[1].Unsure)).slice(0, 15);
  $('#doc-rows').replaceChildren(...(ranked.length ? ranked.map(([d, p]) => el('tr', { class: String(d) === docFilter ? 'on' : '',
    onclick: () => { docFilter = String(d) === docFilter ? '' : String(d); shown = 50; renderDash(); } },
    el('td', { text: docs[d]?.title || 'Document ' + d, title: docs[d]?.title || '' }), el('td', { class: 'num', text: fmt(p.Conflict) }), el('td', { class: 'num', text: fmt(p.Unsure) })))
    : [el('tr', {}, el('td', { colspan: '3', text: 'No conflicts or unsure pairs.' }))]));
  renderPairs();
}

function renderPairs() {
  const q = query.toLowerCase();
  visible = results.items.filter((it) => it.label === tab &&
    (!docFilter || String(it.doc_a) === docFilter || String(it.doc_b) === docFilter) &&
    (!q || (it.a + ' ' + it.b + ' ' + it.title_a + ' ' + it.title_b).toLowerCase().includes(q)));
  const ul = $('#pairs');
  if (!visible.length) {
    ul.replaceChildren(el('li', { class: 'empty', text: query || docFilter ? 'No pairs match these filters.' : 'No pairs with this label.' }));
    $('#more').hidden = true; return;
  }
  ul.replaceChildren(...visible.slice(0, shown).map((it, i) => {
    const js = Object.entries(it.judges || {});
    const why = js.map(([, j]) => j.reason).find(Boolean) || '';
    return el('li', {}, el('button', { type: 'button', onclick: () => openPair(i) },
      el('span', { class: 'sent a', text: it.a }), el('span', { class: 'sent b', text: it.b }),
      why ? el('span', { class: 'why', text: why }) : null,
      el('span', { class: 'meta' },
        el('span', { text: it.doc_a === it.doc_b ? it.title_a : it.title_a + ' ↔ ' + it.title_b }),
        ...js.map(([n, j]) => el('span', { class: 'chip ' + (j.decision || '').replace(' ', ''), text: n + ': ' + (j.decision || 'error') })),
        el('span', { text: 'v4 ' + it.v4.toFixed(3) }))));
  }));
  $('#more').hidden = visible.length <= shown;
  $('#more').textContent = 'Show more (' + fmt(visible.length - shown) + ' left)';
}
$('#more').addEventListener('click', () => { shown += 50; renderPairs(); });
let searchT; $('#search').addEventListener('input', (e) => { clearTimeout(searchT); searchT = setTimeout(() => { query = e.target.value.trim(); shown = 50; renderPairs(); }, 150); });
$('#doc-filter').addEventListener('change', (e) => { docFilter = e.target.value; shown = 50; renderDash(); });

/* ---------- pair detail ---------- */
let current = -1;
function openPair(i) {
  current = i; const it = visible[i]; if (!it) return;
  const lab = $('#d-label'); lab.textContent = it.label; lab.className = 'pill ' + it.label.replace(' ', '');
  const doc = (t, k) => t + ' · sentence ' + (k + 1);
  $('#d-doc-a').textContent = doc(it.title_a, it.sent_a); $('#d-doc-b').textContent = doc(it.title_b, it.sent_b);
  $('#d-pa').textContent = it.ctx_a?.[0] || ''; $('#d-a').textContent = it.a; $('#d-na').textContent = it.ctx_a?.[1] || '';
  $('#d-pb').textContent = it.ctx_b?.[0] || ''; $('#d-b').textContent = it.b; $('#d-nb').textContent = it.ctx_b?.[1] || '';
  $('#d-judges').replaceChildren(...Object.entries(it.judges || {}).map(([n, j]) => el('div', { class: 'd-judge' },
    el('div', { class: 'h' }, el('span', { class: 'chip ' + (j.decision || '').replace(' ', ''), text: j.decision || 'error' }), el('span', { text: JUDGE_NAME[n] || j.model || n })),
    el('div', { text: j.reason || j.error || '' }))));
  const sc = [['v4 conflict score', it.v4.toFixed(4)], ['v4 duplicate score', (it.v4_dup ?? 0).toFixed(4)], ['Embedding similarity', it.cos.toFixed(3)],
    ['Where', it.doc_a === it.doc_b ? 'Same document' : 'Two documents']];
  $('#d-scores').replaceChildren(...sc.flatMap(([k, val]) => [el('dt', { text: k }), el('dd', { text: val })]));
  $('#drawer').hidden = false; $('#d-close').focus();
  $('#d-prev').disabled = i <= 0; $('#d-next').disabled = i >= Math.min(visible.length, shown) - 1;
}
const closePair = () => { $('#drawer').hidden = true; };
$('#d-close').addEventListener('click', closePair); $('#drawer-bg').addEventListener('click', closePair);
$('#d-prev').addEventListener('click', () => openPair(current - 1)); $('#d-next').addEventListener('click', () => openPair(current + 1));
document.addEventListener('keydown', (e) => {
  if ($('#drawer').hidden) return;
  if (e.key === 'Escape') closePair();
  if (e.key === 'ArrowLeft' && current > 0) openPair(current - 1);
  if (e.key === 'ArrowRight' && current < Math.min(visible.length, shown) - 1) openPair(current + 1);
});

/* ---------- downloads ---------- */
function save(name, text, type) {
  const a = el('a', { href: URL.createObjectURL(new Blob([text], { type })), download: name }); document.body.append(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
const slug = () => (results.job?.name || 'corpusiq').replace(/[^\w-]+/g, '-').slice(0, 60);
$('#dl-json').addEventListener('click', () => save(slug() + '.json', JSON.stringify(results), 'application/json'));
$('#dl-csv').addEventListener('click', () => {
  const q = (x) => '"' + String(x ?? '').replace(/"/g, '""') + '"';
  const names = [...new Set(results.items.flatMap((it) => Object.keys(it.judges || {})))];
  const head = ['label', 'document_a', 'sentence_a', 'document_b', 'sentence_b', 'v4_conflict', 'v4_duplicate', 'similarity',
    ...names.flatMap((n) => [n + '_decision', n + '_reason'])];
  const rows = results.items.map((it) => [it.label, it.title_a, it.a, it.title_b, it.b, it.v4, it.v4_dup, it.cos,
    ...names.flatMap((n) => [it.judges?.[n]?.decision, it.judges?.[n]?.reason])].map(q).join(','));
  save(slug() + '.csv', '﻿' + [head.map(q).join(','), ...rows].join('\n'), 'text/csv');
});

checkServer(); route();
})();
