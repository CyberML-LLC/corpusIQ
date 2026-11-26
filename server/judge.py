# Copyright 2026 CyberML (https://cyberml.ai)
# SPDX-License-Identifier: Apache-2.0
"""LLM judge for candidate pairs: prompt C (revised rules, short reasoning before the decision), the best prompt in
CyberML's internal judge tests.

Two judges by default, deepseek-v4-flash and deepseek-v4-pro, combined as:
  both say Conflict                          -> Conflict
  one says Conflict, or either says Unclear  -> Unsure
  otherwise either says Duplicate            -> Duplicate
  otherwise                                  -> No Conflict
On the confdoc004 gold set (16 confirmed pairs) this surfaces 11 as Conflict or Unsure; the best single judge finds 9.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

GENERIC_DOMAIN = ("DOMAIN CONTEXT: General documents. Statements should be factually consistent with each other "
                  "within and across documents.")

RULES = """Treat the following as NOT a contradiction:
- Paraphrases with identical meaning
- Differences in specificity (one omits non-essential detail)
- Optional or conditional statements vs. unconditional statements elsewhere
- Different scope or audience (regions/editions/versions/populations), when the statements clearly refer to different ones
- Different ordering of facts that does not change the meaning
- Ranges/approximations compatible after unit/date normalization (e.g., 'about 12,000' vs '12,142')
- Approximate vs. precise figures that are compatible (e.g., 'about 50' vs. '52')
- Attributed claims ONLY when the text itself presents the matter as disputed (e.g. 'sources disagree', 'others claim')
- Statements about different geographic scopes (e.g., national vs. regional stats)
- One sentence gives more detail, the other less - no contradiction

Gaps: a missing fact counts as a contradiction ONLY IF both statements describe the SAME subject and timeframe AND the missing fact is asserted as definitive by the other statement.

Normalization checklist (apply before judging):
- Units (e.g., km vs miles, kg vs lb), dates (YYYY-MM-DD), time windows, numeric ranges
- Synonyms and near-synonyms (e.g., founded/established, died/passed away)
- Conditional markers (if/when/unless) and scope qualifiers ("in the United States", "as of 2020", "in the second edition")

Use your alignment assessment to drive the decision. If the ENTITY/SUBJECT itself cannot be established as the same, prefer Unclear. It is common and EXPECTED for a single isolated sentence to leave scope, time/version, or conditions unstated - use "Unclear" for those fields rather than guessing, and do not treat Unclear as evidence against a contradiction.

Signs of a genuine contradiction:
- Two different dates for the same event
- Mismatched numeric values (e.g., population, elevation, price)
- Contradictory status (alive vs. dead, founded vs. dissolved)
- Conflicting attribution (who discovered, who created)
- Contradictory superlatives (first vs. not-first, largest vs. smallest)
- Two different dates, ages, values or descriptions for the same event, origin, first occurrence, record or rank, even if
  one is phrased as a belief, an estimate or an attributed claim, or refers to time differently
- An attributed claim that disagrees with a fact the document states elsewhere as definite

Same-event rule: if both statements plausibly describe the same event, origin, quantity or rank, judge them as the same
and decide whether they can both be true. Treat them as different events or periods ONLY if the text makes that explicit.

Classify the relationship between the two statements. They may come from the same document or from two different documents.
- "Conflict": they cannot both be true under the rules above.
- "Duplicate": they state the same fact (paraphrase), even if one adds minor detail.
- "No Conflict": compatible and not the same fact.
- "Unclear": cannot be decided from the two sentences.
Respond with a JSON object: {"reason": "<one or two sentences: are these the same event/fact, and can both be true?>", "decision": "<label>"}"""

JUDGES = {
    'flash': dict(model='deepseek-v4-flash', temperature=0, max_tokens=400, extra_body={'thinking': {'type': 'disabled'}}),
    'pro': dict(model='deepseek-v4-pro', temperature=0, max_tokens=400, extra_body={'thinking': {'type': 'disabled'}}),
}
MODES = {'both': ['flash', 'pro'], 'flash': ['flash'], 'pro': ['pro']}
LABELS = ('Conflict', 'Duplicate', 'No Conflict', 'Unclear')


def system_prompt(domain: str = '') -> str:
    domain = (domain or '').strip()
    head = f'DOMAIN CONTEXT: {domain}' if domain else GENERIC_DOMAIN
    return head + '\n\n' + RULES


def combine(decisions: List[Optional[str]]) -> str:
    d = [x for x in decisions if x]
    if not d: return 'Unsure'
    if all(x == 'Conflict' for x in d): return 'Conflict'
    if 'Conflict' in d or 'Unclear' in d or len(d) < len(decisions): return 'Unsure'
    if 'Duplicate' in d: return 'Duplicate'
    return 'No Conflict'


class Judge:
    def __init__(self, api_key: str, base_url: str, mode: str = 'both', domain: str = '', workers: int = 12):
        from openai import OpenAI
        if mode not in MODES: raise ValueError(f'judge mode must be one of {list(MODES)}')
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.names = MODES[mode]; self.system = system_prompt(domain); self.workers = workers

    def _ask(self, name: str, title_a: str, a: str, title_b: str, b: str) -> Dict:
        msg = f'S1 (document: {title_a}): {a}\nS2 (document: {title_b}): {b}'
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(response_format={'type': 'json_object'}, **JUDGES[name],
                    messages=[{'role': 'system', 'content': self.system}, {'role': 'user', 'content': msg}])
                out = json.loads(r.choices[0].message.content)
                d = out.get('decision'); d = d if d in LABELS else None
                return {'decision': d, 'reason': str(out.get('reason') or ''), 'model': JUDGES[name]['model']}
            except Exception as e:  # network or parse error: retry, then record it
                err = f'{type(e).__name__}: {str(e)[:160]}'; time.sleep(1.5 * (attempt + 1))
        return {'decision': None, 'reason': '', 'error': err, 'model': JUDGES[name]['model']}

    def run(self, pairs: List[Dict], progress: Callable[[int, int], None]) -> List[Dict]:
        """pairs: dicts with title_a, a, title_b, b. Returns, per pair, {'label', 'judges': {name: {...}}}."""
        jobs = [(k, n) for k in range(len(pairs)) for n in self.names]
        res: Dict = {}; done = [0]

        def work(job):
            k, n = job; p = pairs[k]
            res[(k, n)] = self._ask(n, p['title_a'], p['a'], p['title_b'], p['b'])
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(jobs): progress(done[0], len(jobs))

        with ThreadPoolExecutor(self.workers) as ex: list(ex.map(work, jobs))
        out = []
        for k in range(len(pairs)):
            js = {n: res[(k, n)] for n in self.names}
            out.append({'label': combine([js[n]['decision'] for n in self.names]), 'judges': js})
        return out
