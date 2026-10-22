#!/usr/bin/env python3
"""Analyze results.jsonl -> summary table addressing critiques 1-3."""
import json, collections

rows = [json.loads(l) for l in open('results.jsonl') if l.strip()]
print(f"total responses: {len(rows)}")

def rate(cond, qid=None, variant='', minority_only=True):
    sel = [r for r in rows if r['cond'] == cond and r.get('variant', '') == variant
           and (qid is None or r['qid'] == qid)]
    n = len(sel)
    if not n: return (0, 0)
    m = sum(1 for r in sel if r['minority'])
    u = sum(1 for r in sel if r['answer'] is None)
    return (m, n, u)

for cond in ['A', 'B', 'C', 'D', 'E']:
    for v in (['', 'refuse_opt', 'open'] if cond == 'E' else ['']):
        m, n, u = rate(cond, variant=v)
        tag = f"{cond}/{v}" if v else cond
        print(f"{tag:12s} minority={m}/{n} ({100*m/n:.1f}%)  unparseable={u}")

print("\n-- per question (A/B/C) --")
for qid in ['Q1', 'Q2', 'Q3', 'Q4']:
    a = rate('A', qid); b = rate('B', qid); c = rate('C', qid)
    print(f"{qid}: A {a[0]}/{a[1]}  B {b[0]}/{b[1]}  C {c[0]}/{c[1]}")

print("\n-- D: per-persona minority count across 3 samples (T=0.8) --")
by_pq = collections.defaultdict(list)
for r in rows:
    if r['cond'] == 'D': by_pq[(r['pid'], r['qid'])].append(r['minority'])
hist = collections.Counter(sum(1 for v in vs if v) for vs in by_pq.values())
print("minority answers out of 3 samples:", dict(sorted(hist.items())), f"(n={len(by_pq)} persona-questions)")

print("\n-- E refuse_opt: chose 0 --")
for qid in ['Q1', 'Q2', 'Q3', 'Q4']:
    sel = [r for r in rows if r['cond'] == 'E' and r.get('variant') == 'refuse_opt' and r['qid'] == qid]
    z = sum(1 for r in sel if r['answer'] == 0)
    print(f"{qid}: refused={z}/{len(sel)}")
