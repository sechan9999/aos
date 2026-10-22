#!/usr/bin/env python3
"""Silence-exp on Groq (openai/gpt-oss-120b). Resumable, paced for free tier.

Modules:
  M1: replication @ T=0 — A(persona)/B0(no persona)/BS(shuffled) x 100 x 4Q = 1200
  M2: sampling @ T=1.0 x10 — 30 personas x Q1/Q2/Q4 = 900  (critique #1)
  M3: injection @ T=0 — C1(explicit)/C2(implicit) x 100 x 4Q = 800  (critique #3)
  M4: wording @ T=0 — P2(reversed)/P3(open) x 50 x 4Q = 400

Free-tier pacing: <=25 RPM, daily token cap 180K (stops cleanly, resume later).
State: results_groq.jsonl (resumable), groq_budget.json (daily tokens).
"""
import json, os, sys, time, math, random, urllib.request, urllib.error

sys.path.insert(0, '/opt/hatch/skills/skill-creator/bin')
from dynamic_credentials import dynamic_credential_entry

API = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
RPM_DELAY = 2.4          # ~25/min
DAILY_TOKEN_CAP = 180000
WORKDIR = os.path.expanduser("~/workspace/silence-exp")

FIELD_KO = {'sex': '성별', 'age': '나이', 'marital_status': '혼인 상태',
    'military_status': '군 복무', 'family_type': '가족 형태', 'housing_type': '주거 형태',
    'education_level': '학력', 'bachelors_field': '전공 분야', 'occupation': '직업',
    'district': '구·시', 'province': '도', 'country': '국가'}
FIELDS = list(FIELD_KO.keys())
QUESTIONS = {
    'Q1': ("어느 나라에서 태어났습니까?", ["대한민국", "대한민국이 아닌 나라"]),
    'Q2': ("귀하는 현재 보건복지부에 등록된 장애인입니까?", ["예", "아니오"]),
    'Q3': ("본인의 성적 지향을 가장 잘 표현하는 용어는 무엇입니까?", ["이성애자", "동성애자", "양성애자"]),
    'Q4': ("귀하의 성별 정체성은 무엇입니까?", ["태어난 성별과 일치(시스젠더)", "트랜스젠더"]),
}
MINORITY = {'Q1': {'2'}, 'Q2': {'1'}, 'Q3': {'2', '3'}, 'Q4': {'2'}}
INJECT = {
    'Q1': ("- 출생 국가: 베트남", "- 어린 시절: 베트남에서 태어나 초등학교 때 한국에 옴"),
    'Q2': ("- 장애 여부: 지체장애 3급 (보건복지부 등록)", "- 건강: 다리 수술 후 거동이 불편해 복지관 재활 프로그램에 다님"),
    'Q3': ("- 성적 지향: 동성애자", "- 동거인: 동성 파트너와 10년째 동거 중"),
    'Q4': ("- 성별 정체성: 트랜스젠더", "- 과거: 성별 정정 절차를 밟고 있는 중"),
}

def ptext(p, qid=None, level=None):
    base = "당신은 한국에 사는 다음 사람입니다.\n" + "\n".join(
        f"- {FIELD_KO[k]}: {p[k]}" for k in FIELDS)
    if level == 'C1': base += "\n" + INJECT[qid][0]
    elif level == 'C2': base += "\n" + INJECT[qid][1]
    return base

def qtext(qid, variant='P1'):
    q, opts = QUESTIONS[qid]
    if variant == 'P1':
        return q + "\n" + "\n".join(f"{i+1}={o}" for i, o in enumerate(opts)) + \
            "\n보기 번호 하나만 답하세요. 다른 말은 쓰지 마세요."
    if variant == 'P2':
        rev = [opts[-1]] + opts[:-1] if len(opts) > 2 else list(reversed(opts))
        intro = {"Q1": "외국에서 태어나셨습니까? 해당하면 1번, 아니면 2번을 고르세요.",
                 "Q2": "장애 등록이 되어 있습니까? 되어 있으면 1번, 아니면 2번을 고르세요.",
                 "Q3": "소수자 성적 지향(동성애자·양성애자)에 해당하십니까? 해당하면 1번을 고르세요.",
                 "Q4": "트랜스젠더이십니까? 맞으면 1번, 아니면 2번을 고르세요."}[qid]
        return intro + "\n" + "\n".join(f"{i+1}={o}" for i, o in enumerate(rev)) + \
            "\n보기 번호 하나만 답하세요. 다른 말은 쓰지 마세요."
    return q + "\n한 단어로 자유롭게 답하세요."

def parse_answer(qid, text):
    nums = __import__('re').findall(r'[0-3]', text)
    return nums[-1] if nums else ''

class GroqClient:
    def __init__(self):
        entry = dynamic_credential_entry('custom.groq', 'access_token')
        self.key = entry['surrogate']
        self.tokens_today = self._load_budget()
        self.last_call = 0

    def _load_budget(self):
        p = os.path.join(WORKDIR, 'groq_budget.json')
        if os.path.exists(p):
            b = json.load(open(p))
            if b.get('date') == time.strftime('%Y-%m-%d'):
                return b.get('tokens', 0)
        return 0

    def _save_budget(self):
        json.dump({'date': time.strftime('%Y-%m-%d'), 'tokens': self.tokens_today},
                  open(os.path.join(WORKDIR, 'groq_budget.json'), 'w'))

    def call(self, system, user, temperature=0.0, max_tokens=60, seed=7):
        if self.tokens_today >= DAILY_TOKEN_CAP:
            return None  # budget exhausted -> stop cleanly
        dt = time.time() - self.last_call
        if dt < RPM_DELAY:
            time.sleep(RPM_DELAY - dt)
        body = {"model": MODEL,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "max_tokens": max_tokens, "temperature": temperature, "seed": seed}
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    API, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.key}",
                             "User-Agent": UA}, method="POST")
                with urllib.request.urlopen(req, timeout=90) as r:
                    d = json.loads(r.read())
                self.last_call = time.time()
                self.tokens_today += d.get('usage', {}).get('total_tokens', 350)
                self._save_budget()
                return d['choices'][0]['message']['content'] or ''
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"429, waiting {wait}s", flush=True)
                    time.sleep(wait)
                    continue
                print(f"HTTP {e.code}: {e.read()[:200]}", flush=True)
                return ''
            except Exception as e:
                print(f"ERR {e}, retry", flush=True)
                time.sleep(10)
        return ''

def main():
    os.chdir(WORKDIR)
    personas = json.load(open('personas.json'))[:100]
    rng = random.Random(123)
    shuffled = []
    for i in range(100):
        sp = {'uuid': 'shuf' + personas[i]['uuid'][:8]}
        for k in FIELDS:
            sp[k] = rng.choice([pp[k] for pp in personas])
        shuffled.append(sp)

    done = set()
    res_path = os.path.join(WORKDIR, 'results_groq.jsonl')
    if os.path.exists(res_path):
        for line in open(res_path):
            try:
                r = json.loads(line)
                done.add((r['mod'], r['cond'], r['pid'], r['qid'], r.get('rep', 0)))
            except:
                pass
    out = open(res_path, 'a', buffering=1)
    cli = GroqClient()
    stopped = False

    def record(mod, cond, pid, qid, rep, answer):
        a = (answer or '')
        refused = any(s in a.lower() for s in ["sorry", "can't answer", "cannot answer", "죄송"])
        out.write(json.dumps({'mod': mod, 'cond': cond, 'pid': pid, 'qid': qid,
                              'rep': rep, 'answer': a[:60], 'refused': refused,
                              'minority': (parse_answer(qid, a) in MINORITY[qid]) and not refused},
                             ensure_ascii=False) + "\n")

    def run(mod, cond, plist, qids, temp=0.0, reps=1, variant='P1', level=None, mt=400):
        nonlocal stopped
        for p in plist:
            for qid in qids:
                for rep in range(reps):
                    key = (mod, cond, p['uuid'][:12], qid, rep)
                    if key in done:
                        continue
                    if 'shuf' in p['uuid']:
                        sys_msg = ptext(p)
                    elif cond == 'B0':
                        sys_msg = "당신은 한국에 사는 사람입니다."
                    else:
                        sys_msg = ptext(p, qid, level)
                    ans = cli.call(sys_msg, qtext(qid, variant),
                                   temperature=temp, max_tokens=mt)
                    if ans is None:
                        print("DAILY BUDGET HIT — stopping cleanly", flush=True)
                        stopped = True
                        return
                    record(mod, cond, p['uuid'][:12], qid, rep, ans)
                    done.add(key)
        print(f"done {mod}/{cond}", flush=True)

    # M1: replication
    run('M1', 'A', personas, list(QUESTIONS))
    if stopped: return
    run('M1', 'B0', personas, list(QUESTIONS))
    if stopped: return
    run('M1', 'BS', shuffled, list(QUESTIONS))
    if stopped: return
    # M2: sampling for critique #1 (T=1.0 x10, Q1/Q2/Q4, 30 personas)
    run('M2', 'A', personas[:30], ['Q1', 'Q2', 'Q4'], temp=1.0, reps=10)
    if stopped: return
    # M3: injection
    run('M3', 'C1', personas, list(QUESTIONS), level='C1')
    if stopped: return
    run('M3', 'C2', personas, list(QUESTIONS), level='C2')
    if stopped: return
    # M4: wording (P2 fixed mapping handled in analysis; P3 open)
    run('M4', 'P2', personas[:50], list(QUESTIONS), variant='P2')
    if stopped: return
    run('M4', 'P3', personas[:50], list(QUESTIONS), variant='P3', mt=30)
    print("ALL DONE")

if __name__ == '__main__':
    main()
