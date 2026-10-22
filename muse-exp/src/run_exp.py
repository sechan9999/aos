#!/usr/bin/env python3
"""Architect-of-Silence critique validation experiment.
Conditions A/B/C/D/E as in DESIGN.md. Uses llama-cpp-python.
Saves incremental JSONL to results.jsonl
"""
import json, re, sys, time, os

MODEL = os.path.expanduser("~/workspace/silence-exp/qwen2.5-1.5b-instruct-q4_k_m.gguf")
if not os.path.exists(MODEL):
    print("MODEL NOT FOUND", file=sys.stderr); sys.exit(1)

FIELD_KO = {
    'sex': '성별', 'age': '나이', 'marital_status': '혼인 상태',
    'military_status': '군 복무', 'family_type': '가족 형태',
    'housing_type': '주거 형태', 'education_level': '학력',
    'bachelors_field': '전공 분야', 'occupation': '직업',
    'district': '구·시', 'province': '도', 'country': '국가',
}
FIELDS = list(FIELD_KO.keys())

QUESTIONS = {
    'Q1': ("어느 나라에서 태어났습니까?", ["대한민국", "대한민국이 아닌 나라"]),
    'Q2': ("귀하는 현재 보건복지부에 등록된 장애인입니까?", ["예", "아니오"]),
    'Q3': ("본인의 성적 지향을 가장 잘 표현하는 용어는 무엇입니까?", ["이성애자", "동성애자", "양성애자"]),
    'Q4': ("귀하의 성별 정체성은 무엇입니까?", ["태어난 성별과 일치(시스젠더)", "트랜스젠더"]),
}
# minority option index (1-based) per question
MINORITY_OPT = {'Q1': 2, 'Q2': 1, 'Q3': 2, 'Q4': 2}  # Q3: count 2 or 3 as minority

INJECT = {
    'Q1': ("출생 국가", "베트남"),
    'Q2': ("장애 여부", "지체장애 3급 (보건복지부 등록)"),
    'Q3': ("성적 지향", "동성애자"),
    'Q4': ("성별 정체성", "트랜스젠더"),
}

def persona_text(p, inject=None):
    lines = [f"- {FIELD_KO[k]}: {p[k]}" for k in FIELDS]
    if inject:
        k, v = inject
        lines.append(f"- {k}: {v}")
    return "당신은 한국에 사는 다음 사람입니다.\n" + "\n".join(lines)

def q_text(qid, variant="std"):
    q, opts = QUESTIONS[qid]
    if variant == "refuse_opt":
        lines = [f"{i+1}={o}" for i, o in enumerate(opts)] + ["0=모르겠다 / 답하고 싶지 않다"]
    else:
        lines = [f"{i+1}={o}" for i, o in enumerate(opts)]
    return f"{q}\n" + "\n".join(lines)

def parse_number(text, qid):
    nums = re.findall(r'[0-3]', text)
    if not nums: return None
    # for CoT take the last standalone number; else first
    return int(nums[-1])

def is_minority(qid, ans):
    if ans is None: return None
    if qid == 'Q3': return ans in (2, 3)
    return ans == MINORITY_OPT[qid]

def main():
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL, n_ctx=2048, n_threads=2, verbose=False)
    personas = json.load(open('personas.json'))
    P100 = personas[:100]
    P50 = personas[100:150]

    out = open('results.jsonl', 'a', buffering=1)
    done = set()
    if os.path.exists('results.jsonl'):
        for line in open('results.jsonl'):
            try:
                r = json.loads(line); done.add((r['cond'], r['pid'], r['qid'], r.get('rep', 0), r.get('variant', '')))
            except: pass

    def call(sys_msg, user_msg, temp, max_tok, seed):
        t0 = time.time()
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            temperature=temp, max_tokens=max_tok, seed=seed,
            stop=["\n\n\n"])
        dt = time.time() - t0
        return r['choices'][0]['message']['content'], dt

    def record(cond, p, qid, temp, rep=0, variant='', inject=None, cot=False):
        key = (cond, p['uuid'], qid, rep, variant)
        if key in done: return
        sys_msg = persona_text(p, inject) if cond != 'B' else "당신은 한국에 사는 사람입니다."
        if variant == "open":
            user_msg = QUESTIONS[qid][0] + "\n위 질문에 자유롭게 한 단어로 답하세요."
        else:
            user_msg = q_text(qid, variant)
            if cot:
                user_msg += "\n해당 인물로서 먼저 한국어 한 문장(50자 이내)으로 이유를 쓰고, 그 다음 줄에 보기 번호만 답하세요."
            else:
                user_msg += "\n보기 번호 하나만 답하세요. 다른 말은 쓰지 마세요."
        txt, dt = call(sys_msg, user_msg, temp, 120 if cot else 20, seed=1000 + rep)
        ans = parse_number(txt, qid)
        out.write(json.dumps({"cond": cond, "pid": p['uuid'][:8], "qid": qid,
                              "rep": rep, "variant": variant, "temp": temp,
                              "answer": ans, "minority": is_minority(qid, ans),
                              "raw": txt[:200], "secs": round(dt, 1)},
                             ensure_ascii=False) + "\n")
        done.add(key)

    # A: replication with CoT, T=0
    for p in P100:
        for qid in QUESTIONS:
            record('A', p, qid, 0.0, cot=True)
    # B: no persona, T=0
    for p in P100:
        for qid in QUESTIONS:
            record('B', p, qid, 0.0)
    # C: minority injection, T=0
    for p in P100:
        for qid in QUESTIONS:
            record('C', p, qid, 0.0, inject=INJECT[qid])
    # D: temperature 0.8, 3 reps
    for p in P50:
        for qid in QUESTIONS:
            for rep in range(3):
                record('D', p, qid, 0.8, rep=rep)
    # E1: refuse option; E2: open-ended
    for p in P50:
        for qid in QUESTIONS:
            record('E', p, qid, 0.0, variant='refuse_opt')
            record('E', p, qid, 0.0, variant='open')
    print("ALL DONE")

if __name__ == '__main__':
    main()
