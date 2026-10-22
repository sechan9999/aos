#!/usr/bin/env python3
"""Design B Module 4: wording sensitivity.
P1 (paper-style closed) / P2 (reversed framing: minority-first wording) /
P3 (open-ended one-word answer).
Resumable via results_wording.jsonl
"""
import json, os, math, re, sys

MODEL = os.path.expanduser("~/workspace/silence-exp/qwen2.5-1.5b-instruct-q4_k_m.gguf")
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
MINORITY_WORDS = {
    'Q1': ['베트남', '중국', '필리핀', '미국', '일본', '외국'],
    'Q2': ['예', '등록', '장애'],
    'Q3': ['동성애', '양성애', '게이', '레즈비언', '바이'],
    'Q4': ['트랜스젠더', '트랜스'],
}
MAJORITY_WORDS = {
    'Q1': ['대한민국', '한국'],
    'Q2': ['아니오', '아니요'],
    'Q3': ['이성애'],
    'Q4': ['시스젠더', '일치'],
}

def persona_text(p):
    return "당신은 한국에 사는 다음 사람입니다.\n" + "\n".join(
        f"- {FIELD_KO[k]}: {p[k]}" for k in FIELDS)

def q_text(qid, variant):
    q, opts = QUESTIONS[qid]
    if variant == 'P1':
        return q + "\n" + "\n".join(f"{i+1}={o}" for i, o in enumerate(opts)) + \
            "\n보기 번호 하나만 답하세요. 다른 말은 쓰지 마세요."
    if variant == 'P2':
        # reversed framing: ask minority-first, minority option listed first
        rev = [opts[-1]] + opts[:-1] if len(opts) > 2 else list(reversed(opts))
        intro = {"Q1": "외국에서 태어나셨습니까? 해당하면 1번, 아니면 2번을 고르세요.",
                 "Q2": "장애 등록이 되어 있습니까? 되어 있으면 1번, 아니면 2번을 고르세요.",
                 "Q3": "소수자 성적 지향(동성애자·양성애자)에 해당하십니까? 해당하면 1번을 고르세요.",
                 "Q4": "트랜스젠더이십니까? 맞으면 1번, 아니면 2번을 고르세요."}[qid]
        return intro + "\n" + "\n".join(f"{i+1}={o}" for i, o in enumerate(rev)) + \
            "\n보기 번호 하나만 답하세요. 다른 말은 쓰지 마세요."
    # P3: open-ended
    return q + "\n한 단어로 자유롭게 답하세요."

def make_prompt(sys_msg, user_msg):
    return (f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n")

def classify_open(qid, text):
    t = text.strip()
    for w in MINORITY_WORDS[qid]:
        if w in t: return 'minority'
    for w in MAJORITY_WORDS[qid]:
        if w in t: return 'majority'
    return 'unclear'

def main():
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL, n_ctx=2048, n_threads=2, verbose=False, logits_all=True)
    personas = json.load(open('personas.json'))[:50]

    done = set()
    if os.path.exists('results_wording.jsonl'):
        for line in open('results_wording.jsonl'):
            try:
                r = json.loads(line); done.add((r['cond'], r['pid'], r['qid']))
            except: pass

    out = open('results_wording.jsonl', 'a', buffering=1)
    for cond in ['P1', 'P2', 'P3']:
        for p in personas:
            for qid in QUESTIONS:
                key = (cond, p['uuid'][:12], qid)
                if key in done: continue
                prompt = make_prompt(persona_text(p), q_text(qid, cond))
                mt = 1 if cond in ('P1', 'P2') else 20
                r = llm.create_completion(prompt, max_tokens=mt, temperature=0,
                                          logprobs=12 if cond != 'P3' else 1, seed=7)
                ch = r['choices'][0]
                txt = ch['text']
                rec = {'cond': cond, 'pid': p['uuid'][:12], 'qid': qid,
                       'answer': txt.strip()[:40]}
                if cond in ('P1', 'P2'):
                    tlp = ch['logprobs']['top_logprobs'][0]
                    probs = {t: math.exp(l) for t, l in tlp.items() if t.strip() in '0123'}
                    for t in '123':
                        probs.setdefault(t, 0.0)
                    # P2 has reversed option order: map back to minority correctly
                    if cond == 'P2':
                        min_p = probs.get('1', 0.0)  # option 1 is the minority-first option
                    else:
                        min_set = {'Q1': {'2'}, 'Q2': {'1'}, 'Q3': {'2', '3'}, 'Q4': {'2'}}[qid]
                        min_p = sum(probs.get(t, 0) for t in min_set)
                    rec['p_minority'] = round(min_p, 4)
                    rec['p_opts'] = {t: round(probs[t], 4) for t in sorted(probs) if t in '123'}
                else:
                    rec['class'] = classify_open(qid, txt)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done.add(key)
        print(f"done {cond}", flush=True)
    print("ALL DONE")

if __name__ == '__main__':
    main()
