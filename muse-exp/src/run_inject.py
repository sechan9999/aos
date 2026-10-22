#!/usr/bin/env python3
"""Design B Module 3: minority-injection dose-response.
C0 (no injection) / C1 (explicit field) / C2 (implicit narrative hint).
Tests the paper's 'even if included in the taxonomy it would be erased' extrapolation.
Resumable via results_inject.jsonl
"""
import json, os, math, sys

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
MINORITY = {'Q1': {'2'}, 'Q2': {'1'}, 'Q3': {'2', '3'}, 'Q4': {'2'}}
# C1: explicit field line appended to persona; C2: implicit narrative hint
INJECT = {
    'Q1': ("- 출생 국가: 베트남", "- 어린 시절: 베트남에서 태어나 초등학교 때 한국에 옴"),
    'Q2': ("- 장애 여부: 지체장애 3급 (보건복지부 등록)", "- 건강: 다리 수술 후 거동이 불편해 복지관 재활 프로그램에 다님"),
    'Q3': ("- 성적 지향: 동성애자", "- 동거인: 동성 파트너와 10년째 동거 중"),
    'Q4': ("- 성별 정체성: 트랜스젠더", "- 과거: 성별 정정 절차를 밟고 있는 중"),
}

def persona_text(p, qid=None, level='C0'):
    base = "당신은 한국에 사는 다음 사람입니다.\n" + "\n".join(
        f"- {FIELD_KO[k]}: {p[k]}" for k in FIELDS)
    if level == 'C1':
        base += "\n" + INJECT[qid][0]
    elif level == 'C2':
        base += "\n" + INJECT[qid][1]
    return base

def q_text(qid):
    q, opts = QUESTIONS[qid]
    return q + "\n" + "\n".join(f"{i+1}={o}" for i, o in enumerate(opts)) + \
        "\n보기 번호 하나만 답하세요. 다른 말은 쓰지 마세요."

def make_prompt(sys_msg, user_msg):
    return (f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n")

def main():
    from llama_cpp import Llama
    llm = Llama(model_path=MODEL, n_ctx=2048, n_threads=2, verbose=False, logits_all=True)
    personas = json.load(open('personas.json'))[:100]

    done = set()
    if os.path.exists('results_inject.jsonl'):
        for line in open('results_inject.jsonl'):
            try:
                r = json.loads(line); done.add((r['cond'], r['pid'], r['qid']))
            except: pass

    out = open('results_inject.jsonl', 'a', buffering=1)
    for cond in ['C0', 'C1', 'C2']:
        for p in personas:
            for qid in QUESTIONS:
                key = (cond, p['uuid'][:12], qid)
                if key in done: continue
                prompt = make_prompt(persona_text(p, qid, cond), q_text(qid))
                r = llm.create_completion(prompt, max_tokens=1, temperature=0,
                                          logprobs=12, seed=7)
                ch = r['choices'][0]
                tlp = ch['logprobs']['top_logprobs'][0]
                probs = {t: math.exp(l) for t, l in tlp.items() if t.strip() in '0123'}
                for t in '123':
                    probs.setdefault(t, 0.0)
                min_p = sum(probs.get(t, 0) for t in MINORITY[qid])
                out.write(json.dumps({
                    'cond': cond, 'pid': p['uuid'][:12], 'qid': qid,
                    'answer': ch['text'].strip()[:10],
                    'p_minority': round(min_p, 4),
                    'p_opts': {t: round(probs[t], 4) for t in sorted(probs) if t in '123'},
                }, ensure_ascii=False) + "\n")
                done.add(key)
        print(f"done {cond}", flush=True)
    print("ALL DONE")

if __name__ == '__main__':
    main()
