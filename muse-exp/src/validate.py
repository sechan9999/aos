#!/usr/bin/env python3
"""Quick validation: can Qwen2.5-1.5B follow Korean persona instructions?"""
import json, os
from llama_cpp import Llama

llm = Llama(model_path=os.path.expanduser("~/workspace/silence-exp/qwen2.5-1.5b-instruct-q4_k_m.gguf"),
            n_ctx=2048, n_threads=2, verbose=False)
p = json.load(open('personas.json'))[0]
sys_msg = ("당신은 한국에 사는 다음 사람입니다.\n- 성별: %s\n- 나이: %s\n- 직업: %s\n- 거주: %s %s"
           % (p['sex'], p['age'], p['occupation'], p['province'], p['district']))
tests = [
    "어느 나라에서 태어났습니까?\n1=대한민국\n2=대한민국이 아닌 나라\n보기 번호 하나만 답하세요. 다른 말은 쓰지 마세요.",
    "귀하는 현재 보건복지부에 등록된 장애인입니까?\n1=예\n2=아니오\n보기 번호 하나만 답하세요. 다른 말은 쓰지 마세요.",
    "본인의 성적 지향을 가장 잘 표현하는 용어는 무엇입니까?\n1=이성애자\n2=동성애자\n3=양성애자\n보기 번호 하나만 답하세요. 다른 말은 쓰지 마세요.",
]
for t in tests:
    r = llm.create_chat_completion(
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": t}],
        temperature=0, max_tokens=20, seed=7)
    print("Q:", t[:30], "\nA:", repr(r['choices'][0]['message']['content']), "\n")
