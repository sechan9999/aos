"""모듈 1. 로그확률 감사 — 보기별 P(answer)를 1회 패스로 직접 읽음.

변형
  mode  : direct (번호만, max_tokens=1) | cot (이유 한 문장 → '답: N', 답 토큰 확률)
  order : orig (논문 보기 순서) | rev (보기 순서 뒤집고 번호 재부여 → 원 보기로 역매핑)
출력: 페르소나 × 문항 × 변형별 보기 확률 (top-20 밖 보기는 상한 = 20위 확률, 하한 0)
주의: 논문과 달리 문항을 대화 누적 없이 독립 호출, JSON 스키마 없이 평문.
"""
from __future__ import annotations
import json, math, os, re, sys, time, argparse, pathlib
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from openai import OpenAI
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from prompts import system_message

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=open(os.path.expanduser("~/.secrets/openrouter")).read().strip())
MODEL, PROVIDER = "meta-llama/llama-3.3-70b-instruct", os.environ.get("LP_PROVIDER", "Novita")


def item_prompt(it, order, mode):
    keys = list(it["options"])                     # 원 번호
    shown = keys if order == "orig" else keys[::-1]
    label = {str(i + 1): k for i, k in enumerate(shown)}   # 화면 번호 → 원 번호
    opts = " / ".join(f"{i+1}={it['options'][k]}" for i, k in enumerate(shown))
    tail = ("보기 번호 하나만 숫자로 답하세요." if mode == "direct"
            else "먼저 해당 인물로서 이유를 한국어 한 문장으로 쓰고, 다음 줄에 '답: 번호' 형식으로 답하세요.")
    return f"{it['text']} ({opts})\n{tail}", label


def digit_of(tok, prefix=None):
    """토큰 끝 숫자 한 자리. prefix가 주어지면 숫자 앞부분이 같아야 함 (': 1' vs ': 2')."""
    m = re.fullmatch(r"(\D*?)(\d)", tok.strip())
    if not m or (prefix is not None and m.group(1) != prefix): return None
    return m.group(2)


def read_dist(top, label, prefix=""):
    """top_logprobs → 원 보기 번호별 확률. 같은 숫자 토큰(' 1','1')은 합산."""
    p = {k: 0.0 for k in label.values()}
    seen = set()
    for t in top:
        d = digit_of(t.token, prefix)
        if d in label: p[label[d]] += math.exp(t.logprob); seen.add(label[d])
    floor = math.exp(min(t.logprob for t in top)) if top else 1.0
    return p, {k: (k not in seen) for k in p}, floor


def _req(messages, max_tokens):
    for a in range(8):
        try:
            return client.chat.completions.create(
                model=MODEL, temperature=0, seed=20260727, logprobs=True, top_logprobs=20,
                max_tokens=max_tokens, messages=messages,
                extra_body={"provider": {"order": [PROVIDER], "allow_fallbacks": False, "require_parameters": True}})
        except Exception as e:
            if a == 7: raise
            time.sleep(min(60, 3 * 2 ** a))


def call(persona, it, order, mode, sysmsg):
    """direct: 번호 토큰 1개. cot: (1) T=0으로 이유 생성 → (2) '이유\n답:' prefill 후 다음 토큰 1개."""
    prompt, label = item_prompt(it, order, mode)
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}]
    try:
        reason, tok_in = "", 0
        if mode == "cot":
            r1 = _req(msgs, 120)
            gen = r1.choices[0].message.content or ""
            m = re.search(r"^(.*?)\s*답\s*:", gen, re.S)
            if not m:
                return {"content": gen, "parse_fail": True}
            reason, tok_in = m.group(1).strip(), r1.usage.prompt_tokens
            msgs = msgs + [{"role": "assistant", "content": f"{reason}\n답:"}]
        r = _req(msgs, 1)
    except Exception as e:
        return {"error": str(e)[:200]}
    lp = r.choices[0].logprobs
    if not lp or not lp.content:
        return {"content": r.choices[0].message.content, "parse_fail": True}
    p, beyond, floor = read_dist(lp.content[0].top_logprobs, label, "")
    return {"reason": reason, "content": r.choices[0].message.content, "p": p, "beyond_top20": beyond,
            "floor": floor, "usage": r.usage.prompt_tokens + tok_in, "provider": getattr(r, "provider", None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="data/personas_150.parquet")
    ap.add_argument("--items", default="data/items_pilot.json")
    ap.add_argument("--modes", default="direct,cot"); ap.add_argument("--orders", default="orig,rev")
    ap.add_argument("--item_ids", default=None); ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--condition", default="A"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [i for t in json.load(open(a.items, encoding="utf-8"))["turns"] for i in t["items"]]
    if a.item_ids: items = [i for i in items if i["id"] in a.item_ids.split(",")]
    personas = pd.read_parquet(a.personas).to_dict("records")[: a.n]
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():                    # 성공 행만 남기고 오류·파싱 실패는 재실행
        keep = [l for l in out.open(encoding="utf-8") if "p" in json.loads(l)]
        out.write_text("".join(keep), encoding="utf-8")
        done = {json.loads(l)["key"] for l in keep}
    jobs = []
    for p in personas:
        sysmsg = system_message(None if a.condition == "B" else p, a.condition)
        for it in items:
            for m in a.modes.split(","):
                for o in a.orders.split(","):
                    key = f"{p['uuid']}|{a.condition}|{it['id']}|{m}|{o}"
                    if key not in done: jobs.append((key, p, it, o, m, sysmsg))

    def work(j):
        key, p, it, o, m, s = j
        return {"key": key, "uuid": p["uuid"], "condition": a.condition, "item": it["id"], "mode": m,
                "order": o, "sex": p.get("sex"), "age": p.get("age"),
                "marital_status": p.get("marital_status"), **call(p, it, o, m, s)}

    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        for i, row in enumerate(ex.map(work, jobs)):
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            if i % 200 == 0: print(i, "/", len(jobs), flush=True)
    print("done", len(jobs))


if __name__ == "__main__":
    main()
