"""Step 5. 설문 대화 실행기.

조건 축
  condition : A (논문 재현) | B (페르소나 없음) | C (속성 주입, inject=4종)
  wording   : orig (논문) | refuse (보기에 '응답하지 않음' 추가) | open (보기 없이 자유 응답)
  temperature/reps : A 재현은 T=0·1회, 비판 1은 T=0.8·3회

대화 구조 (논문과 동일): system 1개 → 주제별 user 턴, 매 턴 모델 JSON 원문을 이력에 누적.
턴마다 문항별 {reason(한국어 한 문장 ≤50자), answer(보기 번호)} 을 JSON으로 받음.

백엔드: OpenAI 호환 /chat/completions (Azure AI Foundry · vLLM · Ollama · OpenRouter 공통) 또는 mock.
  환경변수 LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
"""
from __future__ import annotations
import json, os, random, time, pathlib, argparse
import pandas as pd
from prompts import system_message

REFUSE = "응답하지 않음"


# ---------- 문항 → 프롬프트 · 스키마 ----------
def render_turn(turn: dict, wording: str) -> tuple[str, dict]:
    lines, props = [f"[{turn['topic']}]"], {}
    for it in turn["items"]:
        opts = dict(it["options"])
        if wording == "refuse":
            opts[str(len(opts) + 1)] = REFUSE
        if wording == "open":
            lines.append(f"- {it['id']}: {it['text']} (자유롭게 답하세요)")
            ans = {"type": "string"}
        else:
            o = " / ".join(f"{k}={v}" for k, v in opts.items())
            lines.append(f"- {it['id']}: {it['text']} ({o})")
            ans = {"type": "string", "enum": list(opts)}
        props[it["id"]] = {"type": "object", "additionalProperties": False,
                           "required": ["reason", "answer"],
                           "properties": {"reason": {"type": "string"}, "answer": ans}}
    lines.append("\n각 문항마다 먼저 해당 인물로서 응답 이유를 한국어 한 문장(50자 이내)으로 쓰고, "
                 + ("답을 자유롭게 쓰세요." if wording == "open" else "보기 번호를 고르세요.")
                 + " 문항 id를 키로 하는 JSON으로만 답하세요.")
    schema = {"type": "object", "additionalProperties": False,
              "required": list(props), "properties": props}
    return "\n".join(lines), schema


def applicable(turn: dict, persona: dict | None) -> bool:
    rule = turn.get("skip_if")
    if not rule or persona is None:          # B 조건은 혼인상태 정보 없음 → 묻고 사후 처리
        return True
    return persona.get("marital_status") == rule["marital_status_not"]


# ---------- 백엔드 ----------
class OpenAICompat:
    def __init__(self, model=None):
        from openai import OpenAI
        self.c = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ.get("LLM_API_KEY", "x"))
        self.model = model or os.environ["LLM_MODEL"]

    def __call__(self, messages, schema, temperature, seed):
        for attempt in range(4):
            try:
                extra = {}
                if os.environ.get("LLM_PROVIDER"):        # OpenRouter: 제공업체 고정 (재현성)
                    extra["provider"] = {"order": [os.environ["LLM_PROVIDER"]],
                                         "allow_fallbacks": False, "require_parameters": True}
                r = self.c.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature, seed=seed, max_tokens=1500,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "survey", "strict": True, "schema": schema}},
                    extra_body=extra or None)
                self.last_provider = getattr(r, "provider", None)
                return r.choices[0].message.content
            except Exception as e:
                if attempt == 3: raise
                time.sleep(2 ** attempt)


class Mock:
    """오프라인 파이프라인 점검용. 일부러 규칙 위반(영어·장문·비JSON)을 섞어 검증기를 시험."""
    model = "mock"
    def __call__(self, messages, schema, temperature, seed):
        rng = random.Random(f"{seed}-{len(messages)}")
        if rng.random() < 0.05: return "Sure! Here is my answer: 2"
        out = {}
        for k, spec in schema["properties"].items():
            enum = spec["properties"]["answer"].get("enum")
            reason = rng.choice(["평소 건강하게 지내고 있어서 그렇게 답합니다.",
                                 "I am fine.", "가족과 사이가 좋고 별다른 일이 없었습니다." * 3])
            out[k] = {"reason": reason, "answer": rng.choice(enum) if enum else "없습니다"}
        return json.dumps(out, ensure_ascii=False)


# ---------- 한 페르소나 × 한 조건 대화 ----------
def run_conversation(backend, persona, items, condition, wording, temperature, seed, inject=None):
    sys_msg = system_message(persona, condition, inject)
    messages, rows = [{"role": "system", "content": sys_msg}], []
    for turn in items["turns"]:
        if not applicable(turn, persona):
            continue
        prompt, schema = render_turn(turn, wording)
        messages.append({"role": "user", "content": prompt})
        raw = backend(messages, schema, temperature, seed)
        messages.append({"role": "assistant", "content": raw})   # 원문 누적 (논문과 동일)
        rows.append({"turn": turn["turn"], "raw": raw, "item_ids": [i["id"] for i in turn["items"]]})
    return {"system": sys_msg, "turns": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", default="data/personas_150.parquet")
    ap.add_argument("--items", default="data/items_pilot.json")
    ap.add_argument("--condition", choices=["A", "B", "C"], default="A")
    ap.add_argument("--inject", choices=["disability", "lgb", "transgender", "foreign_born"])
    ap.add_argument("--wording", choices=["orig", "refuse", "open"], default="orig")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--n", type=int, default=None, help="앞에서 n명만 (smoke test)")
    ap.add_argument("--backend", choices=["openai", "mock"], default="openai")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = json.load(open(a.items, encoding="utf-8"))
    backend = Mock() if a.backend == "mock" else OpenAICompat()
    personas = pd.read_parquet(a.personas).to_dict("records")[: a.n]
    if a.condition == "B":                      # 페르소나 없음: 같은 질문을 n회 (seed만 다름)
        personas = [{"uuid": f"none-{i:03d}"} for i in range(len(personas))]

    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    with out.open("a", encoding="utf-8") as f:
        for p in personas:
            for r in range(a.reps):
                key = f"{p['uuid']}|{a.condition}|{a.inject}|{a.wording}|{a.temperature}|{r}"
                if key in done: continue           # 재시작 가능
                conv = run_conversation(backend, None if a.condition == "B" else p, items,
                                        a.condition, a.wording, a.temperature, 20260727 + r, a.inject)
                f.write(json.dumps({"key": key, "uuid": p["uuid"], "model": backend.model,
                                    "condition": a.condition, "inject": a.inject, "wording": a.wording,
                                    "temperature": a.temperature, "rep": r,
                                    "marital_status": p.get("marital_status"), **conv},
                                   ensure_ascii=False) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
