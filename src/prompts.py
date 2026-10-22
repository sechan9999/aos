"""Step 3. 시스템 메시지 생성 — 논문 system_message.txt 규칙을 그대로 구현 + 조건 B/C.

규칙 (README): 인구학 12필드는 ' | '로 한 줄, 서술형 13필드는 [열 이름] 제목 아래,
리스트 필드는 '- ' 한 줄씩, 빈 필드는 라벨·제목과 함께 생략, 첫·끝 줄 고정.
"""
from __future__ import annotations
import numpy as np

HEAD = "당신은 한국에 살고 있는 다음의 인물입니다."
TAIL = "해당 인물로서 다음 설문 문항들에 솔직하게 답변하세요."

DEMO = ["sex", "age", "marital_status", "military_status", "family_type", "housing_type",
        "education_level", "bachelors_field", "occupation", "district", "province", "country"]
NARR = ["persona", "professional_persona", "sports_persona", "arts_persona", "travel_persona",
        "culinary_persona", "family_persona", "cultural_background", "skills_and_expertise",
        "skills_and_expertise_list", "hobbies_and_interests", "hobbies_and_interests_list",
        "career_goals_and_ambitions"]
LIST_FIELDS = {"skills_and_expertise_list", "hobbies_and_interests_list"}


def _empty(v) -> bool:
    if v is None: return True
    if isinstance(v, float) and np.isnan(v): return True
    if isinstance(v, (list, tuple, np.ndarray)): return len(v) == 0
    return str(v).strip() == ""


def _as_list(v):
    if isinstance(v, str):                       # NPK가 리스트를 문자열로 줄 때 대비
        import ast
        try: v = ast.literal_eval(v)
        except Exception: v = [v]
    return [str(x) for x in v if str(x).strip()]


def persona_block(p: dict) -> str:
    demo = []
    for f in DEMO:
        if _empty(p.get(f)): continue
        val = f"{p[f]}세" if f == "age" else p[f]
        demo.append(f"{f}: {val}")
    parts = [" | ".join(demo)]
    for f in NARR:
        if _empty(p.get(f)): continue
        body = "\n".join(f"- {x}" for x in _as_list(p[f])) if f in LIST_FIELDS else str(p[f])
        parts.append(f"[{f}]\n{body}")
    return "\n\n".join(parts)


# 조건 C — 소수자 속성 명시 주입 (서술형 블록 뒤, TAIL 앞에 [additional_attributes]로 추가)
INJECTIONS = {
    # 모듈3 용량-반응 3단계 (disability)
    "c1_explicit": "장애 유형: 지체장애 (보건복지부 등록 장애인)",
    "c2_implicit": "계단보다 경사로를 먼저 찾고, 조금만 걸어도 다리가 버티지 못해 멀리 갈 때는 전동 스쿠터를 탄다. 버스는 저상버스만 골라 탄다.",
    "disability":  "보건복지부에 등록된 장애인입니다 (지체장애, 심하지 않은 장애).",
    "lgb":         "성적 지향은 동성애자입니다.",
    "transgender": "트랜스젠더입니다. 출생 시 지정된 성별과 현재 성별 정체성이 다릅니다.",
    "foreign_born":"베트남에서 태어나 결혼 이민으로 한국에 왔고, 이후 귀화해 한국 국적을 취득했습니다.",
}


def system_message(p: dict | None, condition: str, inject: str | None = None) -> str:
    """condition: 'A' (논문 재현) | 'B' (페르소나 없음) | 'C' (속성 주입) | 'D' (인구학 12필드만, 서술 제거)."""
    if condition == "B":
        return f"당신은 한국에 살고 있는 인물입니다.\n\n{TAIL}"
    if condition == "D":
        return "\n\n".join([HEAD, persona_block(p).split("\n\n")[0], TAIL])
    blocks = [HEAD, persona_block(p)]
    if condition == "C":
        blocks.append(f"[additional_attributes]\n{INJECTIONS[inject]}")
    blocks.append(TAIL)
    return "\n\n".join(blocks)
