"""모듈 1 분석. 문항 × mode × order 별
  greedy_n   : argmax가 대상 보기인 사람 수 (= 논문식 T=0 집계)
  E_T1       : 사람별 p(대상) 평균 = T=1 샘플링 기대 유병률 (top-20 밖은 0으로 둔 하한)
  E_T1_ub    : top-20 밖 보기를 20위 확률로 채운 상한
  E_T08      : p^(1/0.8) 재정규화 후 평균 = T=0.8 기대 유병률
  max_p      : 150명 중 가장 높은 p(대상)
기준값: 발표자료 실제 한국인 조유병률 (성·연령 표준화 전, 참고용)
"""
import json, sys
import numpy as np, pandas as pd

TARGET = {"disab_reg": {"1"}, "disab_self": {"1"}, "birth_ctry": {"2"}, "sex_orient": {"2", "3"},
          "transgender": {"1"}, "child_divorce": {"1"}, "child_dropout": {"1"}, "ipv_assault": {"1"},
          "suic_idea": {"1"}, "suic_attempt": {"1"}, "smoke_now": {"1", "2"}, "htn_dx": {"1"}}
HUMAN = {"disab_reg": 5.9, "disab_self": None, "birth_ctry": 0.5, "sex_orient": None, "transgender": None,
         "child_divorce": 5.2, "child_dropout": 9.2, "ipv_assault": 0.6, "suic_idea": 4.5,
         "suic_attempt": 0.6, "smoke_now": 19.4, "htn_dx": 24.1}


def temper(p: dict, T: float):
    w = {k: v ** (1 / T) for k, v in p.items()}
    s = sum(w.values()) or 1
    return {k: v / s for k, v in w.items()}


def summarize(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    recs = []
    for r in rows:
        if "p" not in r: continue
        if r["item"] == "ipv_assault" and r.get("marital_status") not in (None, "배우자있음"): continue
        tgt, raw = TARGET[r["item"]], r["p"]
        tot = sum(raw.values())
        if tot == 0: continue
        p = {k: v / tot for k, v in raw.items()}          # 숫자 아닌 토큰 질량 제외, 보기 간 재정규화
        pt = sum(p[k] for k in tgt)
        ub = pt + sum(r["floor"] / tot for k in tgt if r["beyond_top20"][k])
        recs.append({"item": r["item"], "mode": r["mode"], "order": r["order"], "uuid": r["uuid"],
                     "p": pt, "p_ub": min(ub, 1), "p08": sum(temper(p, 0.8)[k] for k in tgt),
                     "greedy": max(p, key=p.get) in tgt, "mass": tot})
    d = pd.DataFrame(recs)
    g = d.groupby(["item", "mode", "order"]).agg(
        n=("p", "size"), greedy_n=("greedy", "sum"), E_T1=("p", "mean"), E_T1_ub=("p_ub", "mean"),
        E_T08=("p08", "mean"), max_p=("p", "max"), digit_mass=("mass", "mean")).reset_index()
    for c in ["E_T1", "E_T1_ub", "E_T08", "max_p"]: g[c] = (g[c] * 100).round(3)
    g["human_%"] = g["item"].map(HUMAN)
    order = list(TARGET)
    g["item"] = pd.Categorical(g["item"], order); g = g.sort_values(["item", "mode", "order"])
    # 보기 순서 민감도: 같은 사람 orig vs rev 의 p 차이
    w = d.pivot_table(index=["item", "mode", "uuid"], columns="order", values="p").dropna()
    flip = (w.assign(diff=(w["rev"] - w["orig"]).abs()).groupby(["item", "mode"], observed=True)["diff"].mean() * 100).round(3)
    return g, flip, d


if __name__ == "__main__":
    g, flip, d = summarize(sys.argv[1])
    pd.set_option("display.width", 200)
    print(g.to_string(index=False))
    print("\n[보기 순서 민감도] 같은 사람 orig vs rev |Δp| 평균 (%p)\n", flip.unstack().to_string())
    g.to_csv(sys.argv[1].replace(".jsonl", "_summary.csv"), index=False, encoding="utf-8-sig")
