"""모듈 4. 보기 순서 민감도 (option-order flip).

설계: 같은 페르소나 20명 × 조건 2종(C0b 주입없음 / C1b 명시라벨) × 보기순서 2종(orig/rev)
      → 사람 안에서 짝지은 40쌍. (C2b는 orig만 돌려서 순서 분석에서 빠짐)

보기 표시
  orig : 1=예 / 2=아니오,  self_health 1=매우 좋음 … 5=매우 나쁨
  rev  : 1=아니오 / 2=예,  self_health 1=매우 나쁨 … 5=매우 좋음
→ 둘 다 '정답 의미' 기준(canonical)으로 되돌린 뒤 비교한다.

unmet_reason 은 분석에서 제외: 문두에 '해당 없으면 5'가 박혀 있는데 보기만 뒤집혀서
rev 조건에서 문두 번호(5)와 라벨(5=경제적 부담)이 어긋난다. 설계 결함이므로 뺀다.
(독립적으로 두 subagent가 같은 문제를 보고함)
"""
import json, pathlib, sys
import pandas as pd
from scipy import stats

S = pathlib.Path("/tmp/claude-0/-home-claude/806e9892-fea6-59ba-9044-5dcbedf32527/scratchpad/s4")
BINARY = ["disab_reg", "disab_self", "unmet_care", "activity_limit",
          "discrim", "social_support", "depress_2wk", "suic_idea"]
EXCLUDE = {"unmet_reason"}           # 문두/라벨 불일치 (위 참조)


def canon(item, ans, order):
    """표시 번호 → 의미. 이진은 True=예, self_health는 1=매우 좋음 … 5=매우 나쁨."""
    a = str(ans).strip()
    if item in BINARY:
        return {"orig": {"1": True, "2": False}, "rev": {"1": False, "2": True}}[order].get(a)
    if item == "self_health":
        if a not in list("12345"): return None
        return int(a) if order == "orig" else 6 - int(a)
    return None


def load():
    idx = pd.read_csv("out/m4_index.csv")
    rows = []
    for r in idx.itertuples():
        p = S / f"{r.f}.json"
        if not p.exists(): continue
        txt = p.read_text(encoding="utf-8")
        txt = txt[txt.find("{"): txt.rfind("}") + 1]          # 앞뒤 설명문 방어
        obj = json.loads(txt)
        rec = {"uuid": r.uuid, "cond": r.cond, "order": r.order, "age": r.age, "f": r.f}
        for item in BINARY + ["self_health"]:
            v = obj.get(item)
            rec[item] = canon(item, v.get("answer") if isinstance(v, dict) else v, r.order)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    d = load()
    d.to_csv("out/m4_canonical.csv", index=False, encoding="utf-8-sig")
    print(f"읽은 응답 {len(d)}건", d.groupby(['cond', 'order']).size().to_dict(), "\n")

    pair = d[d.cond.isin(["C0b", "C1b"])]
    w = pair.pivot_table(index=["uuid", "cond"], columns="order",
                         values=BINARY + ["self_health"], aggfunc="first").dropna()
    print(f"짝지은 쌍 {len(w)}개 (C0b 20 + C1b 20)\n")

    # --- 이진 문항: 같은 사람 orig vs rev 일치율 + McNemar(정확) ---
    out = []
    for cond in ["C0b", "C1b", "전체"]:
        sub = w if cond == "전체" else w.xs(cond, level="cond")
        for it in BINARY:
            o, r = sub[(it, "orig")].astype(bool), sub[(it, "rev")].astype(bool)
            b, c = int((o & ~r).sum()), int((~o & r).sum())      # orig만 예 / rev만 예
            p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            out.append({"cond": cond, "item": it, "n": len(sub),
                        "orig_예": int(o.sum()), "rev_예": int(r.sum()),
                        "불일치": b + c, "일치율%": round((1 - (b + c) / len(sub)) * 100, 1),
                        "McNemar_p": round(p, 4)})
    t = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print("[이진 문항] 보기 순서만 뒤집었을 때 — 같은 사람 안에서의 응답 일치")
    print(t.to_string(index=False), "\n")

    # --- self_health: 평균 차 + Wilcoxon ---
    print("[self_health] 1=매우 좋음 … 5=매우 나쁨 (의미 기준으로 되돌림)")
    for cond in ["C0b", "C1b", "전체"]:
        sub = w if cond == "전체" else w.xs(cond, level="cond")
        o, r = sub[("self_health", "orig")], sub[("self_health", "rev")]
        diff = (r - o)
        pw = stats.wilcoxon(o, r).pvalue if diff.abs().sum() else 1.0
        print(f"  {cond:4s} n={len(sub):2d}  orig 평균 {o.mean():.2f}  rev 평균 {r.mean():.2f} "
              f" 차이 {diff.mean():+.2f}  |차이|>0 인 사람 {int((diff != 0).sum())}명  Wilcoxon p={pw:.4f}")

    # --- 라벨 유지(disab_reg)가 순서에 흔들리는지: C1b 핵심 ---
    c1 = w.xs("C1b", level="cond")
    print(f"\n[C1b 라벨 유지] disab_reg '예'  orig {int(c1[('disab_reg','orig')].sum())}/20  "
          f"rev {int(c1[('disab_reg','rev')].sum())}/20")
    t.to_csv("out/m4_order_flip.csv", index=False, encoding="utf-8-sig")
    print("\n제외: unmet_reason (문두 '해당 없으면 5'와 뒤집힌 라벨 불일치)")


if __name__ == "__main__":
    main()
