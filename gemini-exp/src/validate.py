"""Step 6. 한국어 지시 이행 검증.

문항 단위 체크
  json_ok     턴 응답이 JSON으로 파싱됨
  keys_ok     요구한 문항 id가 모두 있음
  korean_ok   reason의 한글 비율 ≥ 0.5 (공백·숫자·구두점 제외)
  len_ok      reason ≤ 50자
  one_sent_ok reason이 한 문장 (종결부호 ≤ 1)
  option_ok   answer가 허용 보기 안 (open 조건은 비어 있지 않음)
  refused     '응답하지 않음' 선택 여부 (refuse 조건)
조건 C 추가: inject_kept — 주입한 속성에 맞는 답을 했는지
"""
import json, re, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

HANGUL = re.compile(r"[가-힣]")
LETTER = re.compile(r"[^\s\d\W_]", re.UNICODE)
INJECT_EXPECT = {"disability": ("disab_reg", "1"), "lgb": ("sex_orient", {"2", "3"}),
                 "transgender": ("transgender", "1"), "foreign_born": ("birth_ctry", "2")}


def check_reason(s: str):
    letters = LETTER.findall(s or "")
    kor = len(HANGUL.findall(s or "")) / max(len(letters), 1)
    return kor >= 0.5, len(s or "") <= 50, len(re.findall(r"[.!?。]", (s or "").rstrip(".!?。"))) == 0


def item_rows(rec, items):
    opts = {i["id"]: i["options"] for t in items["turns"] for i in t["items"]}
    for t in rec["turns"]:
        try:
            obj, jok = json.loads(t["raw"]), True
        except Exception:
            obj, jok = {}, False
        for iid in t["item_ids"]:
            a = obj.get(iid) if isinstance(obj, dict) else None
            reason = a.get("reason", "") if isinstance(a, dict) else ""
            ans = str(a.get("answer", "")) if isinstance(a, dict) else ""
            k, l, s = check_reason(reason)
            allowed = set(opts[iid]) | ({str(len(opts[iid]) + 1)} if rec["wording"] == "refuse" else set())
            yield {"key": rec["key"], "uuid": rec["uuid"], "condition": rec["condition"],
                   "inject": rec["inject"], "wording": rec["wording"], "temperature": rec["temperature"],
                   "rep": rec["rep"], "item": iid, "answer": ans, "reason": reason,
                   "json_ok": jok, "keys_ok": isinstance(a, dict), "korean_ok": k, "len_ok": l,
                   "one_sent_ok": s,
                   "option_ok": bool(ans.strip()) if rec["wording"] == "open" else ans in allowed,
                   "refused": rec["wording"] == "refuse" and ans == str(len(opts[iid]) + 1)}


def main(log_path, items_path):
    items = json.load(open(items_path, encoding="utf-8"))
    recs = [json.loads(l) for l in open(log_path, encoding="utf-8")]
    df = pd.DataFrame([r for rec in recs for r in item_rows(rec, items)])
    checks = ["json_ok", "keys_ok", "korean_ok", "len_ok", "one_sent_ok", "option_ok"]
    df["all_ok"] = df[checks].all(axis=1)
    print(f"conversations={len(recs)}  item-responses={len(df)}")
    print((df.groupby(["condition", "wording"])[checks + ["all_ok", "refused"]].mean() * 100).round(1).to_string())

    c = df[df.condition == "C"]
    if len(c):
        kept = [(inj, (c[(c.inject == inj) & (c.item == iid)].answer
                       .isin(exp if isinstance(exp, set) else {exp})).mean())
                for inj, (iid, exp) in INJECT_EXPECT.items() if (c.inject == inj).any()]
        print("\n[C] 주입 속성 유지율:", {k: round(v * 100, 1) for k, v in kept})
    df.to_csv(log_path.replace(".jsonl", "_validated.csv"), index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
