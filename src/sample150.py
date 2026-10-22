"""Step 1. 논문 1k 하위 표본에서 150명 비복원 단순무작위 추출 (seed 고정, 해시 기록)."""
import hashlib, random, sys, pathlib
REPO = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
EXPECT = {"npk_sample_10k.uuids.txt": "9463e5198c8cdbbe2828492af3decb43ea66d79bddb1ecae0369044f93477b96",
          "npk_sample_1k.uuids.txt":  "8340706122d1cd8b6727146d9a944778f6268c4615f712f6e077137a7bb80995"}
for f, h in EXPECT.items():
    assert hashlib.sha256((REPO/f).read_bytes()).hexdigest() == h, f"hash mismatch: {f}"
k10 = set((REPO/"npk_sample_10k.uuids.txt").read_text().split("\n"))
k1  = sorted((REPO/"npk_sample_1k.uuids.txt").read_text().split("\n"))
assert len(k10) == 10_000 and len(k1) == 1_000 and set(k1) <= k10
s150 = sorted(random.Random(20261019).sample(k1, 150))     # 정렬된 1k 목록 → seed 20261019
body = "\n".join(s150)                                      # 논문과 같은 형식: LF, 끝 줄바꿈 없음
(OUT/"sample_150.uuids.txt").write_text(body, newline="\n")
print("n =", len(s150), "| sha256 =", hashlib.sha256(body.encode()).hexdigest())
