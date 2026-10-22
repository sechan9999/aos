"""Step 2. NPK 리비전 ada0f5b5에서 150명 페르소나를 불러와 parquet으로 저장.

사용: python src/load_personas.py data/sample_150.uuids.txt data/personas_150.parquet
HF 접근이 필요함 (huggingface.co). 전체 100만 행을 받지 않도록 parquet 샤드를 열 단위로 스트리밍.
"""
import sys, pathlib
from datasets import load_dataset

REVISION = "ada0f5b53a38bb5a30cce09358adde883c1ab63a"
uuids = set(pathlib.Path(sys.argv[1]).read_text().split("\n"))

ds = load_dataset("nvidia/Nemotron-Personas-Korea", revision=REVISION,
                  split="train", streaming=True)
rows = [r for r in ds if r["uuid"] in uuids]          # 1 pass, 메모리 = 150행
assert len(rows) == len(uuids), f"{len(rows)} / {len(uuids)} found"

import pandas as pd
df = pd.DataFrame(rows).sort_values("uuid").reset_index(drop=True)
assert df.shape[1] == 26, f"expected 26 cols (uuid + 25), got {df.shape[1]}"
df.to_parquet(sys.argv[2], index=False)
print(df.shape, df["age"].describe()[["min", "max"]].to_dict(), df["sex"].value_counts().to_dict())
