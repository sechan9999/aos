# Architect of Silence — 재현·비판 실험 (파일럿 150명)

## 표본
- 논문 저장소 oeh-snu/architect-of-silence-llm @087975f 의 1k 하위 표본(SHA-256 검증 완료)에서 150명 비복원 추출
- seed 20261019, `data/sample_150.uuids.txt` SHA-256 `ef98881b0c19f96c605b223d0dabf76782efb54675d97246abe05a472c651704`
- 1k 하위 표본에서 뽑았으므로 논문 4개 모델 응답과 uuid로 짝지을 수 있음

## 실행 순서
```bash
pip install datasets huggingface_hub openai jsonschema pyarrow pandas
python src/load_personas.py data/sample_150.uuids.txt data/personas_150.parquet   # HF 접근 필요
export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...                             # OpenAI 호환 엔드포인트
L=out/llama33.jsonl
python src/run.py --out $L --n 5                                   # smoke test
python src/run.py --out $L                                         # A: 논문 재현 T=0
python src/run.py --out $L --temperature 0.8 --reps 3              # 비판1: 온도
python src/run.py --out $L --wording refuse                        # 비판1: 거부 보기
python src/run.py --out $L --wording open                          # 비판1: 개방형
python src/run.py --out $L --condition B                           # 비판2: 페르소나 없음
for i in disability lgb transgender foreign_born; do python src/run.py --out $L --condition C --inject $i; done  # 비판3
python src/validate.py $L data/items_pilot.json
```
`run.py`는 key 기준 이어쓰기라 중단 후 다시 실행해도 됨.

## 주의
- `items_pilot.json` 12문항 중 원문 확인은 2문항뿐(verbatim=true). 논문 31문항은 Supp. Table 2 확보 후 교체
- B 조건은 혼인 상태가 없어 배우자 폭력 문항을 모두에게 물음 → 분석 시 별도 처리
