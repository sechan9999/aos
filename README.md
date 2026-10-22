# aos — 「침묵의 설계자」 재현과 검증

> Kim, Paeng, Kim. *Architect of Silence: Persona-prompted Large Language Models Make Minority Population and Social Suffering Invisible.* 서울대학교 보건대학원.

논문의 결과를 독립 재현하고, 그 결과가 모델의 작동 방식에서 나오는지 아니면 설문 설계에서 나오는지 가르기 위한 검증 실험 모음입니다. 
논문 한국어 발표 자료 Download: https://oeh.snu.ac.kr/ai-fairness

**종합 보고서 → [`docs/00_replication_report.md`](docs/00_replication_report.md)**

## 한 줄 결론

논문의 핵심 주장은 재현되었고 네 가지 검증을 통과했습니다. 다만 논문이 제시한 처방은 절반만 작동합니다. 자세한 근거는 종합 보고서에 있습니다.

## 실험 구성

| 문서 | 내용 | n |
|---|---|---|
| [`docs/01_reproduction_100.md`](docs/01_reproduction_100.md) | 논문 조건 그대로 재현 | 100명 × 12문항 |
| [`docs/02_counterfactual_age.md`](docs/02_counterfactual_age.md) | 나이만 교체 + 서술형 필드 제거 | 15명 × 2집단 × 4조건 |
| [`docs/03_injection_spillover.md`](docs/03_injection_spillover.md) | 장애 속성 주입과 고통 파급 | 20명 × 3조건 |
| [`docs/04_dose_response.md`](docs/04_dose_response.md) | 주입 강도 용량-반응 재실험 | 20명 × 3조건 |
| [`docs/05_option_order.md`](docs/05_option_order.md) | 보기 순서 뒤집기 | 20명 × 2조건 × 2순서 |

로그확률 감사(`src/logprob_audit.py`)는 코드만 있고 아직 돌리지 않았습니다.

> **별도 트랙**: [`muse-exp/`](muse-exp/)에는 Muse 에이전트가 독립 실행한 로그확률 감사(Qwen2.5-1.5B)와 120B 샘플링 재현(gpt-oss-120b, 진행 중)이 있습니다. 표본과 문항이 이 트랙과 다르니 비교 시 그 폴더의 README를 먼저 보세요.
>
> [`gemini-exp/`](gemini-exp/)에는 Gemini가 실행한 트랙의 응답 원자료(gemini-2.5-flash 1,650 실행 + qwen2.5-coder 부분 실행)가 있습니다.

## 데이터

- 페르소나: [Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea), 리비전 `ada0f5b5`, train split — 논문과 동일
- 표본: 논문 공개 1k 하위 표본에서 seed 20261019로 150명 추출
  - `data/sample_150.uuids.txt` SHA-256 `ef98881b0c19f96c605b223d0dabf76782efb54675d97246abe05a472c651704`
  - 1k 하위 표본에서 뽑았으므로 논문의 4개 모델 응답과 uuid로 짝지을 수 있음

**페르소나 원자료는 이 저장소에 넣지 않았습니다.** 논문 저장소와 같은 방식으로 uuid 목록만 두고, 아래 명령으로 각자 내려받습니다.

## 재현 방법

```bash
pip install datasets huggingface_hub openai pandas pyarrow scipy

# 1. uuid 목록으로 페르소나 내려받기 (Hugging Face 접근 필요)
python src/load_personas.py data/sample_150.uuids.txt data/personas_150.parquet

# 2. 설문 실행 — OpenAI 호환 엔드포인트면 무엇이든 가능
export LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...
python src/run.py --out out/run.jsonl --n 5          # 연결 확인
python src/run.py --out out/run.jsonl                # A: 논문 재현, T=0
python src/run.py --out out/run.jsonl --condition B  # 페르소나 없음 (모델 prior)
python src/run.py --out out/run.jsonl --condition C --inject disability  # 속성 주입

# 3. 한국어 지시 이행 검증
python src/validate.py out/run.jsonl data/items_pilot.json

# 4. 보기 순서 분석
python src/analyze_m4.py
```

`run.py`는 key 기준으로 이어쓰기 때문에 중단 후 다시 실행해도 됩니다.

로그확률 감사는 `top_logprobs`를 돌려주는 제공업체가 필요합니다. 재현성을 위해 제공업체를 고정합니다.

```bash
LP_PROVIDER=Novita python src/logprob_audit.py --n 100 --out out/lp_A.jsonl
python src/analyze_lp.py out/lp_A.jsonl
```

## 디렉터리

```
src/     실행·분석 코드
data/    문항 정의, uuid 목록, 논문 system_message
out/     응답 원자료와 집계 결과
docs/    종합 보고서와 실험별 문서
```

## 알아둘 제약

- **문항이 논문 원문이 아닙니다.** 보충표 2가 아직 공개되지 않아, 발표자료에서 확인한 2문항 외에는 재구성한 초안입니다. 모든 수치를 논문과 직접 비교할 수 없습니다
- 응답자는 Claude Code 하네스 시스템 프롬프트가 함께 들어간 상태의 Opus입니다. 논문 조건과 다릅니다
- 기본 온도로 1회만 샘플링했습니다. 온도에 따른 분포는 확인하지 않았습니다
- 표본이 작습니다. 반사실 집단당 15명, 주입·순서 실험 20명입니다

## 출처와 라이선스

- 원 논문 저장소 [`oeh-snu/architect-of-silence-llm`](https://github.com/oeh-snu/architect-of-silence-llm) (CC BY 4.0). `data/system_message.txt`와 `data/npk_sample_1k.uuids.txt`는 그 저장소에서 가져왔습니다
- 이 저장소의 코드와 문서는 CC BY 4.0을 따릅니다
