# gemini-exp — Gemini 실행 트랙

이 폴더는 Gemini가 실행한 「침묵의 설계자」 재현·비판 실험입니다. 2026-10-22, HK의 로컬 폴더(`aos-replication`)에서 산출물(`out.zip`)과 실행 코드(`src.zip`)를 전달받아 별도 트랙으로 올렸습니다. 루트 Claude 트랙, `muse-exp/` 트랙과 독립적입니다.

실행 설계는 세 비판 축으로 구성됩니다 — 비판 1: 온도·문구(거부 보기/개방형), 비판 2: 페르소나 없음(B), 비판 3: 속성 주입(C). 조건·문구·주입을 바꾸는 하네스(`src/run.py`)는 모델 비종속(OpenAI 호환 엔드포인트)이며, 두 실행(gemini25, qwen25)이 이 코드로 돌았습니다. 자세한 실행 순서는 `RUN_README.md`(원본)와 `src/run_pipeline.py` 머리말을 보세요.

> 세 트랙의 비교 평가는 [`docs/06_cross_track_evaluation.md`](../docs/06_cross_track_evaluation.md)에 있습니다.

## 내용물

### 코드 (`src/`)
| 파일 | 내용 |
|---|---|
| `run.py` | 설문 대화 실행기 — 조건 A/B/C × 문구 orig/refuse/open × 온도·반복, key 기준 이어쓰기 |
| `run_pipeline.py` | 전체 셀 자동 실행 드라이버 (Gemini on Vertex AI) |
| `prompts.py` | 시스템 메시지 구성 (논문 저장소 규칙 구현) |
| `validate.py` | 응답 형식 검증기 (JSON·키·한국어·길이·보기·거부 판정) |
| `load_personas.py` / `sample150.py` | 페르소나 로딩·표본 추출 |

### 산출물 (`out/`)
| 파일 | 내용 |
|---|---|
| `gemini25.jsonl` | 응답 원자료 — google/gemini-2.5-flash, 1,650 실행, 페르소나 150명 |
| `gemini25_validated.csv` | 문항 단위 검증 결과 — 19,180행, 12문항 |
| `qwen25.jsonl` | 응답 원자료 — qwen2.5-coder:latest (로컬), 55 실행, 조건 A만 (부분 실행) |
| `qwen25_validated.csv` | 문항 단위 검증 결과 — 57행 |

## 실행 구성 (gemini25)

- 표본: 논문 공개 1k 하위 표본에서 seed 20261019로 추출한 150명 (SHA-256 `ef98881b…` — 루트 트랙과 동일 표본 풀이며, 루트 트랙은 앞 100명 사용)
- 조건 A(논문 재현, T=0) 150 · A 온도 변형(T=0.8, 3회 반복) 450 · A 거부 보기 150 · A 개방형 150 · B(무페르소나) 150 · C(속성 주입) 600 (disability / lgb / transgender / foreign_born 각 150)
- 문항 12개 (disab_reg, disab_self, birth_ctry, sex_orient, transgender, child_divorce, child_dropout, ipv_assault, suic_idea, suic_attempt, smoke_now, htn_dx)
- 검증 집계: 거부 13건(전부 거부 보기 조건), 형식 검증 통과(all_ok) 18,923 / 19,180행

## 알아둘 점

- `.env`(API 키)는 보안상 제외했습니다. 루트 `.gitignore`도 `.env`를 커밋 금지로 지정합니다. `__pycache__`도 제외했습니다.
- jsonl에는 system 프롬프트 전문(페르소나 서술 포함)이 들어 있습니다. 루트 트랙의 "uuid 목록만 공유" 방식과 다르며, HK가 전달한 원본 그대로 올렸습니다. 페르소나는 NVIDIA Nemotron-Personas-Korea의 합성 데이터입니다.
- qwen25는 조건 A 55 실행의 부분 결과로, 검증까지 마친 것은 문항당 약 5건입니다.
