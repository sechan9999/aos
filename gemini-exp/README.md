# gemini-exp — Gemini 실행 트랙 (응답 원자료)

이 폴더는 Gemini가 실행한 「침묵의 설계자」 재현 실험의 산출물입니다. 2026-10-22, HK의 로컬 폴더(`aos-replication`)에서 `out.zip`으로 전달받아 별도 트랙으로 올렸습니다. 루트 Claude 트랙, `muse-exp/` 트랙과 독립적입니다.

## 내용물 (`out/`)

| 파일 | 내용 |
|---|---|
| `gemini25.jsonl` | 응답 원자료 — google/gemini-2.5-flash, 1,650 실행, 페르소나 300명 |
| `gemini25_validated.csv` | 문항 단위 검증 결과 — 19,180행, 12문항 |
| `qwen25.jsonl` | 응답 원자료 — qwen2.5-coder:latest (로컬), 55 실행, 조건 A만 (부분 실행) |
| `qwen25_validated.csv` | 문항 단위 검증 결과 — 57행 |

## 실행 구성 (gemini25)

- 조건: A(논문 재현) 900 · B(무페르소나) 150 · C(속성 주입) 600
  - 주입 속성: disability / lgb / transgender / foreign_born 각 150
- 온도 0.0, 문항 12개 (disab_reg, disab_self, birth_ctry, sex_orient, transgender, child_divorce, child_dropout, ipv_assault, suic_idea, suic_attempt, smoke_now, htn_dx)
- 검증 집계: 거부 13건, 형식 검증 통과(all_ok) 18,923 / 19,180행
- 스키마는 루트 `src/run.py` 출력 형식(key·uuid·condition·inject·wording·temperature·rep)과 같습니다

## 알아둘 점

- 업로드에는 **실행 코드가 포함되어 있지 않습니다.** 산출물(응답 원자료 + 검증 CSV)만 있습니다.
- jsonl에는 system 프롬프트 전문(페르소나 서술 포함)이 들어 있습니다. 루트 트랙의 "uuid 목록만 공유" 방식과 다르며, HK가 전달한 원본 그대로 올렸습니다. 페르소나는 NVIDIA Nemotron-Personas-Korea의 합성 데이터입니다.
- 표본은 페르소나 300명으로, 루트 트랙의 150명 표본과 동일하지 않습니다.
- qwen25는 조건 A 55 실행의 부분 결과입니다.
