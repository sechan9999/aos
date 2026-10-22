"""전체 재현 및 비판 실험 파이프라인 자동 실행기 (Gemini on Vertex AI).

실행 순서 (README 기반):
1. A: 논문 재현 T=0 (150명)
2. 비판 1: 온도 T=0.8, reps 3
3. 비판 1: 거부 보기 (wording refuse)
4. 비판 1: 개방형 (wording open)
5. 비판 2: 페르소나 없음 (condition B)
6. 비판 3: 속성 주입 (condition C, 4종: disability, lgb, transgender, foreign_born)
7. 검증 및 결과 표 생성 (validate.py)
"""
import os, sys, subprocess, pathlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ENV = os.environ.copy()
ENV.update({
    "LLM_BASE_URL": "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/agentichackathon-506620/locations/us-central1/endpoints/openapi",
    "LLM_API_KEY": "adc",
    "LLM_MODEL": "google/gemini-2.5-flash",
    "PYTHONIOENCODING": "utf-8"
})

OUT = "out/gemini25.jsonl"
PYTHON = sys.executable

STEPS = [
    ("Step A: 논문 재현 T=0", [PYTHON, "src/run.py", "--out", OUT]),
    ("Step B1: 비판 1 (온도 T=0.8, reps 3)", [PYTHON, "src/run.py", "--out", OUT, "--temperature", "0.8", "--reps", "3"]),
    ("Step B2: 비판 1 (거부 보기)", [PYTHON, "src/run.py", "--out", OUT, "--wording", "refuse"]),
    ("Step B3: 비판 1 (개방형)", [PYTHON, "src/run.py", "--out", OUT, "--wording", "open"]),
    ("Step B4: 비판 2 (페르소나 없음)", [PYTHON, "src/run.py", "--out", OUT, "--condition", "B"]),
    ("Step B5-1: 비판 3 (주입: disability)", [PYTHON, "src/run.py", "--out", OUT, "--condition", "C", "--inject", "disability"]),
    ("Step B5-2: 비판 3 (주입: lgb)", [PYTHON, "src/run.py", "--out", OUT, "--condition", "C", "--inject", "lgb"]),
    ("Step B5-3: 비판 3 (주입: transgender)", [PYTHON, "src/run.py", "--out", OUT, "--condition", "C", "--inject", "transgender"]),
    ("Step B5-4: 비판 3 (주입: foreign_born)", [PYTHON, "src/run.py", "--out", OUT, "--condition", "C", "--inject", "foreign_born"]),
]

def main():
    print("=" * 60)
    print("Architect of Silence - Gemini 2.5 Flash 파이프라인 가동")
    print(f"출력 경로: {OUT}")
    print("=" * 60)

    for name, cmd in STEPS:
        print(f"\n>>> 시작: {name}")
        ret = subprocess.run(cmd, env=ENV)
        if ret.returncode != 0:
            print(f"[ERROR] {name} 실패 (코드: {ret.returncode})")
            sys.exit(ret.returncode)
        print(f">>> 완료: {name}")

    print("\n" + "=" * 60)
    print("전체 실험 완료! 지시 이행 및 속성 유지율 검증 실행...")
    print("=" * 60)
    subprocess.run([PYTHON, "src/validate.py", OUT, "data/items_pilot.json"], env=ENV)

if __name__ == "__main__":
    main()
