#!/usr/bin/env bash
# Phase 4（探索）: Swallow-7B-Instruct と Swallow-13B-Instruct の審判を同一スライスで比較。
# 実体は run_local_slm_phase4_judge_pair.sh（HF 取得・推論は重い）。
#
# 環境変数は run_local_slm_phase4_judge_pair.sh と同じ（SRC / OFFSET / MAX_ROWS / CPU 等）。
# 既定で OUT_* を 7b_13b 用に分離。
#
# 例:
#   MAX_ROWS=5 CPU=1 bash experiments/run_local_slm_phase4_swallow_7b_13b.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export HF_MODEL_A="${HF_MODEL_A:-tokyotech-llm/Swallow-7b-instruct-hf}"
export HF_MODEL_B="${HF_MODEL_B:-tokyotech-llm/Swallow-13b-instruct-hf}"
export OUT_A="${OUT_A:-experiments/logs/phase4_swallow7b_judge.jsonl}"
export OUT_B="${OUT_B:-experiments/logs/phase4_swallow13b_judge.jsonl}"
export OUT_JSON="${OUT_JSON:-experiments/logs/phase4_swallow_7b_13b_agreement.json}"
export OUT_MD="${OUT_MD:-experiments/logs/phase4_swallow_7b_13b_agreement.md}"

exec bash experiments/run_local_slm_phase4_judge_pair.sh
