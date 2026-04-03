#!/usr/bin/env bash
# Phase 4（探索）: 同一 JSONL スライスを 2 つの hf_local 審判モデルで走らせ、
# v7_llm_judge_slm_pair_agreement で JSON+MD を出す。初回は各モデルの HF 取得が重い。
#
# 環境変数:
#   SRC, OFFSET, MAX_ROWS, SEED, TEMPERATURE
#   HF_MODEL_A  既定: tokyotech-llm/Swallow-7b-instruct-hf
#   HF_MODEL_B  既定: Qwen/Qwen2.5-7B-Instruct
#   OUT_A, OUT_B, OUT_JSON, OUT_MD（出力パス）
#   CPU  1 なら --cpu
#   HF_REVISION_A / HF_REVISION_B  省略可
#
# 例:
#   MAX_ROWS=5 bash experiments/run_local_slm_phase4_judge_pair.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SRC="${SRC:-experiments/data/v7_mrmp_sample.jsonl}"
OFFSET="${OFFSET:-0}"
MAX_ROWS="${MAX_ROWS:-3}"
SEED="${SEED:-0}"
TEMPERATURE="${TEMPERATURE:-0}"
HF_MODEL_A="${HF_MODEL_A:-tokyotech-llm/Swallow-7b-instruct-hf}"
HF_MODEL_B="${HF_MODEL_B:-Qwen/Qwen2.5-7B-Instruct}"
OUT_A="${OUT_A:-experiments/logs/phase4_judge_hf_a.jsonl}"
OUT_B="${OUT_B:-experiments/logs/phase4_judge_hf_b.jsonl}"
OUT_JSON="${OUT_JSON:-experiments/logs/phase4_slm_pair_agreement.json}"
OUT_MD="${OUT_MD:-experiments/logs/phase4_slm_pair_agreement.md}"

CPU_FLAG=()
if [[ "${CPU:-0}" == "1" ]]; then
  CPU_FLAG=(--cpu)
fi
REV_A=()
if [[ -n "${HF_REVISION_A:-}" ]]; then
  REV_A=(--hf-revision "$HF_REVISION_A")
fi
REV_B=()
if [[ -n "${HF_REVISION_B:-}" ]]; then
  REV_B=(--hf-revision "$HF_REVISION_B")
fi

mkdir -p "$(dirname "$OUT_A")" "$(dirname "$OUT_B")" "$(dirname "$OUT_JSON")"

echo "=== Phase4 judge model A: $HF_MODEL_A ==="
PYTHONUNBUFFERED=1 python3 experiments/v7_phase1a_llm_judge_six_axes.py \
  --provider hf_local \
  --hf-model "$HF_MODEL_A" \
  "${REV_A[@]}" \
  --jsonl "$SRC" \
  --offset "$OFFSET" \
  --max-rows "$MAX_ROWS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  "${CPU_FLAG[@]}" \
  --out-jsonl "$OUT_A" \
  --out-summary "${OUT_A%.jsonl}_summary.json"

echo "=== Phase4 judge model B: $HF_MODEL_B ==="
PYTHONUNBUFFERED=1 python3 experiments/v7_phase1a_llm_judge_six_axes.py \
  --provider hf_local \
  --hf-model "$HF_MODEL_B" \
  "${REV_B[@]}" \
  --jsonl "$SRC" \
  --offset "$OFFSET" \
  --max-rows "$MAX_ROWS" \
  --temperature "$TEMPERATURE" \
  --seed "$SEED" \
  "${CPU_FLAG[@]}" \
  --out-jsonl "$OUT_B" \
  --out-summary "${OUT_B%.jsonl}_summary.json"

echo "=== Phase4 SLM pair agreement ==="
python3 experiments/v7_llm_judge_slm_pair_agreement.py \
  --jsonl-a "$OUT_A" \
  --jsonl-b "$OUT_B" \
  --out-json "$OUT_JSON" \
  --out-md "$OUT_MD"

echo "=== done ==="
echo "  $OUT_A"
echo "  $OUT_B"
echo "  $OUT_JSON"
echo "  $OUT_MD"
