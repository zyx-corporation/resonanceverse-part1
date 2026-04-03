#!/usr/bin/env bash
# Phase 2 スモーク: Swallow-7B-Instruct（hf_local）で MRMP サンプル先頭 1 行に 6 軸審判を付与。
# 初回は HF から ~14GB 級の取得あり。M3 Max + MPS 想定。
#
# 例:
#   bash experiments/run_local_slm_phase2_smoke.sh
#   CPU=1 bash experiments/run_local_slm_phase2_smoke.sh
#
# 環境変数:
#   JSONL   既定: experiments/data/v7_mrmp_sample.jsonl
#   HF_MODEL 既定: tokyotech-llm/Swallow-7b-instruct-hf
#   OUT_JSONL / OUT_SUMMARY  出力パス
#   CPU     1 なら --cpu

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JSONL="${JSONL:-experiments/data/v7_mrmp_sample.jsonl}"
HF_MODEL="${HF_MODEL:-tokyotech-llm/Swallow-7b-instruct-hf}"
OUT_JSONL="${OUT_JSONL:-experiments/logs/local_slm_phase2_hflocal_judge.jsonl}"
OUT_SUMMARY="${OUT_SUMMARY:-experiments/logs/local_slm_phase2_hflocal_judge_summary.json}"

CPU_FLAG=()
if [[ "${CPU:-0}" == "1" ]]; then
  CPU_FLAG=(--cpu)
fi

mkdir -p "$(dirname "$OUT_JSONL")" "$(dirname "$OUT_SUMMARY")"

echo "=== v7_local_env_check ==="
python3 experiments/v7_local_env_check.py

echo "=== Phase 2 hf_local judge (1 row) ==="
PYTHONUNBUFFERED=1 python3 experiments/v7_phase1a_llm_judge_six_axes.py \
  --provider hf_local \
  --hf-model "$HF_MODEL" \
  --jsonl "$JSONL" \
  --out-jsonl "$OUT_JSONL" \
  --out-summary "$OUT_SUMMARY" \
  --max-rows 1 \
  --temperature 0 \
  --hf-max-new-tokens 256 \
  --seed 0 \
  "${CPU_FLAG[@]}"

echo "=== done ==="
echo "    jsonl: $OUT_JSONL"
echo "    summary: $OUT_SUMMARY"
