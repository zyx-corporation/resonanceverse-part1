#!/usr/bin/env bash
# 審判済み JSONL 2 本（異なる hf_local モデル想定）の軸別一致を JSON にまとめる。
# 本体は v7_llm_judge_slm_pair_agreement.py（プラン v7_local_slm_m3_japanese_plan.md §10）。
#
# 環境変数:
#   JSONL_A  既定: experiments/logs/judge_slm_a.jsonl
#   JSONL_B  既定: experiments/logs/judge_slm_b.jsonl
#   OUT_JSON 既定: experiments/logs/judge_slm_pair_agreement.json
#   OUT_MD   省略可: 指定時は --out-md も渡す
#
# 例:
#   JSONL_A=logs/swallow.jsonl JSONL_B=logs/qwen.jsonl OUT_JSON=logs/agree.json \\
#     bash experiments/run_local_slm_judge_pair_agreement.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JSONL_A="${JSONL_A:-experiments/logs/judge_slm_a.jsonl}"
JSONL_B="${JSONL_B:-experiments/logs/judge_slm_b.jsonl}"
OUT_JSON="${OUT_JSON:-experiments/logs/judge_slm_pair_agreement.json}"
OUT_MD_ARG=()
if [[ -n "${OUT_MD:-}" ]]; then
  OUT_MD_ARG=(--out-md "$OUT_MD")
fi

mkdir -p "$(dirname "$OUT_JSON")"
if [[ -n "${OUT_MD:-}" ]]; then
  mkdir -p "$(dirname "$OUT_MD")"
fi

python3 experiments/v7_llm_judge_slm_pair_agreement.py \
  --jsonl-a "$JSONL_A" \
  --jsonl-b "$JSONL_B" \
  --out-json "$OUT_JSON" \
  "${OUT_MD_ARG[@]}"
