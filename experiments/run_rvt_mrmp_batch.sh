#!/usr/bin/env bash
# MRMP windows.jsonl を RVT L1 拡張でバッチ処理（モデル 1 回ロード）。
#
# 環境変数:
#   JSONL  既定: experiments/logs/mrmp_prepared/windows.jsonl
#   LINE   既定: 0（物理行 0 始まり）
#   MAX_ROWS 既定: 10
#   MODEL  既定: gpt2
#   LAYER  既定: -1
#   SEED   既定: 42
#   CPU    1 なら --cpu
#   ACCUMULATE_AWAI  1 なら --accumulate-awai
#   AWAI_OUT  例: experiments/logs/rvt_awai_accum.json（任意）
#   RVT_L2_MODE   sym / wasym（空または base で L2 なし）
#   RVT_L2_EPS    既定 0.05
#   RVT_L2_ALL_LAYERS  1 なら --rvt-l2-all-layers
#   STRICT   1 なら --strict
#   PRETTY   1 なら --pretty-array
#   OUT      バッチ JSONL の出力パス（任意・python --out）
#
# 例:
#   bash experiments/run_rvt_mrmp_batch.sh
#   LINE=100 MAX_ROWS=50 MODEL=rinna/japanese-gpt2-medium CPU=1 bash experiments/run_rvt_mrmp_batch.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JSONL="${JSONL:-experiments/logs/mrmp_prepared/windows.jsonl}"
LINE="${LINE:-0}"
MAX_ROWS="${MAX_ROWS:-10}"
MODEL="${MODEL:-gpt2}"
LAYER="${LAYER:--1}"
SEED="${SEED:-42}"
CPU_FLAG=()
if [[ "${CPU:-0}" == "1" ]]; then
  CPU_FLAG=(--cpu)
fi
AWAI_FLAGS=()
if [[ "${ACCUMULATE_AWAI:-0}" == "1" ]]; then
  AWAI_FLAGS=(--accumulate-awai)
  if [[ -n "${AWAI_OUT:-}" ]]; then
    AWAI_FLAGS+=(--awai-out "$AWAI_OUT")
  fi
fi
STRICT_FLAG=()
if [[ "${STRICT:-0}" == "1" ]]; then
  STRICT_FLAG=(--strict)
fi
PRETTY_FLAG=()
if [[ "${PRETTY:-0}" == "1" ]]; then
  PRETTY_FLAG=(--pretty-array)
fi
OUT_FLAG=()
if [[ -n "${OUT:-}" ]]; then
  OUT_FLAG=(--out "$OUT")
fi
L2_FLAGS=()
if [[ -n "${RVT_L2_MODE:-}" && "${RVT_L2_MODE}" != "base" ]]; then
  L2_FLAGS+=(--rvt-l2-mode "$RVT_L2_MODE")
  L2_FLAGS+=(--rvt-l2-eps "${RVT_L2_EPS:-0.05}")
  if [[ "${RVT_L2_ALL_LAYERS:-0}" == "1" ]]; then
    L2_FLAGS+=(--rvt-l2-all-layers)
  fi
fi

python3 experiments/rvt_exp_2026_008_mrmp_row.py \
  --jsonl "$JSONL" \
  --line "$LINE" \
  --max-rows "$MAX_ROWS" \
  --model "$MODEL" \
  --layer "$LAYER" \
  --seed "$SEED" \
  "${CPU_FLAG[@]}" \
  "${AWAI_FLAGS[@]}" \
  "${STRICT_FLAG[@]}" \
  "${PRETTY_FLAG[@]}" \
  "${OUT_FLAG[@]}" \
  "${L2_FLAGS[@]}" \
  "$@"
