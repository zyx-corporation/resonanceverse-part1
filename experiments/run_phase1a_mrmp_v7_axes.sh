#!/usr/bin/env bash
# Phase I-A: MRMP 窓 JSONL（6 軸 trust_ab … history_ba 等）× 最終層 ||S_asym||_F の相関
#
# 既定は審判済み 10k 相当パス（存在しなければローカルでパスを差し替え）。
# 環境変数:
#   WINDOWS  既定: experiments/logs/v7_judge_10k.jsonl
#   MAX_ROWS 既定: 3146
#   OUT      既定: experiments/logs/v7_phase1a_mrmp_judge_v7_axes_corr.json
#   MODEL    既定: gpt2
#   CPU      1 なら --cpu
#
# 例:
#   bash experiments/run_phase1a_mrmp_v7_axes.sh
#   MAX_ROWS=200 CPU=1 bash experiments/run_phase1a_mrmp_v7_axes.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WINDOWS="${WINDOWS:-experiments/logs/v7_judge_10k.jsonl}"
MAX_ROWS="${MAX_ROWS:-3146}"
OUT="${OUT:-experiments/logs/v7_phase1a_mrmp_judge_v7_axes_corr.json}"
MODEL="${MODEL:-gpt2}"

CPU_FLAG=()
if [[ "${CPU:-0}" == "1" ]]; then
  CPU_FLAG=(--cpu)
fi

mkdir -p "$(dirname "$OUT")"

echo "=== run_phase1a_mrmp_v7_axes: pilot_jsonl (v7 six axes, not --mrmp-labels) ==="
PYTHONUNBUFFERED=1 python3 experiments/v7_phase1a_pilot_jsonl.py \
  --jsonl "$WINDOWS" \
  --max-rows "$MAX_ROWS" \
  --model "$MODEL" \
  "${CPU_FLAG[@]}" \
  --seed "${SEED:-0}" \
  --layer "${LAYER:--1}" \
  --out "$OUT"

echo "=== done: $OUT ==="
