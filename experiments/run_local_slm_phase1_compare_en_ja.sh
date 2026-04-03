#!/usr/bin/env bash
# Phase 1 補助: 同一 JSONL スライスで英語 gpt2 と日本語 rinna の R(τ) を並走し、
# v7_phase2a_compare_runs.py で表（JSON・MD）を出す。数値の優劣は解釈しない。
#
# 環境変数:
#   JSONL, MAX_DIALOGUES, MAX_ROWS, TAU_MAX, SEED, LAYER, BASE（出力プレフィックス）
#   CPU  1 なら --cpu
#
# 例:
#   bash experiments/run_local_slm_phase1_compare_en_ja.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JSONL="${JSONL:-experiments/data/v7_mrmp_sample.jsonl}"
MAX_DIALOGUES="${MAX_DIALOGUES:-1}"
MAX_ROWS="${MAX_ROWS:-8}"
TAU_MAX="${TAU_MAX:-4}"
SEED="${SEED:-0}"
LAYER="${LAYER:--1}"
BASE="${BASE:-experiments/logs/local_slm_en_ja_compare}"

CPU_FLAG=()
if [[ "${CPU:-0}" == "1" ]]; then
  CPU_FLAG=(--cpu)
fi

OUT_EN="${BASE}_gpt2_with_contrib.json"
OUT_JA="${BASE}_rinna_with_contrib.json"

mkdir -p "$(dirname "$OUT_EN")"

COMMON=(
  --jsonl "$JSONL"
  --max-dialogues "$MAX_DIALOGUES"
  --max-rows "$MAX_ROWS"
  --tau-max "$TAU_MAX"
  --seed "$SEED"
  --layer "$LAYER"
  --export-contributions
  "${CPU_FLAG[@]}"
)

echo "=== compare en_ja: gpt2 ==="
PYTHONUNBUFFERED=1 python3 experiments/v7_phase2a_empirical_run.py \
  "${COMMON[@]}" \
  --model gpt2 \
  --out "$OUT_EN"

echo "=== compare en_ja: rinna ==="
PYTHONUNBUFFERED=1 python3 experiments/v7_phase2a_empirical_run.py \
  "${COMMON[@]}" \
  --model rinna/japanese-gpt2-medium \
  --out "$OUT_JA"

echo "=== compare en_ja: table ==="
python3 experiments/v7_phase2a_compare_runs.py \
  "$OUT_EN" "$OUT_JA" \
  --labels en_gpt2,ja_rinna \
  --out-json "${BASE}_compare.json" \
  --out-md "${BASE}_compare.md"

echo "=== done ==="
echo "  $OUT_EN"
echo "  $OUT_JA"
echo "  ${BASE}_compare.json"
echo "  ${BASE}_compare.md"
