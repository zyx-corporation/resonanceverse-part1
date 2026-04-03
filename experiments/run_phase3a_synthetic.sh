#!/usr/bin/env bash
# Phase III（合成）: あわい Ω の軌跡デモ → JSON（人手アノテ無し）
#
# 環境変数で上書き可能:
#   SEED  既定: 0
#   T     既定: 200（時系列長）
#   D     既定: 6（ベクトル次元）
#   OUT   既定: experiments/logs/v7_phase3a_synthetic_default.json
# 用法:
#   bash experiments/run_phase3a_synthetic.sh
#   T=500 SEED=1 bash experiments/run_phase3a_synthetic.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-0}"
T="${T:-200}"
D="${D:-6}"
OUT="${OUT:-experiments/logs/v7_phase3a_synthetic_default.json}"

mkdir -p "$(dirname "$OUT")"

echo "=== run_phase3a_synthetic: v7_phase3a_awai_metrics ==="
python3 experiments/v7_phase3a_awai_metrics.py \
  --seed "$SEED" \
  --T "$T" \
  --d "$D" \
  --out "$OUT"

echo "=== run_phase3a_synthetic done ==="
echo "  $OUT"
