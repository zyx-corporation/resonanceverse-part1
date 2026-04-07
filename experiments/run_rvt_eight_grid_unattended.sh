#!/usr/bin/env bash
# 八条件グリッドを無人で順実行（既定: dry-run。本実行は環境変数参照）。
# RVT_EIGHT_GRID_PREPEND_EXPLORE=1 で各条件プラン先頭に rvt_explore を付与。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MANIFEST="${RVT_EIGHT_GRID_MANIFEST:-experiments/logs/rvt_eight_grid_manifest.json}"
DRY_ARGS=(--dry-run)
if [[ "${RVT_EIGHT_GRID_NO_DRY_RUN:-}" == "1" ]]; then
  DRY_ARGS=(--no-dry-run)
fi
CONT_ARGS=()
if [[ "${RVT_EIGHT_GRID_CONTINUE:-}" == "1" ]]; then
  CONT_ARGS=(--continue-on-error)
fi
PREPEND_ARGS=()
if [[ "${RVT_EIGHT_GRID_PREPEND_EXPLORE:-}" == "1" ]]; then
  PREPEND_ARGS=(--eight-grid-prepend-explore)
fi

exec python experiments/rvt_exp_2026_008_ablation_runner.py \
  --preset eight_grid \
  --run-eight-grid \
  "${DRY_ARGS[@]}" \
  "${CONT_ARGS[@]}" \
  "${PREPEND_ARGS[@]}" \
  --manifest "$MANIFEST" \
  "$@"
