#!/usr/bin/env bash
# ローカル SLM Phase 3: MRMP 窓を OFFSET / MAX_ROWS で切り出し、Phase II-A 全行程
# （with_contrib → tau_summary → bootstrap）を 1 チャンク分実行する。
# 実体は run_phase2a_mrmp_tau.sh に MODEL 既定を上書きして委譲。
#
# 環境変数（run_phase2a_mrmp_tau.sh と同じに加え）:
#   MODEL     既定: rinna/japanese-gpt2-medium
#   OFFSET    既定: 0
#   MAX_ROWS  既定: 500
#   OUT_PREFIX 既定: experiments/logs/v7_local_slm_chunk_o${OFFSET}_n${MAX_ROWS}
#
# 例:
#   bash experiments/run_local_slm_phase3_mrmp_chunk.sh
#   OFFSET=500 MAX_ROWS=500 OUT_PREFIX=experiments/logs/v7_local_slm_c1 bash experiments/run_local_slm_phase3_mrmp_chunk.sh
#   CPU=1 bash experiments/run_local_slm_phase3_mrmp_chunk.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export MODEL="${MODEL:-rinna/japanese-gpt2-medium}"
export OFFSET="${OFFSET:-0}"
export MAX_ROWS="${MAX_ROWS:-500}"
export OUT_PREFIX="${OUT_PREFIX:-experiments/logs/v7_local_slm_chunk_o${OFFSET}_n${MAX_ROWS}}"

exec bash experiments/run_phase2a_mrmp_tau.sh
