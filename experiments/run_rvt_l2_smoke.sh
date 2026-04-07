#!/usr/bin/env bash
# RVT L2 logits スモーク（eager Causal LM）。Phase II-A と同じ HF ID を試すときは MODEL を上書き。
#
# 例:
#   bash experiments/run_rvt_l2_smoke.sh
#   MODEL=rinna/japanese-gpt2-medium CPU=1 bash experiments/run_rvt_l2_smoke.sh --layer -1
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-gpt2}"
EXTRA=()
if [[ "${CPU:-0}" == "1" ]]; then
  EXTRA+=(--cpu)
fi

python3 experiments/rvt_exp_2026_008_l2_smoke.py --model "$MODEL" "${EXTRA[@]}" "$@"
