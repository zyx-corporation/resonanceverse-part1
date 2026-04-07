#!/usr/bin/env bash
# 7B 審判 ×2（Swallow-7B-Instruct・Qwen2.5-7B-Instruct）と 3B ×1（Qwen2.5-3B-Instruct）を
# 同一入力で n=1000 行連続実行し、pair_agreement を 3 本書き出す。
#
# 前提: experiments/logs/mrmp_prepared/windows.jsonl が存在し、
#       offset+1000 行分以上の非空行があること（未準備なら v7_mrmp_prepare.py）。
#
# 環境変数で上書き可能:
#   JSONL   入力（既定: experiments/logs/mrmp_prepared/windows.jsonl）
#   OFFSET  既定 0
#   N       既定 1000
#   OUT_DIR 既定: experiments/logs/slm_judge_triple_n1000
#   EXTRA   python に追加で渡す引数（例: EXTRA="--cpu"）
#
# 例:
#   bash experiments/run_slm_judge_7b2_3b_n1000.sh
#   JSONL=/path/to/windows.jsonl N=500 bash experiments/run_slm_judge_7b2_3b_n1000.sh
#
# 先頭 100 行のみ（MRMP 標準）: experiments/run_slm_judge_mrmp_n100.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JSONL="${JSONL:-experiments/logs/mrmp_prepared/windows.jsonl}"
OFFSET="${OFFSET:-0}"
N="${N:-1000}"
OUT_DIR="${OUT_DIR:-experiments/logs/slm_judge_triple_n1000}"

exec python3 experiments/v7_slm_judge_triple_run.py \
  --jsonl "$JSONL" \
  --n "$N" \
  --offset "$OFFSET" \
  --out-dir "$OUT_DIR" \
  --temperature 0 \
  ${EXTRA:-}
