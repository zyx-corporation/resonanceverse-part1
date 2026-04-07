#!/usr/bin/env bash
# RVT 探索ワンショット: Day0（任意）→ 小さい MRMP バッチ（既定 5 行）。
# windows.jsonl が無いときはメッセージのみで exit 0（CI や未準備マシン向け）。
#
# 環境変数:
#   JSONL       既定: experiments/logs/mrmp_prepared/windows.jsonl
#   SKIP_DAY0   1 なら Day0 をスキップ
#   DAY0_STRICT 1 なら day0_checks に --strict-manifest（manifest 必須）
#   LINE / MAX_ROWS / MODEL / CPU / ACCUMULATE_AWAI / AWAI_OUT …
#     → run_rvt_mrmp_batch.sh にそのまま渡る（PRESET 相当は手動）
#
# 例:
#   bash experiments/run_rvt_explore.sh
#   DAY0_STRICT=1 MAX_ROWS=20 bash experiments/run_rvt_explore.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JSONL="${JSONL:-experiments/logs/mrmp_prepared/windows.jsonl}"
SKIP_DAY0="${SKIP_DAY0:-0}"
DAY0_STRICT="${DAY0_STRICT:-0}"
LINE="${LINE:-0}"
MAX_ROWS="${MAX_ROWS:-5}"

if [[ ! -f "$JSONL" ]]; then
  echo "RVT explore: スキップ（$JSONL がありません）。v7_mrmp_prepare 後か JSONL= を指定。" >&2
  exit 0
fi

if [[ "$SKIP_DAY0" != "1" ]]; then
  if [[ "$DAY0_STRICT" == "1" ]]; then
    python3 experiments/rvt_exp_2026_008_day0_checks.py \
      --windows "$JSONL" \
      --strict-manifest \
      --min-rows 1
  else
    python3 experiments/rvt_exp_2026_008_day0_checks.py \
      --windows "$JSONL" \
      --no-manifest \
      --min-rows 1
  fi
fi

export JSONL
export LINE
export MAX_ROWS
CPU="${CPU:-1}" bash experiments/run_rvt_mrmp_batch.sh

echo "" >&2
echo "RVT explore: 次の任意ステップ（例）" >&2
echo "  CPU=1 bash experiments/run_rvt_l2_smoke.sh --mode sym" >&2
echo "  python experiments/rvt_exp_2026_008_ablation_runner.py --preset explore --emit-shell" >&2
