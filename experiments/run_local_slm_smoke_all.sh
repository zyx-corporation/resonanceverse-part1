#!/usr/bin/env bash
# Phase 1（注意・rinna）＋ bundle 非 strict 検証 ＋ Phase 2（審判・Swallow 1 行）を順に実行。
# Phase 2 は初回ダウンロードが重い。CPU=1 で CPU 強制可。
#
#   bash experiments/run_local_slm_smoke_all.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

bash experiments/run_local_slm_phase1_smoke.sh

echo "=== bundle validate (local SLM Phase 1 pointer, non-strict) ==="
python3 experiments/v7_phase2a_bundle_validate.py \
  --bundle experiments/baselines/v7_local_slm_phase1_smoke_bundle_v1.json

bash experiments/run_local_slm_phase2_smoke.sh
