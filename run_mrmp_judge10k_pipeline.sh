#!/usr/bin/env bash
# 主結果用: MRMP 審判（hf_local）→ rinna 注意 → Phase II-A 凍結（judge10k bundle）まで。
# 実行例（カレントはどこでも可）:
#   bash run_mrmp_judge10k_pipeline.sh
#   bash path/to/resonanceverse-alpha/run_mrmp_judge10k_pipeline.sh
# revision 固定: HF_REVISION=... REVISION=... bash run_mrmp_judge10k_pipeline.sh
# 審判済みのみ: SKIP_MRMP_JUDGE=1 bash run_mrmp_judge10k_pipeline.sh
#
# MRMP 窓（SRC）の検証は審判前に任意:
#   python experiments/rvt_exp_2026_008_day0_checks.py --windows experiments/logs/mrmp_prepared/windows.jsonl --strict-manifest
# （Phase II-A 本体は WINDOWS=審判 JSONL のため DAY0_CHECK は付けない。詳細は experiments/run_mrmp_judge_hf_local_then_phase2a_judge10k.sh 冒頭）

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$REPO_ROOT"
exec bash "$REPO_ROOT/experiments/run_mrmp_judge_hf_local_then_phase2a_judge10k.sh"
