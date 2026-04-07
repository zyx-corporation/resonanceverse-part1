#!/usr/bin/env bash
# Phase II-A 本番（注意のみ・rinna・先頭 3146 窓）: 図・strict・マニフェスト verify まで。
# 実行例:
#   bash run_phase2a_mrmp_n3146_rinna.sh
# revision 固定: REVISION=... bash run_phase2a_mrmp_n3146_rinna.sh
#
# MRMP 窓本線では Day 0（スキーマ＋manifest）を既定で有効: DAY0_CHECK=1。
# WINDOWS を審判済み JSONL に差し替えるときは DAY0_CHECK=0 を付ける。
# 任意: 本線（bootstrap）のあと RVT 小バッチを同じ WINDOWS で回す → RVT_EXPLORE=1（環境は子へ継承）。

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$REPO_ROOT"

export WINDOWS="${WINDOWS:-experiments/logs/mrmp_prepared/windows.jsonl}"
export MAX_ROWS="${MAX_ROWS:-3146}"
export BOOT="${BOOT:-4000}"
export MODEL="${MODEL:-rinna/japanese-gpt2-medium}"
export OUT_PREFIX="${OUT_PREFIX:-experiments/logs/v7_phase2a_mrmp_tau_n3146}"
export BUNDLE_JSON="${BUNDLE_JSON:-experiments/baselines/v7_phase2a_mrmp_tau_n3146_bundle_v1.json}"
export DAY0_CHECK="${DAY0_CHECK:-1}"
export GENERATE_PAPER_PLOTS=1
export VALIDATE_STRICT=1
export WRITE_REPRO_MANIFEST=1

bash experiments/run_phase2a_mrmp_tau.sh
python3 experiments/v7_phase2a_repro_manifest.py --verify "${OUT_PREFIX}_repro_manifest.json"
