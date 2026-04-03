#!/usr/bin/env bash
# Phase 1 スモーク: 日本語 GPT-2（rinna）で MRMP サンプル 1 対話・短い τ のみ R(τ) を生成する。
# ネットワーク: 初回のみ HF から重み取得。128GB / M3 Max 想定（--cpu も可）。
#
# 例:
#   bash experiments/run_local_slm_phase1_smoke.sh
#   CPU=1 bash experiments/run_local_slm_phase1_smoke.sh
#
# 環境変数:
#   JSONL   既定: experiments/data/v7_mrmp_sample.jsonl
#   MODEL   既定: rinna/japanese-gpt2-medium
#   OUT     既定: experiments/logs/local_slm_phase1_smoke_with_contrib.json
#   CPU     1 なら --cpu

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

JSONL="${JSONL:-experiments/data/v7_mrmp_sample.jsonl}"
MODEL="${MODEL:-rinna/japanese-gpt2-medium}"
OUT="${OUT:-experiments/logs/local_slm_phase1_smoke_with_contrib.json}"

CPU_FLAG=()
if [[ "${CPU:-0}" == "1" ]]; then
  CPU_FLAG=(--cpu)
fi

mkdir -p "$(dirname "$OUT")"

echo "=== v7_local_env_check ==="
python3 experiments/v7_local_env_check.py

echo "=== Phase II-A empirical_run (Japanese CausalLM smoke) ==="
PYTHONUNBUFFERED=1 python3 experiments/v7_phase2a_empirical_run.py \
  --jsonl "$JSONL" \
  --model "$MODEL" \
  "${CPU_FLAG[@]}" \
  --max-dialogues 1 \
  --max-rows 8 \
  --tau-max 4 \
  --seed 0 \
  --layer -1 \
  --out "$OUT"

echo "=== done: $OUT ==="
echo "    taus: check stdout line v7_phase2a_empirical_run_ok"
