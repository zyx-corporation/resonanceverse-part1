#!/usr/bin/env bash
# MRMP Phase II-A: 実データ R(τ) → サマリ → ブートストラップ（寄与付き JSON 前提）
#
# 環境変数で上書き可能:
#   WINDOWS   既定: experiments/logs/mrmp_prepared/windows.jsonl
#   OFFSET    既定: 0（JSONL 先頭からスキップする行数・チャンク用）
#   MAX_ROWS  既定: 3146（先頭 N 行）
#   MODEL     既定: gpt2（例: rinna/japanese-gpt2-medium）
#   REVISION  省略時: HF revision 指定なし（固定時はコミット SHA）
#   CPU       1 なら empirical_run に --cpu
#   OUT_PREFIX 既定: experiments/logs/v7_phase2a_mrmp_tau_n3146
#   BOOT      既定: 4000
# 用法:
#   bash experiments/run_phase2a_mrmp_tau.sh
#   MAX_ROWS=500 bash experiments/run_phase2a_mrmp_tau.sh
#   WINDOWS=experiments/logs/v7_judge_10k.jsonl bash experiments/run_phase2a_mrmp_tau.sh
#     （6 軸 LLM 審判済み JSONL なら with_contrib に auxiliary が載り、サマリ MD/JSON に反映）
#   VALIDATE_STRICT=1 bash experiments/run_phase2a_mrmp_tau.sh
#     （終了時に v7_phase2a_bundle_validate.py --strict を実行）
#   GENERATE_PLOTS=1 bash experiments/run_phase2a_mrmp_tau.sh
#     （終了時に v7_phase2a_tau_plots.py で PNG を出力）
#   GENERATE_PAPER_PLOTS=1 … 同上で --paper（PNG 300dpi + PDF）
#   WRITE_REPRO_MANIFEST=1 BUNDLE_JSON=experiments/baselines/…_bundle_v1.json …
#     （v7_phase2a_repro_manifest.py で ${OUT_PREFIX}_repro_manifest.json）
#   REPRO_MANIFEST_STRICT=1 … マニフェスト時に成果物・図の欠落でも失敗

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WINDOWS="${WINDOWS:-experiments/logs/mrmp_prepared/windows.jsonl}"
OFFSET="${OFFSET:-0}"
MAX_ROWS="${MAX_ROWS:-3146}"
MODEL="${MODEL:-gpt2}"
OUT_PREFIX="${OUT_PREFIX:-experiments/logs/v7_phase2a_mrmp_tau_n3146}"
BOOT="${BOOT:-4000}"

CPU_FLAG=()
if [[ "${CPU:-0}" == "1" ]]; then
  CPU_FLAG=(--cpu)
fi
REV_ARGS=()
if [[ -n "${REVISION:-}" ]]; then
  REV_ARGS=(--revision "$REVISION")
fi

mkdir -p "$(dirname "$OUT_PREFIX")"

echo "=== run_phase2a_mrmp_tau: empirical_run (export contributions) ==="
PYTHONUNBUFFERED=1 python3 experiments/v7_phase2a_empirical_run.py \
  --jsonl "$WINDOWS" \
  --offset "$OFFSET" \
  --max-rows "$MAX_ROWS" \
  --model "$MODEL" \
  "${CPU_FLAG[@]}" \
  "${REV_ARGS[@]}" \
  --export-contributions \
  --out "${OUT_PREFIX}_with_contrib.json"

echo "=== run_phase2a_mrmp_tau: tau_summary ==="
python3 experiments/v7_phase2a_tau_summary.py \
  "${OUT_PREFIX}_with_contrib.json" \
  --out-md "${OUT_PREFIX}_summary.md" \
  --out-json "${OUT_PREFIX}_summary.json"

echo "=== run_phase2a_mrmp_tau: bootstrap (paired + full table) ==="
python3 experiments/v7_phase2a_tau_bootstrap.py \
  "${OUT_PREFIX}_with_contrib.json" \
  --boot "$BOOT" \
  --paired-diff 0,1 \
  --paired-diff 0,103 \
  --out "${OUT_PREFIX}_bootstrap.json" \
  --out-md "${OUT_PREFIX}_bootstrap.md"

echo "=== run_phase2a_mrmp_tau done ==="
echo "  ${OUT_PREFIX}_with_contrib.json"
echo "  ${OUT_PREFIX}_summary.md"
echo "  ${OUT_PREFIX}_summary.json"
echo "  ${OUT_PREFIX}_bootstrap.json"
echo "  ${OUT_PREFIX}_bootstrap.md"

if [[ "${VALIDATE_STRICT:-0}" == "1" ]]; then
  echo "=== run_phase2a_mrmp_tau: bundle_validate --strict (OUT_PREFIX=$OUT_PREFIX) ==="
  python3 experiments/v7_phase2a_bundle_validate.py --strict --out-prefix "$OUT_PREFIX"
fi

if [[ "${GENERATE_PLOTS:-0}" == "1" ]]; then
  echo "=== run_phase2a_mrmp_tau: tau_plots (PNG) ==="
  python3 experiments/v7_phase2a_tau_plots.py "${OUT_PREFIX}_with_contrib.json"
fi

if [[ "${GENERATE_PAPER_PLOTS:-0}" == "1" ]]; then
  echo "=== run_phase2a_mrmp_tau: tau_plots --paper (PNG+PDF) ==="
  python3 experiments/v7_phase2a_tau_plots.py "${OUT_PREFIX}_with_contrib.json" --paper
fi

if [[ "${WRITE_REPRO_MANIFEST:-0}" == "1" ]]; then
  if [[ -n "${BUNDLE_JSON:-}" ]]; then
    echo "=== run_phase2a_mrmp_tau: repro_manifest → ${OUT_PREFIX}_repro_manifest.json ==="
    RM_EXTRA=()
    if [[ "${REPRO_MANIFEST_STRICT:-0}" == "1" ]]; then
      RM_EXTRA=(--strict)
    fi
    python3 experiments/v7_phase2a_repro_manifest.py \
      --bundle "$BUNDLE_JSON" \
      --out-prefix "$OUT_PREFIX" \
      "${RM_EXTRA[@]}" \
      --out "${OUT_PREFIX}_repro_manifest.json"
  else
    echo "WRITE_REPRO_MANIFEST=1 だが BUNDLE_JSON 未設定のためスキップ（例: BUNDLE_JSON=experiments/baselines/v7_phase2a_mrmp_tau_n3146_judge10k_bundle_v1.json）" >&2
  fi
fi
