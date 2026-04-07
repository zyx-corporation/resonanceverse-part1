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
#     （終了時に v7_phase2a_bundle_validate.py --strict。要 pipeline_log → 下記 PIPELINE_LOG）
#   PIPELINE_LOG  既定: experiments/logs/run_phase2a_mrmp_tau_gpu.log（全ステップを tee 上書き）
#   GENERATE_PLOTS=1 bash experiments/run_phase2a_mrmp_tau.sh
#     （終了時に v7_phase2a_tau_plots.py で PNG を出力）
#   GENERATE_PAPER_PLOTS=1 … 同上で --paper（PNG 300dpi + PDF）
#   WRITE_REPRO_MANIFEST=1 BUNDLE_JSON=experiments/baselines/…_bundle_v1.json …
#     （v7_phase2a_repro_manifest.py で ${OUT_PREFIX}_repro_manifest.json）
#   REPRO_MANIFEST_STRICT=1 … マニフェスト時に成果物・図の欠落でも失敗
#   DAY0_CHECK=1 … empirical_run の前に RVT Day 0（必須キー・同階 manifest 件数一致）。
#     **v7_mrmp_prepare 出力の WINDOWS** を使うとき推奨。審判のみ JSONL 等で manifest が無いときは **付けない**。
#   DAY0_MIN_ROWS=N … 上記実行時に --min-rows N を付与（任意）
#   RVT_EXPLORE=1 … ブートストラップ完了後に run_rvt_explore.sh（同一 WINDOWS、小バッチ）。
#     RVT_EXPLORE_SKIP_DAY0 既定 1（本シェルで Day0 済なら二重実行を避ける）。0 で explore 内 Day0 も実行。
#     RVT_EXPLORE_MAX_ROWS / RVT_EXPLORE_LINE / RVT_EXPLORE_CPU（未設定時は CPU と同じ）

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

# 終了時に tee を閉じてから再現マニフェストを走らせるため、元の端末を FD 3/4 に保存する。
exec 3>&1 4>&2

# bundle（v7_phase2a_mrmp_tau_n3146_bundle_v1 等）の pipeline_log と整合。VALIDATE_STRICT=1 に必須。
PIPELINE_LOG="${PIPELINE_LOG:-experiments/logs/run_phase2a_mrmp_tau_gpu.log}"
mkdir -p "$(dirname "$PIPELINE_LOG")"
exec > >(tee "$PIPELINE_LOG") 2>&1

echo "=== run_phase2a_mrmp_tau: start $(date -Iseconds) ==="
echo "  PIPELINE_LOG=$PIPELINE_LOG OUT_PREFIX=$OUT_PREFIX WINDOWS=$WINDOWS MODEL=$MODEL MAX_ROWS=$MAX_ROWS"

if [[ "${DAY0_CHECK:-0}" == "1" ]]; then
  echo "=== run_phase2a_mrmp_tau: RVT Day 0 (schema + manifest) ==="
  DAY0_EXTRA=()
  if [[ -n "${DAY0_MIN_ROWS:-}" ]]; then
    DAY0_EXTRA=(--min-rows "$DAY0_MIN_ROWS")
  fi
  python3 experiments/rvt_exp_2026_008_day0_checks.py \
    --windows "$WINDOWS" \
    --strict-manifest \
    "${DAY0_EXTRA[@]}"
fi

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

if [[ "${RVT_EXPLORE:-0}" == "1" ]]; then
  echo "=== run_phase2a_mrmp_tau: RVT explore (optional, WINDOWS=$WINDOWS) ==="
  EXPL_CPU="${RVT_EXPLORE_CPU:-${CPU:-0}}"
  JSONL="$WINDOWS" \
    SKIP_DAY0="${RVT_EXPLORE_SKIP_DAY0:-1}" \
    MAX_ROWS="${RVT_EXPLORE_MAX_ROWS:-5}" \
    LINE="${RVT_EXPLORE_LINE:-0}" \
    CPU="$EXPL_CPU" \
    bash experiments/run_rvt_explore.sh
fi

if [[ "${GENERATE_PLOTS:-0}" == "1" ]]; then
  echo "=== run_phase2a_mrmp_tau: tau_plots (PNG) ==="
  python3 experiments/v7_phase2a_tau_plots.py "${OUT_PREFIX}_with_contrib.json"
fi

if [[ "${GENERATE_PAPER_PLOTS:-0}" == "1" ]]; then
  echo "=== run_phase2a_mrmp_tau: tau_plots --paper (PNG+PDF) ==="
  python3 experiments/v7_phase2a_tau_plots.py "${OUT_PREFIX}_with_contrib.json" --paper
fi

# strict 検証のあと tee を閉じ、終了行だけログに追記し、再現マニフェストは端末へ（stdout が pipeline_log を汚さない）
if [[ "${VALIDATE_STRICT:-0}" == "1" ]]; then
  echo "=== run_phase2a_mrmp_tau: bundle_validate --strict (OUT_PREFIX=$OUT_PREFIX) ==="
  if [[ -n "${BUNDLE_JSON:-}" ]]; then
    python3 experiments/v7_phase2a_bundle_validate.py --strict --bundle "$BUNDLE_JSON" --out-prefix "$OUT_PREFIX"
  else
    python3 experiments/v7_phase2a_bundle_validate.py --strict --out-prefix "$OUT_PREFIX"
  fi
fi

exec 1>&3 2>&4
wait || true
echo "=== run_phase2a_mrmp_tau: end $(date -Iseconds) ===" | tee -a "$PIPELINE_LOG"

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
