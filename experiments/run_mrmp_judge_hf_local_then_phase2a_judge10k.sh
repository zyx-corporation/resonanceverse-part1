#!/usr/bin/env bash
# MRMP 窓に 6 軸審判（hf_local）→ 審判済み JSONL → Phase II-A（注意 rinna）→ judge10k bundle 相当の凍結。
# 主結果に「審判＋補助 6 軸」を載せるときの推奨一括（時間: 審判が支配的・数時間級になりうる）。
#
# 前提: python experiments/v7_mrmp_prepare.py で SRC（既定 windows.jsonl）が存在すること。
#
# 環境変数（審判）:
#   SRC              既定: experiments/logs/mrmp_prepared/windows.jsonl
#   OUT_JUDGE        既定: experiments/logs/v7_judge_10k.jsonl（bundle の input 名に合わせる）
#   CHUNK, N_CHUNKS  既定: 3146 と 1（先頭 3146 窓を 1 チャンク）
#   HF_MODEL, HF_REVISION, HF_MAX_NEW_TOKENS, TEMPERATURE, CPU
#   SKIP_MRMP_JUDGE  1 なら審判をスキップ（既存 OUT_JUDGE を WINDOWS に使う）
#
# 環境変数（Phase II-A）:
#   MAX_ROWS, MODEL, REVISION, BOOT, OUT_PREFIX, CPU
#   DAY0_CHECK  既定は未設定（Phase II-A の WINDOWS は審判済み JSONL のため --strict-manifest 非推奨）。
#     MRMP 窓のみ検証するなら審判**前**に: python experiments/rvt_exp_2026_008_day0_checks.py --windows "$SRC" --strict-manifest
#   OUT_PREFIX 既定: experiments/logs/v7_phase2a_mrmp_tau_n3146_judge10k
#   BUNDLE_JSON 既定: experiments/baselines/v7_phase2a_mrmp_tau_n3146_judge10k_bundle_v1.json
#   REPRO_MANIFEST_STRICT 既定: 1（図・PDF 欠落でマニフェスト失敗）
#
# 例（リポジトリルートで実行。角括弧 < > はシェルでリダイレクトになるので使わない）:
#   bash experiments/run_mrmp_judge_hf_local_then_phase2a_judge10k.sh
#   HF_REVISION=abcd1234 REVISION=ef567890 bash experiments/run_mrmp_judge_hf_local_then_phase2a_judge10k.sh
#   SKIP_MRMP_JUDGE=1 bash experiments/run_mrmp_judge_hf_local_then_phase2a_judge10k.sh
#
# 注意モデル ID に / が含まれるため、未クォートの代入で壊れないようこのスクリプト内では export で設定済み。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ ! -f experiments/run_phase2a_mrmp_tau.sh ]]; then
  echo "error: リポジトリルートが不正です（experiments/run_phase2a_mrmp_tau.sh が見つかりません）。" >&2
  echo "  cd /path/to/resonanceverse-alpha && bash experiments/run_mrmp_judge_hf_local_then_phase2a_judge10k.sh" >&2
  exit 2
fi

SRC="${SRC:-experiments/logs/mrmp_prepared/windows.jsonl}"
OUT_JUDGE="${OUT_JUDGE:-experiments/logs/v7_judge_10k.jsonl}"
CHUNK="${CHUNK:-3146}"
N_CHUNKS="${N_CHUNKS:-1}"

if [[ "${SKIP_MRMP_JUDGE:-0}" != "1" ]]; then
  export SRC
  export OUT="$OUT_JUDGE"
  export CHUNK
  export N_CHUNKS
  bash experiments/run_mrmp_llm_judge_chunks_hf_local.sh
else
  echo "=== SKIP_MRMP_JUDGE=1: 審判スキップ（WINDOWS=$OUT_JUDGE を使用）==="
  if [[ ! -f "$OUT_JUDGE" ]]; then
    echo "OUT_JUDGE not found: $OUT_JUDGE" >&2
    exit 1
  fi
fi

export WINDOWS="$OUT_JUDGE"
export MAX_ROWS="${MAX_ROWS:-3146}"
export MODEL="${MODEL:-rinna/japanese-gpt2-medium}"
export OUT_PREFIX="${OUT_PREFIX:-experiments/logs/v7_phase2a_mrmp_tau_n3146_judge10k}"
export BOOT="${BOOT:-4000}"
export BUNDLE_JSON="${BUNDLE_JSON:-experiments/baselines/v7_phase2a_mrmp_tau_n3146_judge10k_bundle_v1.json}"
export GENERATE_PLOTS=1
export GENERATE_PAPER_PLOTS=1
export VALIDATE_STRICT=1
export WRITE_REPRO_MANIFEST=1
export REPRO_MANIFEST_STRICT="${REPRO_MANIFEST_STRICT:-1}"

bash experiments/run_phase2a_mrmp_tau.sh

python3 experiments/v7_phase2a_repro_manifest.py --verify "${OUT_PREFIX}_repro_manifest.json"
