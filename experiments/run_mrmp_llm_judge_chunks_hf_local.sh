#!/usr/bin/env bash
# MRMP 窓に対し 6 軸審判を hf_local（既定 Swallow-7B-Instruct）でチャンク実行。
# run_mrmp_llm_judge_chunks.sh と同じ CHUNK / N_CHUNKS / SRC / OUT だが OpenAI ではなく
# --provider hf_local。初回はモデル取得が重い。
#
# 環境変数:
#   SRC, OUT, CHUNK, N_CHUNKS, LOG（run_mrmp_llm_judge_chunks.sh と同様）
#   HF_MODEL  既定: tokyotech-llm/Swallow-7b-instruct-hf
#   CPU       1 なら --cpu
#   HF_REVISION  省略可（固定時は SHA）
#   HF_MAX_NEW_TOKENS  既定: 256
#   TEMPERATURE  既定: 0
#
# 例:
#   bash experiments/run_mrmp_llm_judge_chunks_hf_local.sh
#   CHUNK=100 N_CHUNKS=2 CPU=1 bash experiments/run_mrmp_llm_judge_chunks_hf_local.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SRC="${SRC:-experiments/logs/mrmp_prepared/windows.jsonl}"
OUT="${OUT:-experiments/logs/v7_judge_hf_local_chunks.jsonl}"
CHUNK="${CHUNK:-1000}"
N_CHUNKS="${N_CHUNKS:-10}"
LOG="${LOG:-${OUT%.jsonl}_hf_local_run.log}"
HF_MODEL="${HF_MODEL:-tokyotech-llm/Swallow-7b-instruct-hf}"
HF_MAX_NEW_TOKENS="${HF_MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0}"

CPU_FLAG=()
if [[ "${CPU:-0}" == "1" ]]; then
  CPU_FLAG=(--cpu)
fi
REV_ARGS=()
if [[ -n "${HF_REVISION:-}" ]]; then
  REV_ARGS=(--hf-revision "$HF_REVISION")
fi

if [[ ! -f "$SRC" ]]; then
  echo "SRC not found: $SRC" >&2
  echo "先に: python experiments/v7_mrmp_prepare.py" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

PY=(
  python3 experiments/v7_phase1a_llm_judge_six_axes.py
  --provider hf_local
  --hf-model "$HF_MODEL"
  --hf-max-new-tokens "$HF_MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  "${CPU_FLAG[@]}"
  "${REV_ARGS[@]}"
)

{
  echo "=== run_mrmp_llm_judge_chunks_hf_local start $(date -Iseconds) ==="
  echo "SRC=$SRC OUT=$OUT CHUNK=$CHUNK N_CHUNKS=$N_CHUNKS HF_MODEL=$HF_MODEL"
  echo

  "${PY[@]}" --jsonl "$SRC" --offset 0 --max-rows "$CHUNK" \
    --out-jsonl "$OUT" \
    --out-summary "${OUT%.jsonl}_summary_c01.json"

  for ((i = 2; i <= N_CHUNKS; i++)); do
    printf -v TAG '%02d' "$i"
    echo "=== chunk $i/$N_CHUNKS $(date -Iseconds) lines=$(wc -l <"$OUT") ==="
    "${PY[@]}" --jsonl "$SRC" --resume --max-rows "$CHUNK" \
      --out-jsonl "$OUT" \
      --out-summary "${OUT%.jsonl}_summary_c${TAG}.json"
  done

  echo "=== run_mrmp_llm_judge_chunks_hf_local done $(date -Iseconds) ==="
  wc -l "$OUT"
} 2>&1 | tee "$LOG"
