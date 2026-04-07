#!/usr/bin/env bash
# Oboro L3 の HF なし合成 JSON（CI・スキーマ確認用）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PROF="${RVT_OBORO_PROFILE:-full}"
exec python experiments/rvt_exp_2026_008_oboro_generate.py --demo --profile "$PROF" "$@"
