#!/usr/bin/env bash
# v7 話者ペアチャット UI（Streamlit）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec streamlit run tools/v7_pair_chat_app.py "$@"
