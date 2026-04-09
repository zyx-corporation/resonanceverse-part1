#!/usr/bin/env bash
# v7 話者ペアチャット UI（Streamlit）。既定 SLM: Qwen2.5-3B-Instruct（RVT-IMPL-2026-008-SLM）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# 開発時: ファイル保存でブラウザが自動再実行（手動 Always rerun 不要）
exec streamlit run tools/v7_pair_chat_app.py --server.runOnSave=true "$@"
