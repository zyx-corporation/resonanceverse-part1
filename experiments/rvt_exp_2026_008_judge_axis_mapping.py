"""
RVT-EXP-2026-008 の **6 軸（計画書ログ用）** と v7 LLM 審判 JSONL キーの対応。

教師あり ``M``（``train_head_axis_m_supervised_jsonl``）の **既定**は
**話者ブロックの ab 方向** 6 本のみ。ba を含める場合は別関数で拡張する。

v7 正本: ``v7_phase1a_pilot_jsonl.PILOT_KEYS``（12 キー・方向付き）。
"""

from __future__ import annotations

# 計画書・ログで使う軸ラベル（固定順）
RVT_PLAN_AXIS_NAMES: tuple[str, ...] = (
    "trust",
    "authority",
    "proximity",
    "intent",
    "affect",
    "history",
)

# 軸名 → v7 審判スコア（*_ab）。計画書「1 本／軸」に寄せた対応表。
V7_JUDGE_AB_KEY_BY_PLAN_AXIS: dict[str, str] = {
    "trust": "trust_ab",
    "authority": "authority_ab",
    "proximity": "proximity_ab",
    "intent": "intent_ab",
    "affect": "affect_ab",
    "history": "history_ab",
}

ORDERED_V7_AB_KEYS: tuple[str, ...] = tuple(
    V7_JUDGE_AB_KEY_BY_PLAN_AXIS[k] for k in RVT_PLAN_AXIS_NAMES
)

# 後方互換: head_axis_matrix が使う名前
SUPERVISED_TARGET_KEYS_AB: tuple[str, ...] = ORDERED_V7_AB_KEYS


def validate_mapping() -> None:
    """テスト用: 軸数とキー数の一致。"""
    if len(RVT_PLAN_AXIS_NAMES) != len(ORDERED_V7_AB_KEYS):
        raise RuntimeError("rvt judge axis mapping: length mismatch")
    if len(V7_JUDGE_AB_KEY_BY_PLAN_AXIS) != len(RVT_PLAN_AXIS_NAMES):
        raise RuntimeError("rvt judge axis dict: incomplete")
