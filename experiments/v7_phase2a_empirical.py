"""
v7 Phase II-A: MRMP 実データの τ 掃引・R(τ)（事前登録 revision 1）。

シミュレーション本体は ``v7_phase2a_delay_sweep.py``。本モジュールは
コーパス代理（tau_star_corpus_proxy）の定義・集約のみを担当する。

確定仕様（s_asym_ab_ba_definition revision 1）
------------------------------------------
層 l の行ソフトマックス済み注意 A = A^(l)（単一ヘッド平均は呼び出し側）について、
窓内の話者ブロックを話者 ID（表示名）で分ける。

- block_a: 窓内テキストで話者 a（発話者）に帰属するトークン全インデックス
- block_b: 同様に話者 b（応答者）に帰属するトークン全インデックス

S_asym_ab(t) = || A[block_a → block_b] - A[block_b → block_a]^T ||_F

ここで A[block_a → block_b] は行 block_a・列 block_b の部分行列（形状 |a|×|b|）、
A[block_b → block_a] は |b|×|a|。減算は A[ia, ib] - A[ib, ia].T。

S_asym_ba(t) は block_a と block_b を入れ替えた同型の定義。
注: フロベニウスノルムの下で ||M||=||-M^T|| となるため、数値的には
S_asym_ab と S_asym_ba は一致しうる。主解析の積は事前登録式に従う。

対話間重み（等価）::

    R(τ) = (1/D) Σ_d (1/N_{d,τ}) Σ_{t∈T_{d,τ}} S_asym_ab_d(t) · S_asym_ba_d(t-τ)

T_{d,τ} は対話 d で t≥τ かつ両スカラーが有限なインデックス集合。
N_{d,τ} はその件数（対話内平均の分母）。

6 軸スコアは主解析に使わず、同一 τ グリッドの補助列として別出力（呼び出し側）。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# 事前登録 auxiliary_analysis.label_trajectory_delay_coherence の 6 軸（方向は _ab / _ba）
AUXILIARY_LABEL_AXES = (
    "trust",
    "authority",
    "proximity",
    "intent",
    "affect",
    "history",
)

# 事前登録 JSON と同一の固定キー（ログ用）
PREREG_SPAN_SPEC = {
    "span_unit": "speaker_id",
    "block_a": "utterer_all_tokens_in_window",
    "block_b": "responder_all_tokens_in_window",
    "weighting": "equal_per_dialogue",
    "tau_star_criterion": "dVar_sign_reversal",
    "data_source": "mrmp_windows.jsonl",
    "layer_index": -1,
}


def speaker_token_indices_mrmp_window(
    text: str,
    utterer_name: str,
    responder_name: str,
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """
    MRMP 窓テキスト（「話者名: 発話」改行区切り）から、話者表示名に帰属するトークン index を集める。
    utterer = speaker_tgt、responder = speaker_src（MRMP 行のペア規則に合わせる）。
    FastTokenizer の ``return_offsets_mapping=True`` が必要。
    """
    ut = (utterer_name or "").strip()
    res = (responder_name or "").strip()
    if not ut or not res:
        return [], []
    try:
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
    except Exception:
        return [], []
    off = enc.get("offset_mapping")
    if off is None:
        return [], []
    if hasattr(off, "tolist"):
        off = off.tolist()
    if len(off) > 0 and isinstance(off[0][0], (list, tuple)):
        off = off[0]
    u_set: set[int] = set()
    r_set: set[int] = set()
    pos = 0
    for line in text.splitlines():
        line_len = len(line)
        line_start = pos
        line_end = pos + line_len
        if ":" in line:
            spk = line.split(":", 1)[0].strip()
            for ti, span in enumerate(off):
                if ti >= len(off):
                    break
                cs, ce = int(span[0]), int(span[1])
                if ce <= cs:
                    continue
                mid = (cs + ce) // 2
                if line_start <= mid < line_end:
                    if spk == ut:
                        u_set.add(ti)
                    elif spk == res:
                        r_set.add(ti)
        pos += line_len + 1
    return sorted(u_set), sorted(r_set)


def pair_block_asymmetry_frobenius(
    A: np.ndarray,
    idx_a: Sequence[int],
    idx_b: Sequence[int],
) -> float:
    """
    || A[ia, ib] - A[ib, ia].T ||_F。idx は空でない想定。空なら nan。
    A は正方 (L, L)、行ソフトマックス済み。
    """
    ia = np.asarray(idx_a, dtype=np.int64)
    ib = np.asarray(idx_b, dtype=np.int64)
    if ia.size == 0 or ib.size == 0:
        return float("nan")
    m_ab = A[np.ix_(ia, ib)]
    m_ba = A[np.ix_(ib, ia)]
    diff = m_ab - m_ba.T
    return float(np.linalg.norm(diff, ord="fro"))


def dialogue_level_R_product_means(
    series_by_dialogue: list[dict[str, Any]],
    tau: int,
    *,
    value_key_ab: str = "s_asym_ab",
    value_key_ba: str = "s_asym_ba",
) -> list[dict[str, Any]]:
    """
    各対話について、有効な (t, t-τ) ペアに対する S_ab(t)·S_ba(t-τ) の平均を R_d とする。
    ペアが 1 つも無い対話は含めない。
    """
    out: list[dict[str, Any]] = []
    for d in series_by_dialogue:
        did = str(d.get("dialogue_id", ""))
        vals = d.get("values") or []
        by_t = {int(r["t"]): r for r in vals}
        acc: list[float] = []
        for t in sorted(by_t.keys()):
            if t < tau:
                continue
            cur = by_t[t]
            prev = by_t.get(t - tau)
            if prev is None:
                continue
            sab = cur.get(value_key_ab)
            sba0 = prev.get(value_key_ba)
            if not isinstance(sab, (int, float)) or not isinstance(sba0, (int, float)):
                continue
            if math.isnan(float(sab)) or math.isnan(float(sba0)):
                continue
            acc.append(float(sab) * float(sba0))
        if not acc:
            continue
        out.append({"dialogue_id": did, "R_d": sum(acc) / len(acc)})
    return out


def R_tau_equal_weight_per_dialogue(
    series_by_dialogue: list[dict[str, Any]],
    *,
    tau_max: int,
    value_key_ab: str = "s_asym_ab",
    value_key_ba: str = "s_asym_ba",
) -> list[dict[str, Any]]:
    """
    各対話の時系列は ``values`` に格納されたスカラー列とする。

    series_by_dialogue: 各要素は
      ``{"dialogue_id": str, "values": [{"t": int, value_key_ab: float, value_key_ba: float}, ...]}``
      t は窓 index（0..T_d-1 昇順）。

    戻り値: τ ごとの行 dict（required_log_fields_per_tau 用に拡張可能）。
    """
    rows: list[dict[str, Any]] = []
    for tau in range(0, tau_max + 1):
        per_d = dialogue_level_R_product_means(
            series_by_dialogue,
            tau,
            value_key_ab=value_key_ab,
            value_key_ba=value_key_ba,
        )
        contrib = [float(x["R_d"]) for x in per_d]
        d_count = len(contrib)
        r_mean = sum(contrib) / d_count if d_count else float("nan")
        r_var = (
            float(np.var(np.array(contrib, dtype=np.float64), ddof=1))
            if d_count > 1
            else 0.0
        )
        rows.append(
            {
                "tau": tau,
                "R_mean": r_mean,
                "R_var": r_var,
                "n": d_count,
                "data_source": PREREG_SPAN_SPEC["data_source"],
                "mode": "empirical",
                "tau_star_candidate": False,
            }
        )
    _mark_tau_star_candidates(rows)
    return rows


def series_has_auxiliary_label_scores(series_by_dialogue: list[dict[str, Any]]) -> bool:
    """いずれかの窓に、補助解析用の数値ラベル（6 軸いずれか）があれば True。"""
    for d in series_by_dialogue:
        for v in d.get("values") or []:
            for axis in AUXILIARY_LABEL_AXES:
                for suffix in ("_ab", "_ba"):
                    key = f"{axis}{suffix}"
                    x = v.get(key)
                    if isinstance(x, (int, float)) and not math.isnan(float(x)):
                        return True
    return False


def auxiliary_label_delay_coherence_by_axis(
    series_by_dialogue: list[dict[str, Any]],
    *,
    tau_max: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    主解析 R(τ) と同型に、各軸について
    mean_d mean_t (axis_ab(t) · axis_ba(t-τ)) を τ グリッドで返す。
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for axis in AUXILIARY_LABEL_AXES:
        kab = f"{axis}_ab"
        kba = f"{axis}_ba"
        out[axis] = R_tau_equal_weight_per_dialogue(
            series_by_dialogue,
            tau_max=tau_max,
            value_key_ab=kab,
            value_key_ba=kba,
        )
    return out


def _mark_tau_star_candidates(rows: list[dict[str, Any]]) -> None:
    """Var(R,τ) の隣接 τ 差分が正→負に転じる点を候補（事前登録: dVar_sign_reversal）。"""
    if len(rows) < 3:
        return
    vars_ = [float(r["R_var"]) for r in rows]
    dvar = [0.0]
    for i in range(1, len(vars_)):
        dvar.append(vars_[i] - vars_[i - 1])
    for i in range(1, len(dvar)):
        if dvar[i - 1] > 0.0 and dvar[i] < 0.0:
            rows[i]["tau_star_candidate"] = True


if __name__ == "__main__":
    # 合成系列で R(τ) の形だけ確認（本番は HF 注意行列＋トークンスパンが必要）
    demo = [
        {
            "dialogue_id": "D1",
            "values": [
                {"t": 0, "s_asym_ab": 1.0, "s_asym_ba": 0.5},
                {"t": 1, "s_asym_ab": 1.2, "s_asym_ba": 0.6},
                {"t": 2, "s_asym_ab": 1.1, "s_asym_ba": 0.55},
            ],
        },
        {
            "dialogue_id": "D2",
            "values": [
                {"t": 0, "s_asym_ab": 2.0, "s_asym_ba": 0.25},
                {"t": 1, "s_asym_ab": 2.1, "s_asym_ba": 0.3},
            ],
        },
    ]
    out = R_tau_equal_weight_per_dialogue(demo, tau_max=2)
    print("v7_phase2a_empirical_smoke", len(out), out[0].get("tau"))
