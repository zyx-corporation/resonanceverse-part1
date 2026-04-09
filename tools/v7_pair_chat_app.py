"""
v7 ツール用 Streamlit: 通常 LLM チャット + 話者ペア Ω 解析。

既定モデルは RVT-IMPL-2026-008-SLM に沿い **Qwen2.5-3B-Instruct**（日本語対話・128K）。

起動（リポジトリルートで）::

    streamlit run tools/v7_pair_chat_app.py
    # リポジトリ直下の .streamlit/config.toml により保存時に自動再実行（runOnSave）

    # または
    bash tools/run_v7_pair_chat.sh
"""

from __future__ import annotations

import os

# HF tokenizers の Rust 並列経路が multiprocessing セマフォを使う。Streamlit では
# 応答待ち中に resource_tracker のセマフォ警告が出ることがあるため、単スレッドに固定する。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings

# PyTorch / 子ライブラリが残すセマフォを終了時に resource_tracker が報告する。
# チャット生成中に出ることがあり、rerun では消えない（終了時スキャンなので）ため抑止する。
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be \d+ leaked semaphore objects to clean up at shutdown",
    category=UserWarning,
    module=r"multiprocessing\.resource_tracker",
)

import contextlib
import inspect
import json
import sys
import uuid
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.v7_pair_w_trajectory import PairRelationLinear  # noqa: E402
from core.v9_rhythm_policy import V9RhythmConfig, V9RhythmState  # noqa: E402
from tools.v7_awai_chat_generate import (  # noqa: E402
    init_awai_out_head_from_lm_head,
)
from tools.rvt_w_asym_attention import (  # noqa: E402
    chat_window_text_from_messages,
    run_w_asym_attention_analysis,
)
from tools.v7_pair_chat_engine import (  # noqa: E402
    DEFAULT_RVT_SLM_INSTRUCT,
    analyze_chat_messages_for_omega,
    analyze_transcript_turns,
    build_awai_integrated,
    load_awai_checkpoint,
    load_causal_lm,
    resolve_device,
    tokenizer_supports_offset_pool,
    unpack_model_bundle,
)
from tools.v7_llm_chat_with_v9 import (  # noqa: E402
    append_v9_turn_jsonl,
    generate_assistant_turn_v7,
)

# よく使う HF ID（プルダウン）。任意 ID は下のテキスト欄に直接入力可能。
HF_MODEL_PRESETS: list[str] = [
    DEFAULT_RVT_SLM_INSTRUCT,
    "Qwen/Qwen2.5-7B-Instruct",
    "gpt2",
    "rinna/japanese-gpt2-medium",
    "tokyotech-llm/Swallow-7b-instruct-hf",
    "microsoft/Phi-3-mini-4k-instruct",
]


def _sync_hf_preset_to_text() -> None:
    pick = st.session_state.get("hf_model_preset_sb")
    if pick and pick != "（手入力のみ）":
        st.session_state.hf_model_text = pick


st.set_page_config(page_title="Resonanceverse チャット / v7 Ω", layout="wide")
st.title("Resonanceverse — LLM チャット & v7 話者ペア Ω")

if "turns" not in st.session_state:
    st.session_state.turns: list[tuple[str, str]] = []
if "model_name_loaded" not in st.session_state:
    st.session_state.model_name_loaded: str | None = None
if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages: list[dict[str, str]] = []
if "chat_omega_result" not in st.session_state:
    st.session_state.chat_omega_result = None
if "v7_chat_pending_reply" not in st.session_state:
    st.session_state.v7_chat_pending_reply = False
if "w_asym_viz_result" not in st.session_state:
    st.session_state.w_asym_viz_result = None
if "chat_system_summary" not in st.session_state:
    st.session_state.chat_system_summary = ""
if "hf_model_text" not in st.session_state:
    st.session_state.hf_model_text = DEFAULT_RVT_SLM_INSTRUCT
if "hf_model_preset_sb" not in st.session_state:
    st.session_state.hf_model_preset_sb = DEFAULT_RVT_SLM_INSTRUCT
if "v9_rhythm_state" not in st.session_state:
    st.session_state.v9_rhythm_state = V9RhythmState()


def _v9_rhythm_config_from_sidebar() -> V9RhythmConfig:
    _theta_unc = float(st.session_state.get("v9_theta_high_unc", 0.0))
    _obo_ht = int(st.session_state.get("v9_oboro_history_tail", 0))
    return V9RhythmConfig(
        omega_calm=float(st.session_state.get("v9_omega_calm", 0.15)),
        window_w=int(st.session_state.get("v9_window_w", 8)),
        theta_stuck=float(st.session_state.get("v9_theta_stuck", 0.6)),
        n_calm=int(st.session_state.get("v9_n_calm", 4)),
        n_cd=int(st.session_state.get("v9_n_cd", 6)),
        t_suspend_max=int(st.session_state.get("v9_t_suspend_max", 8000)),
        append_system_honesty_default=bool(
            st.session_state.get("v9_honesty_default", False)
        ),
        oboro_relax_max_new_tokens=bool(
            st.session_state.get("v9_oboro_relax_tokens", True)
        ),
        oboro_max_new_tokens_multiplier=float(
            st.session_state.get("v9_oboro_token_mult", 2.0)
        ),
        oboro_max_new_tokens_cap=int(
            st.session_state.get("v9_oboro_token_cap", 2048)
        ),
        oboro_history_tail_messages=(
            _obo_ht if _obo_ht > 0 else None
        ),
        theta_high_unc=_theta_unc if _theta_unc > 1e-9 else None,
        delta_suspend_ms_high_unc=int(
            st.session_state.get("v9_delta_high_unc", 400),
        ),
        phase2a_apply_logits_to_axis_suspend=bool(
            st.session_state.get("v9_p2a_axis_logits", False),
        ),
        phase2b_trajectory_stuck=bool(
            st.session_state.get("v9_p2b_traj", False),
        ),
        trajectory_epsilon_w=float(
            st.session_state.get("v9_traj_eps", 0.08),
        ),
        phase2c_enabled=bool(st.session_state.get("v9_p2c_on", False)),
        phase2c_silence_probability=float(
            st.session_state.get("v9_p2c_ps", 0.0),
        ),
        phase2c_refuse_probability=float(
            st.session_state.get("v9_p2c_pr", 0.0),
        ),
        phase2c_seed=int(st.session_state.get("v9_p2c_seed", 0)),
    )


with st.sidebar:
    st.header("モデル")
    st.selectbox(
        "プリセットから選ぶ",
        options=["（手入力のみ）"] + HF_MODEL_PRESETS,
        key="hf_model_preset_sb",
        on_change=_sync_hf_preset_to_text,
        help="選ぶと下のテキスト欄に上書きされます。リストに無い ID は手入力してください。",
    )
    st.text_input(
        "HF モデル ID（手入力・編集可）",
        key="hf_model_text",
    )
    model_name = (st.session_state.get("hf_model_text") or "").strip() or DEFAULT_RVT_SLM_INSTRUCT
    force_cpu = st.checkbox("CPU 強制", value=False)
    use_awai = st.checkbox(
        "Resonanceverse 統合（AwaiIntegratedSLM）",
        value=False,
        help="ベース HF 因果 LM に ResonantCore + out_head を重ねた経路で生成。"
        " generate() 非対応のため自前デコード（やや遅い）。",
    )
    awai_ckpt = st.text_input(
        "統合チェックポイント（任意）",
        value="",
        placeholder="/path/to/awai_state.pt",
        disabled=not use_awai,
        help="学習済み state_dict。空なら lm_head から out_head をコピーして初期化。",
    )
    awai_cultural = st.checkbox(
        "cultural modulation（Phase 1B）",
        value=False,
        disabled=not use_awai,
    )
    if st.button("モデルをロード / 再ロード"):
        st.session_state.model_bundle = None
        st.session_state.model_name_loaded = None
        st.session_state.last_result = None
        st.session_state.chat_messages = []
        st.session_state.chat_omega_result = None
        st.session_state.w_asym_viz_result = None
        st.session_state.v7_chat_pending_reply = False
        st.session_state.v9_rhythm_state = V9RhythmState()
        with st.spinner("ロード中…"):
            try:
                device = resolve_device(force_cpu=force_cpu)
                model, tok = load_causal_lm(
                    model_name,
                    device=device,
                    cpu=force_cpu,
                    attn_implementation="eager",
                )
                awai = None
                if use_awai:
                    awai = build_awai_integrated(
                        model,
                        device,
                        cultural_modulation=bool(awai_cultural),
                    )
                    ckpt = (awai_ckpt or "").strip()
                    if ckpt:
                        p = Path(ckpt)
                        if p.is_file():
                            load_awai_checkpoint(awai, str(p), map_location=device)
                        else:
                            st.warning(
                                f"チェックポイントが見つかりません: {ckpt}。"
                                " lm_head から out_head を初期化します。"
                            )
                            if not init_awai_out_head_from_lm_head(awai):
                                st.warning(
                                    "lm_head の形状が合わず out_head をコピーできませんでした。"
                                )
                    else:
                        if not init_awai_out_head_from_lm_head(awai):
                            st.warning(
                                "lm_head から out_head をコピーできませんでした（出力は不安定になりやすい）。"
                            )
                st.session_state.model_bundle = (model, tok, device, awai)
                st.session_state.model_name_loaded = model_name
                fast_ok, _ = tokenizer_supports_offset_pool(tok)
                hint = "（FastTokenizer: Ω / W_asym 可）" if fast_ok else "（Ω・W_asym は Fast トークナ必須）"
                mode_h = " + Awai" if awai is not None else ""
                st.success(
                    f"OK: {model_name}{mode_h} on {device} {hint}"
                    " · attentions は eager で取得"
                )
            except Exception as e:
                st.exception(e)

    st.divider()
    st.subheader("通常チャット（生成）")
    max_new_tokens = st.slider("max_new_tokens", 16, 1024, 128, 16)
    temperature = st.slider("temperature", 0.01, 1.5, 0.7, 0.05)
    do_sample = st.checkbox("サンプリング", value=True)
    no_repeat_ngram = st.slider(
        "no_repeat_ngram_size（0=オフ）",
        0,
        8,
        4,
        help="HF 因果 LM の generate のみ。Awai 統合時は無効（repetition_penalty のみ）。",
    )
    rep_pen = st.slider("repetition_penalty", 1.0, 1.5, 1.25, 0.05)
    st.divider()
    st.subheader("v9.0 リズム（実験）")
    st.checkbox(
        "v9 リズムを有効化",
        value=False,
        key="v9_rhythm_enabled",
        help="応答直前に Ω を再計算し、固着に応じた遅延（Suspend）や朧フラグを付与。",
    )
    st.text_input(
        "JSONL ログパス（空でオフ）",
        key="v9_rhythm_jsonl_path",
        placeholder="experiments/logs/v9_rhythm.jsonl",
    )
    with st.expander("v9 閾値（上級）"):
        st.slider("Ω_calm（凪閾値）", 0.01, 0.5, 0.15, 0.01, key="v9_omega_calm")
        st.number_input("固着窓 W", min_value=1, max_value=64, value=8, step=1, key="v9_window_w")
        st.slider("θ_stuck（朧ゲート）", 0.1, 1.0, 0.6, 0.05, key="v9_theta_stuck")
        st.number_input("凪連続 N_calm", min_value=1, max_value=32, value=4, step=1, key="v9_n_calm")
        st.number_input("朧クールダウン", min_value=0, max_value=32, value=6, step=1, key="v9_n_cd")
        st.number_input(
            "T_suspend_max（ms）",
            min_value=0,
            max_value=60_000,
            value=8000,
            step=500,
            key="v9_t_suspend_max",
        )
        st.checkbox(
            "既定で誠実さ一文を system に付与",
            value=False,
            key="v9_honesty_default",
        )
        st.checkbox(
            "朧（oboro）時に max_new_tokens を緩和（M3）",
            value=True,
            key="v9_oboro_relax_tokens",
        )
        st.slider(
            "朧倍率（× 基準 max_new_tokens）",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.25,
            key="v9_oboro_token_mult",
        )
        st.number_input(
            "朧上限 cap（トークン）",
            min_value=16,
            max_value=8192,
            value=2048,
            step=16,
            key="v9_oboro_token_cap",
        )
        st.number_input(
            "朧時・生成プロンプト末尾メッセージ数（0=スライスなし）",
            min_value=0,
            max_value=256,
            value=0,
            step=1,
            key="v9_oboro_history_tail",
            help="朧バースト時のみ、生成に使う chat メッセージを末尾 N 件に限定（Ω 計算は全履歴のまま）。",
        )
        st.divider()
        st.markdown("**M4（実験・任意）**")
        st.checkbox(
            "2a: decide に logits 分散を反映（API/上級・既定オフ）",
            value=False,
            key="v9_p2a_axis_logits",
        )
        st.number_input(
            "logits 閾値 θ_high_unc（0=未使用・軸 Suspend 加算のみ）",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            key="v9_theta_high_unc",
        )
        st.number_input(
            "閾値超過時の追加 ms（delta）",
            min_value=0,
            max_value=30_000,
            value=400,
            step=50,
            key="v9_delta_high_unc",
        )
        st.checkbox(
            "2b: w_ij 軌道固着を S_stuck に max 統合",
            value=False,
            key="v9_p2b_traj",
        )
        st.slider(
            "軌道 ε（L2）",
            min_value=0.001,
            max_value=1.0,
            value=0.08,
            step=0.001,
            format="%.3f",
            key="v9_traj_eps",
        )
        st.checkbox(
            "2c: 朧時に沈黙/拒否を抽選（実験・注意）",
            value=False,
            key="v9_p2c_on",
        )
        st.caption(
            "理論上は詳細展望 §6.1: 実装・実験層の摂動であり、自律的境界の同一物ではありません。"
        )
        st.slider("2c: 沈黙確率", 0.0, 1.0, 0.0, 0.05, key="v9_p2c_ps")
        st.slider("2c: 拒否確率", 0.0, 1.0, 0.0, 0.05, key="v9_p2c_pr")
        st.number_input("2c: seed", min_value=0, max_value=2**31 - 1, value=0, step=1, key="v9_p2c_seed")
    st.caption(
        "有効時、Fast トークナ必須で Ω を取れない場合は axis_signal_source=omega_unavailable としてログされます。"
    )
    st.divider()
    st.subheader("要約メモリ（system）")
    st.text_area(
        "要約・長期メモ",
        key="chat_system_summary",
        height=120,
        placeholder=(
            "空欄のときは注入しません。会話の要約や常に守らせたい前提を書くと、"
            "毎回の生成で先頭の system としてモデルに渡ります（チャット欄には表示されません）。"
        ),
        help="Qwen 等は chat_template 経由で system を解釈します。chat_template 無しモデルでは平文の System: 行になります。",
    )
    if st.button("会話をクリア"):
        st.session_state.chat_messages = []
        st.session_state.chat_omega_result = None
        st.session_state.w_asym_viz_result = None
        st.session_state.v7_chat_pending_reply = False
        st.session_state.v9_rhythm_state = V9RhythmState()
        st.rerun()

    st.divider()
    st.subheader("v7 Ω 解析")
    layer_index = st.number_input(
        "隠れ層 index（-1=最終）",
        min_value=-1,
        value=-1,
        step=1,
    )
    d = st.number_input("d（関係ベクトル次元）", min_value=2, value=6, step=1)
    delay_tau = st.number_input("遅延 τ（ターン）", min_value=0, value=0, step=1)
    use_linear = st.checkbox(
        "PairRelationLinear（2H→d）",
        value=False,
        help="未学習のランダム重み。既定は concat_truncate。",
    )
    st.caption("Ω 入力窓（チャット / v7 タブ共通）")
    omega_window_mode = st.radio(
        "窓の作り方",
        options=["cumulative", "last_n"],
        format_func=lambda x: (
            "累積（先頭から全ターン）"
            if x == "cumulative"
            else "直近 N ターンのみ"
        ),
        index=0,
    )
    omega_last_n = st.number_input(
        "N（直近モード時・2 以上）",
        min_value=2,
        max_value=128,
        value=8,
        step=1,
    )

tab_chat, tab_w, tab_v7 = st.tabs(
    ["通常の LLM チャット", "W_asym / 注意可視化", "v7 話者ペア / Ω 解析"]
)

with tab_chat:
    st.caption(
        "既定は **Qwen2.5-3B-Instruct**（RVT-IMPL-2026-008-SLM）。chat_template により日本語の多ターン対話に適しています。"
        " gpt2 / rinna 等のベース因果 LM はチャット専用ではなく続き生成寄りです。"
        " Instruct 以外を選ぶ場合は出力の癖に注意してください。"
    )
    bundle = st.session_state.model_bundle
    if bundle is None:
        st.info("左のサイドバーでモデルをロードしてください。")

    # 会話が長いとき履歴だけ縦スクロール（入力は下の chat_input に固定）
    _chat_history_px = 520
    _hist_msgs = st.session_state.chat_messages
    _hist_pending = bundle is not None and st.session_state.v7_chat_pending_reply
    if _hist_msgs or _hist_pending:
        try:
            _box_kw: dict[str, object] = {"height": _chat_history_px}
            if "autoscroll" in inspect.signature(st.container).parameters:
                _box_kw["autoscroll"] = True
            _history_box = st.container(**_box_kw)
        except TypeError:
            _history_box = contextlib.nullcontext()
        with _history_box:
            for msg in _hist_msgs:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            if _hist_pending:
                base_model, tok, device, awai_model = unpack_model_bundle(bundle)
                with st.chat_message("assistant"):
                    with st.spinner("応答を生成中…"):
                        v9_payload: dict | None = None
                        v9_on = bool(st.session_state.get("v9_rhythm_enabled"))
                        try:
                            v9_st = st.session_state.v9_rhythm_state
                            if not isinstance(v9_st, V9RhythmState):
                                v9_st = V9RhythmState()
                                st.session_state.v9_rhythm_state = v9_st
                            v9_cfg = _v9_rhythm_config_from_sidebar()
                            proj_v9: PairRelationLinear | None = None
                            if use_linear:
                                _hs = int(base_model.config.hidden_size)
                                proj_v9 = PairRelationLinear(_hs, int(d)).to(device)
                                proj_v9.eval()
                            reply, v9_st_new, v9_payload = (
                                generate_assistant_turn_v7(
                                    chat_messages=st.session_state.chat_messages,
                                    system_summary=st.session_state.get(
                                        "chat_system_summary"
                                    ),
                                    base_model=base_model,
                                    tokenizer=tok,
                                    device=device,
                                    awai_model=awai_model,
                                    v9_enabled=v9_on,
                                    v9_state=v9_st,
                                    v9_cfg=v9_cfg,
                                    pair_projector=proj_v9,
                                    layer_index=int(layer_index),
                                    d=int(d),
                                    delay_tau=int(delay_tau),
                                    omega_window_mode=str(omega_window_mode),
                                    omega_last_n=int(omega_last_n),
                                    model_name=str(
                                        st.session_state.model_name_loaded or ""
                                    ),
                                    max_new_tokens=int(max_new_tokens),
                                    temperature=float(temperature),
                                    do_sample=bool(do_sample),
                                    no_repeat_ngram_size=int(no_repeat_ngram),
                                    repetition_penalty=float(rep_pen),
                                )
                            )
                            st.session_state.v9_rhythm_state = v9_st_new
                        except Exception as e:
                            reply = f"（エラー）{e}"
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": reply}
                )
                _log_p = (st.session_state.get("v9_rhythm_jsonl_path") or "").strip()
                _sid = st.session_state.setdefault(
                    "v9_session_id",
                    str(uuid.uuid4()),
                )
                _n_asst = sum(
                    1
                    for _m in st.session_state.chat_messages
                    if _m.get("role") == "assistant"
                )
                append_v9_turn_jsonl(
                    _log_p,
                    session_id=_sid,
                    turn_index=int(_n_asst - 1),
                    reply=reply,
                    v9_enabled=v9_on,
                    v9_payload=v9_payload,
                )
                st.session_state.v7_chat_pending_reply = False
                st.rerun()

    # 送信ランでは入力欄が先にクリアされるため、ユーザーメッセージを即 session に積んで
    # 一度 rerun し、次ランで履歴を描画してから生成する（質問が消えて見えるのを防ぐ）。
    if bundle is not None:
        if user_text := st.chat_input("メッセージを入力…"):
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_text}
            )
            st.session_state.v7_chat_pending_reply = True
            st.rerun()

    with st.expander("チャット履歴で Ω を計算（要 FastTokenizer）"):
        st.caption(
            "User / Assistant を話者名として MRMP 同型の窓にし、"
            "サイドバーの窓設定（累積 / 直近 N）を適用します。"
        )
        if len(st.session_state.chat_messages) < 2:
            st.info("ユーザーとアシスタントのメッセージが合計 2 件以上必要です。")
        if st.button("この履歴で Ω 系列を計算", key="btn_chat_omega"):
            cb = st.session_state.model_bundle
            if cb is None:
                st.error("モデルをロードしてください。")
            else:
                model_o, tok_o, dev_o, _ = unpack_model_bundle(cb)
                _ok, _fm = tokenizer_supports_offset_pool(tok_o)
                if not _ok:
                    st.error(_fm)
                else:
                    h_o = int(model_o.config.hidden_size)
                    proj_o: PairRelationLinear | None = None
                    if use_linear:
                        proj_o = PairRelationLinear(h_o, int(d)).to(dev_o)
                        proj_o.eval()
                    with st.spinner("Ω 計算中…"):
                        try:
                            st.session_state.chat_omega_result = (
                                analyze_chat_messages_for_omega(
                                    st.session_state.chat_messages,
                                    model=model_o,
                                    tokenizer=tok_o,
                                    device=dev_o,
                                    layer_index=int(layer_index),
                                    d=int(d),
                                    delay_tau=int(delay_tau),
                                    pair_projector=proj_o,
                                    window_mode=str(omega_window_mode),
                                    last_n=int(omega_last_n),
                                )
                            )
                        except Exception as e:
                            st.session_state.chat_omega_result = {"error": str(e)}
                    st.rerun()
        cor = st.session_state.chat_omega_result
        if cor:
            if cor.get("error"):
                st.error(cor["error"])
            else:
                st.success(
                    f"source={cor.get('source')} "
                    f"window={cor.get('window_mode')} "
                    f"steps={cor.get('n_pair_steps')}"
                )
                om_c = cor.get("omega") or []
                if om_c:
                    df_c = pd.DataFrame(
                        {
                            "pair_step": list(range(len(om_c))),
                            "omega": om_c,
                        }
                    )
                    st.line_chart(df_c.set_index("pair_step")[["omega"]])
                with st.expander("チャット Ω メタデータ"):
                    st.json(
                        {
                            k: v
                            for k, v in cor.items()
                            if k not in ("w_ij", "w_ji")
                        }
                    )
                raw_c = json.dumps(cor, ensure_ascii=False, indent=2)
                st.download_button(
                    "チャット Ω JSON",
                    data=raw_c,
                    file_name="v7_chat_omega.json",
                    mime="application/json",
                    key="dl_chat_o",
                )

with tab_w:
    st.caption(
        "RVT-IMPL-2026-008 §4.1: User / Assistant ブロック間で "
        "S = A_AB − A_BA^T（層ごとに全ヘッドの ‖S‖_F 平均）。"
        " 窓はサイドバーの Ω 設定（累積 / 直近 N）と同一。"
        " MRMP 同型の「User:」「Assistant:」行テキストでトークン境界を取ります。"
    )
    bundle_w = st.session_state.model_bundle
    if bundle_w is None:
        st.info("サイドバーでモデルをロードしてください。")
    else:
        base_w, tok_w, dev_w, awai_w = unpack_model_bundle(bundle_w)
        fast_w, msg_w = tokenizer_supports_offset_pool(tok_w)
        if not fast_w:
            st.error(msg_w)
        else:
            model_w = awai_w.base_model if awai_w is not None else base_w
            if awai_w is not None:
                st.warning(
                    "注意・W_asym はベース因果 LM（Awai 下層）に対して計算しています。"
                )
            heat_layer_w = st.number_input(
                "注意ヒートマップの層 index（-1=最終）",
                min_value=-1,
                value=-1,
                step=1,
                key="w_heat_layer",
            )
            heat_max_w = st.slider(
                "ヒートマップの最大辺（超えると間引き）",
                min_value=64,
                max_value=320,
                value=160,
                step=16,
                key="w_heat_max",
            )
            log_scale_w = st.checkbox(
                "ヒートマップを log1p 表示",
                value=False,
                key="w_log_scale",
            )
            if st.button("チャット履歴で W_asym・注意を計算", type="primary", key="btn_w_asym"):
                msgs_w = st.session_state.chat_messages
                if len(msgs_w) < 2:
                    st.warning(
                        "ユーザーとアシスタントのメッセージを合わせて 2 件以上必要です。"
                    )
                else:
                    wtxt = chat_window_text_from_messages(
                        msgs_w,
                        window_mode=str(omega_window_mode),
                        last_n=int(omega_last_n),
                    )
                    with st.spinner("前向き実行中…"):
                        try:
                            st.session_state.w_asym_viz_result = (
                                run_w_asym_attention_analysis(
                                    model=model_w,
                                    tokenizer=tok_w,
                                    device=dev_w,
                                    window_text=wtxt,
                                    layer_for_heatmap=int(heat_layer_w),
                                    heatmap_max_side=int(heat_max_w),
                                )
                            )
                        except Exception as e:
                            st.session_state.w_asym_viz_result = {
                                "schema_version": "rvt_w_asym_attention.v1",
                                "error": str(e),
                            }
                    st.rerun()

    wr = st.session_state.w_asym_viz_result
    if wr:
        if wr.get("error"):
            st.error(wr["error"])
        else:
            st.success(
                f"tokens={wr['n_tokens']} · layers={wr['n_layers']} · "
                f"heatmap_layer={wr['layer_heatmap']}"
            )
            c_m1, c_m2 = st.columns(2)
            ua = wr.get("mean_attn_user_to_assistant")
            au = wr.get("mean_attn_assistant_to_user")
            with c_m1:
                st.metric(
                    "平均注意 User→Asst（選択層）",
                    f"{float(ua):.5f}" if ua is not None and not np.isnan(ua) else "n/a",
                )
            with c_m2:
                st.metric(
                    "平均注意 Asst→User（選択層）",
                    f"{float(au):.5f}" if au is not None and not np.isnan(au) else "n/a",
                )
            frob = wr.get("w_asym_mean_frobenius_per_layer") or []
            if frob:
                df_w = pd.DataFrame(
                    {
                        "layer": list(range(len(frob))),
                        "mean_Frobenius_W_asym": frob,
                    }
                )
                st.subheader("層別 ‖W_asym‖_F（ヘッド平均）")
                st.line_chart(
                    df_w.set_index("layer")[["mean_Frobenius_W_asym"]]
                )
            mat = wr.get("attention_heatmap_matrix")
            if mat is not None:
                st.subheader("注意ヒートマップ（選択層・ヘッド平均・間引き済み）")
                arr = np.asarray(mat, dtype=np.float64)
                plot_m = (
                    np.log1p(arr)
                    if st.session_state.get("w_log_scale", False)
                    else arr
                )
                fig, ax = plt.subplots(figsize=(6.5, 5.2))
                im = ax.imshow(
                    plot_m,
                    aspect="auto",
                    cmap="magma",
                    interpolation="nearest",
                )
                ax.set_xlabel("key（列・間引き）")
                ax.set_ylabel("query（行・間引き）")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with st.expander("窓テキスト先頭"):
                st.text(wr.get("window_text_preview", ""))
            dl_payload = {
                k: v
                for k, v in wr.items()
                if k != "attention_heatmap_matrix"
            }
            if mat is not None:
                dl_payload["attention_heatmap_matrix_shape"] = list(
                    np.asarray(mat).shape
                )
            st.download_button(
                "W_asym メタ JSON",
                data=json.dumps(dl_payload, ensure_ascii=False, indent=2),
                file_name="rvt_w_asym_chat.json",
                mime="application/json",
                key="dl_w_asym",
            )

with tab_v7:
    st.caption(
        "MRMP 同型: 現在話者=tgt・直前話者=src。"
        " 窓をトークナイズし話者ブロック平均プール → w_ij / w_ji → Ω。"
    )
    fast_ok = False
    if st.session_state.model_bundle:
        _b, tok_v7, _d, _ = unpack_model_bundle(st.session_state.model_bundle)
        fast_ok, fast_msg = tokenizer_supports_offset_pool(tok_v7)
        if not fast_ok:
            st.warning(fast_msg)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        speaker = st.text_input("話者名", value="A", key="spk")
    with c2:
        utterance = st.text_input(
            "発話（話者名は含めない）",
            value="こんにちは",
            key="utt",
        )
    with c3:
        st.write("")
        st.write("")
        if st.button("ターンを追加"):
            if not speaker.strip() or not utterance.strip():
                st.warning("話者名と発話を入力してください。")
            else:
                st.session_state.turns.append((speaker.strip(), utterance.strip()))
                st.session_state.last_result = None
                st.rerun()

    if st.button("トランスクリプトをクリア", key="clr_tr"):
        st.session_state.turns = []
        st.session_state.last_result = None
        st.rerun()

    st.subheader("現在のトランスクリプト")
    if not st.session_state.turns:
        st.info("ターンを 2 つ以上追加すると解析できます。")
    else:
        for i, (sp, tx) in enumerate(st.session_state.turns):
            st.text(f"{i + 1}. {sp}: {tx}")

    run_analyze = st.button("Ω 系列を計算", type="primary", key="run_o")
    if run_analyze:
        bundle = st.session_state.model_bundle
        if bundle is None:
            st.error("サイドバーでモデルをロードしてください。")
        elif not fast_ok:
            st.error("Ω 解析には FastTokenizer（offset_mapping）が必要です。")
        else:
            model, tok, device, _ = unpack_model_bundle(bundle)
            h = int(model.config.hidden_size)
            proj: PairRelationLinear | None = None
            if use_linear:
                proj = PairRelationLinear(h, int(d)).to(device)
                proj.eval()
            with st.spinner("前向き実行中…"):
                try:
                    res = analyze_transcript_turns(
                        st.session_state.turns,
                        model=model,
                        tokenizer=tok,
                        device=device,
                        layer_index=int(layer_index),
                        d=int(d),
                        delay_tau=int(delay_tau),
                        pair_projector=proj,
                        window_mode=str(omega_window_mode),
                        last_n=int(omega_last_n),
                    )
                    st.session_state.last_result = res
                except Exception as e:
                    st.exception(e)
                    st.session_state.last_result = {"error": str(e)}

    res = st.session_state.last_result
    if res:
        if res.get("error"):
            st.error(res["error"])
        else:
            st.success(
                f"ステップ数 {res.get('n_pair_steps', 0)} "
                f"（map: {res.get('pair_map')}） "
                f"窓: {res.get('window_mode')}"
            )
            om = res.get("omega") or []
            if om:
                df = pd.DataFrame(
                    {
                        "pair_step": list(range(len(om))),
                        "t_turn": [r["t"] for r in res.get("per_step", [])][
                            : len(om)
                        ],
                        "omega": om,
                    }
                )
                st.line_chart(df.set_index("pair_step")[["omega"]])
            with st.expander("メタデータ（JSON）"):
                st.json(
                    {
                        k: v
                        for k, v in res.items()
                        if k not in ("w_ij", "w_ji")
                    }
                )
            with st.expander("w_ij / w_ji（ベクトル）"):
                st.json({"w_ij": res.get("w_ij"), "w_ji": res.get("w_ji")})
            raw = json.dumps(res, ensure_ascii=False, indent=2)
            st.download_button(
                "結果 JSON をダウンロード",
                data=raw,
                file_name="v7_pair_chat_analysis.json",
                mime="application/json",
                key="dl_json",
            )
