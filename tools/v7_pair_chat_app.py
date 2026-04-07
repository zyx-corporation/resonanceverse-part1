"""
v7 ツール用 Streamlit: 通常 LLM チャット + 話者ペア Ω 解析。

起動（リポジトリルートで）::

    streamlit run tools/v7_pair_chat_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.v7_pair_w_trajectory import PairRelationLinear  # noqa: E402
from tools.v7_awai_chat_generate import (  # noqa: E402
    generate_completion_awai,
    init_awai_out_head_from_lm_head,
)
from tools.v7_llm_chat_generate import (  # noqa: E402
    build_generation_prompt,
    generate_completion,
)
from tools.v7_pair_chat_engine import (  # noqa: E402
    analyze_chat_messages_for_omega,
    analyze_transcript_turns,
    build_awai_integrated,
    load_awai_checkpoint,
    load_causal_lm,
    resolve_device,
    tokenizer_supports_offset_pool,
    unpack_model_bundle,
)

# よく使う HF ID（プルダウン）。任意 ID は下のテキスト欄に直接入力可能。
HF_MODEL_PRESETS: list[str] = [
    "gpt2",
    "rinna/japanese-gpt2-medium",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
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
if "hf_model_text" not in st.session_state:
    st.session_state.hf_model_text = "gpt2"
if "hf_model_preset_sb" not in st.session_state:
    st.session_state.hf_model_preset_sb = "gpt2"

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
    model_name = (st.session_state.get("hf_model_text") or "").strip() or "gpt2"
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
        st.session_state.v7_chat_pending_reply = False
        with st.spinner("ロード中…"):
            try:
                device = resolve_device(force_cpu=force_cpu)
                model, tok = load_causal_lm(model_name, device=device, cpu=force_cpu)
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
                hint = "（FastTokenizer: Ω 解析可）" if fast_ok else "（Ω 解析は Fast トークナ必須）"
                mode_h = " + Awai" if awai is not None else ""
                st.success(f"OK: {model_name}{mode_h} on {device} {hint}")
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
    if st.button("会話をクリア"):
        st.session_state.chat_messages = []
        st.session_state.chat_omega_result = None
        st.session_state.v7_chat_pending_reply = False
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

tab_chat, tab_v7 = st.tabs(["通常の LLM チャット", "v7 話者ペア / Ω 解析"])

with tab_chat:
    st.caption(
        "gpt2 / rinna 等の**ベース因果LM**は「チャット専用」ではなく、"
        "学習コーパス（小説・歌詞・台本・英日混在）の続きを書く挙動になります。"
        " 英語フレーズや意味のない繰り返しは**モデル本来の出力**です。"
        " 自然な対話には **Instruct 系**（chat_template 付き）を推奨します。"
    )
    bundle = st.session_state.model_bundle
    if bundle is None:
        st.info("左のサイドバーでモデルをロードしてください。")

    # 送信ランでは入力欄が先にクリアされるため、ユーザーメッセージを即 session に積んで
    # 一度 rerun し、次ランで履歴を描画してから生成する（質問が消えて見えるのを防ぐ）。
    if bundle is not None:
        if user_text := st.chat_input("メッセージを入力…"):
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_text}
            )
            st.session_state.v7_chat_pending_reply = True
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if bundle is not None and st.session_state.v7_chat_pending_reply:
        base_model, tok, device, awai_model = unpack_model_bundle(bundle)
        with st.chat_message("assistant"):
            with st.spinner("応答を生成中…"):
                try:
                    prompt = build_generation_prompt(
                        st.session_state.chat_messages, tok
                    )
                    if awai_model is not None:
                        reply = generate_completion_awai(
                            awai_model,
                            tok,
                            device,
                            prompt,
                            max_new_tokens=int(max_new_tokens),
                            temperature=float(temperature),
                            do_sample=bool(do_sample),
                            repetition_penalty=float(rep_pen),
                        )
                    else:
                        reply = generate_completion(
                            base_model,
                            tok,
                            device,
                            prompt,
                            max_new_tokens=int(max_new_tokens),
                            temperature=float(temperature),
                            do_sample=bool(do_sample),
                            no_repeat_ngram_size=int(no_repeat_ngram),
                            repetition_penalty=float(rep_pen),
                        )
                except Exception as e:
                    reply = f"（エラー）{e}"
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": reply}
        )
        st.session_state.v7_chat_pending_reply = False
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
