---
title: v7 日本語ローカル SLM 運用プラン（Apple M3 Max / 128GB）
created: 2026-03-28
language: ja
document_type: planning
---

# v7 日本語ローカル SLM 運用プラン（Apple M3 Max / 128GB）

## 1. 目的と前提

- **目的**: MRMP 等の**日本語コーパス**上で、設計書 v7 に沿う実証（注意統計・審判スコア）を**同一マシンで再現可能**にする。
- **ハード**: MacBook Pro、**Apple M3 Max**、**128GB** ユニファイドメモリ、**ローカル運転**。
- **方針**: 前回合意のとおり **役割分離**（注意系＝軽量、審判系＝7B 指示系）を採用する。

**正本（理論・実験）**: [Resonanceverse v7.0 実証実験設計書](../v7/Resonanceverse_v7.0_Experimental_Design.md)  
**既存パイプライン索引**: [experiments/README.md](../../experiments/README.md)

---

## 2. 採用モデル（固定案）

| 役割 | 推奨モデル（HF） | 想定用途 | 備考 |
|------|------------------|----------|------|
| **B. 審判・指示系（主）** | `tokyotech-llm/Swallow-7b-instruct-hf` | 6 軸スコア、将来の III-A 代理スコア、長文ルーブリック | 日本語指示の第一選択。 |
| **B′. 審判の比較用（任意）** | `Qwen/Qwen2.5-7B-Instruct` | 同一プロンプトでの感度・ブレ比較 | 多言語強；主張は「補助」に留める。 |
| **A. 注意・軽量前向き（第一歩）** | `rinna/japanese-gpt2-medium` | `v7_phase2a_empirical_run`・`v7_phase1a_pilot_jsonl` の**日本語基準線** | GPT2 系で既存コードとの相性確認が速い。 |
| **A′. 注意系の拡張（任意）** | Swallow **base**（instruct ではない）や同等の**日本語因果 LM** | rinna では長文・表現が足りない場合 | **アーキテクチャ変更**に伴い `attn_implementation`・層数のスモーク必須。 |

**13B 化**: 余力検証として `Swallow-13b` 系を **審判 B のみ**に限定して試す（注意全窓スキャンとの同時運転は時間見積り後）。

---

## 3. ソフトウェアスタック

| 項目 | 推奨 | メモ |
|------|------|------|
| Python | リポジトリ既存（3.11 想定） | `requirements.txt` 準拠。 |
| PyTorch | **MPS 対応ビルド** | 注意前向きは `device=mps` を検証。 |
| transformers | 既存以上 | モデルカードに従い `trust_remote_code` が要る場合は**理由をログに記載**。 |
| 推論の第二経路（任意） | **MLX / mlx-lm** | ボトルネック時のみ。再現性は **別マニフェスト**で管理。 |

---

## 4. フェーズ計画

### Phase 0 — 環境・取得（0.5–1 日）

- [x] 仮想環境で `pip install -r requirements.txt`、GPU 相当として **MPS 利用可能**を確認（[`v7_local_env_check.py`](../../experiments/v7_local_env_check.py) で JSON 出力）。
- [ ] Hugging Face にログインが必要な重みがあれば `HF_TOKEN` を設定（**git に書かない**）。
- [ ] キャッシュディレクトリの場所を決め、**ディスク容量**（7B×2 + 小モデルで数十 GB 級）を確保。

**完了基準**: `python -c "import torch; print(torch.backends.mps.is_available())"` が True（または計画した CPU フォールバック方針を文書化）。

### Phase 1 — 注意パイプラインの日本語化（1–3 日）

- [x] **`rinna/japanese-gpt2-medium`** で `v7_phase2a_empirical_run.py` を **極小**（[`run_local_slm_phase1_smoke.sh`](../../experiments/run_local_slm_phase1_smoke.sh)）。
- [x] 出力 JSON に **`model`・`layer_index`・`inference_device`** が意図どおり入ることを確認（生成ログで確認）。
- [x] 既存の **gpt2 英語ベースライン**と**同一 JSONL スライス**で比較表（[`run_local_slm_phase1_compare_en_ja.sh`](../../experiments/run_local_slm_phase1_compare_en_ja.sh) → `v7_phase2a_compare_runs`）。数値の優劣判断はしない。

**完了基準**: `*_with_contrib.json` がエラーなく生成され、`v7_phase2a_bundle_validate.py`（非 strict）が通る。→ ポインタ [`baselines/v7_local_slm_phase1_smoke_bundle_v1.json`](../../experiments/baselines/v7_local_slm_phase1_smoke_bundle_v1.json)。

### Phase 2 — 審判のローカル化（中項目・優先度高）

現状 [`v7_phase1a_llm_judge_six_axes.py`](../../experiments/v7_phase1a_llm_judge_six_axes.py) の本番経路は **`--provider openai`** 前提。

- [x] **設計**: `provider` **`hf_local`**（Swallow-7B-Instruct・[`run_local_slm_phase2_smoke.sh`](../../experiments/run_local_slm_phase2_smoke.sh)）。
- [x] **プロンプト**を [`experiments/prompts/v7_llm_judge_prompt_v1.json`](../../experiments/prompts/v7_llm_judge_prompt_v1.json) に集約。スタブ紐づけ: [`v7_local_slm_llm_judge_prereg_stub_v1.json`](v7_local_slm_llm_judge_prereg_stub_v1.json)。**モデル revision** は実行時 `--hf-revision` / `REVISION` で固定（補遺記載）。
- [x] **決定論**: `temperature=0`・`seed` 固定。生成失敗時のリトライはスクリプト既定（指数バックオフ）。
- **スキップ（方針）**: OpenAI 審判とローカル審判の **一致傾向ログ**は実施しない。主結果・補遺の必須項目としない。
- **代替（実装済み・任意）**: **異なる SLM 同士**の一致傾向は [`v7_llm_judge_slm_pair_agreement.py`](../../experiments/v7_llm_judge_slm_pair_agreement.py) で集計（下記 §10）。OpenAI 比較は引き続き不要。

**完了基準**: ローカルだけで **審判付き JSONL** が再現生成できる（コマンド 1 本＋環境変数で説明可能）。→ 一括: [`run_local_slm_smoke_all.sh`](../../experiments/run_local_slm_smoke_all.sh)。

### Phase 3 — 本番スケールのバッチ（データ次第）

- [x] MRMP 窓に **OFFSET・MAX_ROWS・MODEL**（[`run_phase2a_mrmp_tau.sh`](../../experiments/run_phase2a_mrmp_tau.sh) 拡張）および [`run_local_slm_phase3_mrmp_chunk.sh`](../../experiments/run_local_slm_phase3_mrmp_chunk.sh）でチャンク実行。
- [x] 成果物ごとの **`v7_phase2a_repro_manifest.py`** 運用を **§11** に確定（`WRITE_REPRO_MANIFEST=1`・`BUNDLE_JSON`・任意 `REPRO_MANIFEST_STRICT=1`）。
- [x] オペレータ向け草案: [`baselines/v7_local_slm_phase3_operator_bundle_v1.json`](../../experiments/baselines/v7_local_slm_phase3_operator_bundle_v1.json)（`note_ja` に審判チャンク [`run_mrmp_llm_judge_chunks_hf_local.sh`](../../experiments/run_mrmp_llm_judge_chunks_hf_local.sh)）。

**完了基準**: 論文・補遺に載せられる **再現コマンド＋環境メタ**が揃う。

### Phase 4 — 拡張・任意

- [x] **Swallow 7B vs 13B** 審判 A/B（実走は任意）: [`run_local_slm_phase4_swallow_7b_13b.sh`](../../experiments/run_local_slm_phase4_swallow_7b_13b.sh) → **§10** 集計。品質と時間はログに記録し主結果にしない。
- [ ] **Qwen2.5-7B-Instruct** 等との審判ブレ（実走は任意）: [`run_local_slm_phase4_judge_pair.sh`](../../experiments/run_local_slm_phase4_judge_pair.sh)（既定 A=Swallow7B・B=Qwen）。探索のみ・主結果にしない。
- [ ] **MLX** への移行評価: **§13** の手順・合格基準。リポジトリに MLX 推論コードは未同梱（別環境・別マニフェスト）。

---

## 5. 再現性・記録（必須メタ）

各実行 JSON に以下を含める（既存慣習を拡張）:

- `model_name`（HF ID）、可能なら **`revision`（git SHA）**
- `transformers`・`torch`・Python のバージョン（[`v7_experiment_meta.py`](../../experiments/v7_experiment_meta.py) が使えるなら流用）
- `inference_device`（`mps` / `cpu`）
- 審判経路なら **`provider`**・`temperature`・プロンプト版 ID

---

## 6. リスクと緩和

| リスク | 緩和 |
|--------|------|
| MPS での dtype / アテンションエラー | 先に **1 窓スモーク**；ダメなら該当層のみ CPU フォールバックを記録。 |
| 審判 JSON の破損 | **jsonschema 検証**・失敗行のログ追記・`--resume` 運用。 |
| 推定と審判の同一モデルバイアス | **B と A でモデル分離**（本プランの前提）；論文では「代理指標」と明記。 |
| 時間・電力 | 全窓は **バッチ＋夜間**；13B は審判のみに限定。 |
| Swallow の HF 取得がタイムアウト | 再試行・別回線・`HF_HOME` をローカル SSD に。`run_local_slm_phase2_smoke.sh` は初回のみ大きい。 |

---

## 7. 成果物チェックリスト（マイルストーン）

- [x] Phase 1: 日本語注意スモークログ 1 本（`local_slm_phase1_smoke_with_contrib.json`）＋ [`v7_local_slm_phase1_smoke_bundle_v1.json`](../../experiments/baselines/v7_local_slm_phase1_smoke_bundle_v1.json)
- [x] 英語 gpt2 との並列表（[`run_local_slm_phase1_compare_en_ja.sh`](../../experiments/run_local_slm_phase1_compare_en_ja.sh)、ログは `experiments/logs/`）
- [x] Phase 2: [`experiments/README.md`](../../experiments/README.md)・本書 §9 にコマンド記載（`hf_local`・`run_local_slm_phase2_smoke.sh`）
- [x] Phase 3: オペレータ草案 [`v7_local_slm_phase3_operator_bundle_v1.json`](../../experiments/baselines/v7_local_slm_phase3_operator_bundle_v1.json)（本番凍結時は `*_v2` へ複製・パス確定）
- [x] 事前登録スタブの改訂欄: [`v7_local_slm_llm_judge_prereg_stub_v1.json`](v7_local_slm_llm_judge_prereg_stub_v1.json) の **`prereg_revision`**（運用 **§12**）
- [ ] SLM 同士一致の補遺ログ（任意・§10・主結果に含めない）

---

## 8. 関連ドキュメント

- [Phase I-A / III-A 本番手順設計](Phase_IA_IIIA_本番手順設計_v7.md)
- [v7_phase2a_theory_bridge.md](v7_phase2a_theory_bridge.md)（コーパス指標と理論 τ の区別）
- [v7_corpus_MRMP.md](v7_corpus_MRMP.md)

## 9. 実装済みコマンド（リポジトリ）

| フェーズ | コマンド |
|----------|----------|
| Phase 0 | `python experiments/v7_local_env_check.py`（torch / MPS / transformers を JSON 出力） |
| Phase 1 | `bash experiments/run_local_slm_phase1_smoke.sh`（`rinna/japanese-gpt2-medium`・サンプル JSONL・1 対話）。`CPU=1` で `--cpu` |
| Phase 1 検証 | `python experiments/v7_phase2a_bundle_validate.py --bundle experiments/baselines/v7_local_slm_phase1_smoke_bundle_v1.json`（非 strict・成果物は `with_contrib` のみ） |
| Phase 2 | `bash experiments/run_local_slm_phase2_smoke.sh` または `python experiments/v7_phase1a_llm_judge_six_axes.py --provider hf_local …`（`--cpu` / `--hf-revision` / `--hf-max-new-tokens` 可） |
| Phase 1+2 連続 | `bash experiments/run_local_slm_smoke_all.sh` |
| Phase 1 英日比較表 | `bash experiments/run_local_slm_phase1_compare_en_ja.sh`（`BASE`・`JSONL`・`CPU=1` 可） |
| Phase 3 チャンク（rinna 既定） | `bash experiments/run_local_slm_phase3_mrmp_chunk.sh`（`OFFSET` / `MAX_ROWS` / `OUT_PREFIX`） |
| MRMP 審判 hf_local チャンク | `bash experiments/run_mrmp_llm_judge_chunks_hf_local.sh`（`HF_MODEL`・`CPU=1` 可） |
| run_phase2a に OFFSET/MODEL | `OFFSET=0 MODEL=gpt2 bash experiments/run_phase2a_mrmp_tau.sh`（既定は従来どおり gpt2） |
| 注意 run の revision 固定 | `python experiments/v7_phase2a_empirical_run.py --model rinna/japanese-gpt2-medium --revision <SHA> …` |
| 審判プロンプト正本 | `experiments/prompts/v7_llm_judge_prompt_v1.json` |
| SLM 同士の審判一致（探索） | `python experiments/v7_llm_judge_slm_pair_agreement.py`（`--out-json` / `--out-md`）または `bash experiments/run_local_slm_judge_pair_agreement.sh`（任意 `OUT_MD`） |
| Phase 4 審判 A/B + 一致（探索・重い） | `bash experiments/run_local_slm_phase4_judge_pair.sh`（`HF_MODEL_A` / `HF_MODEL_B`・`MAX_ROWS` 等） |
| Phase 4 Swallow 7B vs 13B（探索・重い） | `bash experiments/run_local_slm_phase4_swallow_7b_13b.sh` |
| チャンク後の再現マニフェスト | **§11**（`WRITE_REPRO_MANIFEST=1`・`BUNDLE_JSON`） |

*改訂時は採用モデルの revision と本書の版日を更新すること。*

---

## 10. SLM 同士の審判一致ログ（実装計画・手順）

### 目的

同一 MRMP 窓（同一 `id`）に対し、**審判モデルだけを変えた** 2 本の JSONL を比較し、12 軸の **Pearson r** と **平均絶対差**を記録する。主張は「モデル間ブレの大きさの感度」に限り、**人手正解や OpenAI との一致は扱わない**。

### 前提

- 両方とも同一 [`v7_llm_judge_prompt_v1.json`](../../experiments/prompts/v7_llm_judge_prompt_v1.json)・同一 `temperature` / seed 規約で [`v7_phase1a_llm_judge_six_axes.py`](../../experiments/v7_phase1a_llm_judge_six_axes.py) `--provider hf_local` を実行した出力。
- 入力 JSONL の **行 `id` が一致**すること（オフセット・行数が違うと突合せ漏れが出る）。欠損・非数値行は集計から除外され、`n_rows_skipped_*` に反映される。

### 手順（推奨）

1. 対象スライスを決める（例: `SRC` の `--offset` / `--max-rows` を A・B で同一にする）。
2. `HF_MODEL=model_a` … で `--out-jsonl judge_a.jsonl` を生成。
3. `HF_MODEL=model_b` … で同じ `SRC`・同じオフセットで `judge_b.jsonl` を生成。
4. `python experiments/v7_llm_judge_slm_pair_agreement.py --jsonl-a judge_a.jsonl --jsonl-b judge_b.jsonl --out-json agreement.json --out-md agreement.md`（MD は任意）
5. 補遺に `llm_judge_meta_sample_*`・`prompt_fingerprint_sha256_expected`・両モデル revision を転記。

### 実装

- 集計 CLI: [`experiments/v7_llm_judge_slm_pair_agreement.py`](../../experiments/v7_llm_judge_slm_pair_agreement.py)（`schema_version: v7_llm_judge_slm_pair_agreement.v1`）。
- 薄いラッパ: [`experiments/run_local_slm_judge_pair_agreement.sh`](../../experiments/run_local_slm_judge_pair_agreement.sh)（`JSONL_A` / `JSONL_B` / `OUT_JSON`、任意 `OUT_MD`）。
- Phase 4 一括（重い）: [`experiments/run_local_slm_phase4_judge_pair.sh`](../../experiments/run_local_slm_phase4_judge_pair.sh)、Swallow 7B/13B 専用: [`experiments/run_local_slm_phase4_swallow_7b_13b.sh`](../../experiments/run_local_slm_phase4_swallow_7b_13b.sh)。
- 審判 JSONL の生成自体の一括シェルは必須としない（モデル 2 回ロードは環境依存が大きい）。必要なら `HF_MODEL` を切り替えて `run_mrmp_llm_judge_chunks_hf_local.sh` を 2 回、同一 `SRC`・同一オフセットで別 `OUT` に出力する。

---

## 11. Phase 3 — 再現マニフェスト運用（確定手順）

`run_phase2a_mrmp_tau.sh` または `run_local_slm_phase3_mrmp_chunk.sh` で **`OUT_PREFIX`** を決めたチャンク（または本番一括）のあと、次を推奨する。

1. **bundle の選び方**: 成果物のキーが揃うポインタを使う。Phase II-A 本番相当なら [`v7_phase2a_mrmp_tau_n3146_bundle_v1.json`](../../experiments/baselines/v7_phase2a_mrmp_tau_n3146_bundle_v1.json) を `BUNDLE_JSON` にし、`--out-prefix` で **`OUT_PREFIX` と一致**させる（[`experiments/README.md`](../../experiments/README.md) の再現チェックリスト参照）。
2. **生成**:  
   `WRITE_REPRO_MANIFEST=1 BUNDLE_JSON=experiments/baselines/v7_phase2a_mrmp_tau_n3146_bundle_v1.json OUT_PREFIX=<あなたのプレフィックス> bash experiments/run_phase2a_mrmp_tau.sh`  
   のように、**既に JSON/MD が存在する状態**でマニフェスト段に入るか、または `python experiments/v7_phase2a_repro_manifest.py --bundle … --out-prefix … --out …` を単独実行する。
3. **図まで必須**: 論文用 PNG/PDF を成果物に含める凍結では `REPRO_MANIFEST_STRICT=1`（`run_phase2a_mrmp_tau.sh`）または `v7_phase2a_repro_manifest.py --strict`。
4. **検証**: `python experiments/v7_phase2a_repro_manifest.py --verify ${OUT_PREFIX}_repro_manifest.json`、または `v7_phase2a_bundle_validate.py --strict --out-prefix "$OUT_PREFIX" --verify-manifest …`。
5. **コードのみの継続記録（CI 相当）**: `v7_phase2a_repro_manifest.py --pin-code-only`（ログ不要）。

---

## 12. 事前登録スタブの revision 運用

対象: [`v7_local_slm_llm_judge_prereg_stub_v1.json`](v7_local_slm_llm_judge_prereg_stub_v1.json) の **`prereg_revision`**（整数）。

次のときに **1 ずつ増やす**（補遺に差分の要約を書く）:

- [`v7_llm_judge_prompt_v1.json`](../../experiments/prompts/v7_llm_judge_prompt_v1.json) の本文変更（`prompt_template_id` を変える場合は別系として扱う）。
- スタブに書いた **既定審判モデル**や手続き（シェルパス・Phase 構成）の変更。
- Phase II-A 本体の事前登録（[`v7_phase2a_prereg_v1.json`](v7_phase2a_prereg_v1.json)）を改訂したうえで、ローカル SLM 手順との整合説明を変えたとき。

増やさずに済む例: 単一実行の HF `revision` 固定（`--hf-revision` / `REVISION`）のみで、プロンプトとスタブの役割定義が変わらない場合。

---

## 13. Phase 4 拡張 — MLX 評価（計画のみ）

**現状**: 本リポジトリは **PyTorch + transformers + MPS/CUDA** 経路を正とする。**MLX / mlx-lm** は未同梱。採用する場合は **別マニフェスト**（モデル変換・バージョン・プロンプト適用差）で管理する。

**評価のすすめ方（探索）**:

1. 同一プロンプト・同一スライスで、**tokens/s または 1 窓あたり秒**を PyTorch 推論と比較（審判 1 行またはごく小さな `max_new_tokens` から）。
2. **合格の目安**: スループットが **2 倍以上**かつ、**12 軸の数値**が §10 の手続きで PyTorch 側と実務上無視できない差にならないこと（厳密一致は期待しない）。
3. 主結果パイプラインに載せる前に、**別節「高速経路」**として文書化し、本プランの Phase 0–3 の正本とは切り分ける。
