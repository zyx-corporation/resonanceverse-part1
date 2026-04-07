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
**RVT-EXP-2026-008 と本リポの対応（L1–L3・未実装の切り分け）**: [rvt_exp_2026_008_architecture_bridge.md](rvt_exp_2026_008_architecture_bridge.md) — Day 0 コーパス検証は `python experiments/rvt_exp_2026_008_day0_checks.py`（§15 手順 2 の前後で実行可）。**ヘッド別注意→6軸代理のバッチ探索**: [`run_rvt_mrmp_batch.sh`](../../experiments/run_rvt_mrmp_batch.sh)（`MAX_ROWS` は小さく、`CPU=1` 推奨）または Day0 込みのワンショット [`run_rvt_explore.sh`](../../experiments/run_rvt_explore.sh)。

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
- [x] **HF_TOKEN / HF_HOME / ディスク**の運用方針を下記 **Phase 0 補足**に記載（**コード変更は不要**・実設定と空き容量は**各自のマシン**で実施）。
- [x] **任意・実機の一覧確認（値は出さない）**: `python experiments/v7_local_optional_checklist.py`（空き GB・MPS・トークン有無・MLX import のみ）。
- [ ] （任意・実機） gated モデルを使うマシンでは `HF_TOKEN` を設定済み、`HF_HOME` を希望パスにし、キャッシュ＋本番出力用に **目安どおりの空き GB** を確保している（上記 JSON と §0 補足で手動確認）。

**完了基準**: `python -c "import torch; print(torch.backends.mps.is_available())"` が True（または計画した CPU フォールバック方針を文書化）。

#### Phase 0 補足 — HF_TOKEN・HF_HOME・ディスク（マシン側運用メモ）

本リポジトリのスクリプトは **`HF_TOKEN` を要求しない**が、Hub 上で **利用規約同意・ゲート**がかかったモデルや、レート制限緩和が必要な取得ではトークンが要る。いずれも **リポジトリや共有ログに書かない**。

**HF_TOKEN**

- **用途**: ゲート付き重みの取得、（必要なら）Hub API の認証付きダウンロード。
- **設定例**（zsh）: シェル起動時のみに読み込む `~/.zshrc` や、**リポ外**の `~/.hf_env` を `source` するなど。**`.env` をコミットしない**（プロジェクトに `.env` を置く場合は `.gitignore` を確認）。
- **代替**: `huggingface-cli login` で `~/.cache/huggingface/token` に保存（マシンローカル）。CI ではシークレットに `HF_TOKEN` を渡す。
- **確認**: `python -c "from huggingface_hub import HfFolder; print('ok' if HfFolder.get_token() else 'no token')"` など（トークン値は表示しない）。

**HF_HOME（キャッシュの置き場）**

- **既定**: 未設定時は Hub クライアントが **`~/.cache/huggingface`** 配下にモデル・データセットをキャッシュする（環境により `XDG_CACHE_HOME` の影響あり）。
- **推奨**: 内蔵 SSD が狭い・外付け SSD に寄せたい場合は、実行前に  
  `export HF_HOME="/path/to/fast_disk/huggingface"`  
  を同一ターミナルセッションで固定する（`run_*` シェルを叩く前に `source` しておくと取りこぼしが減る）。
- **注意**: パスを変えると **別キャッシュ**になるため、既に `~/.cache` に落ちている重みは再ダウンロードになる。長期運用なら最初から `HF_HOME` を決めておくか、移行時に `hub` ディレクトリをコピー＋十分な検証。
- **関連**: 一部ツールは `TRANSFORMERS_CACHE` / `HF_HUB_CACHE` を参照する。混乱する場合は **`HF_HOME` を正**とし、[公式のキャッシュ環境変数](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)に沿って揃える。

**ディスク容量の目安（ローカル SLM 本番想定）**

- **7B 級（fp16 前後）**: おおむね **13–16 GB/モデル**（実ファイル・シャード構成で変動）。
- **本プランの典型**: Swallow-7B（審判）＋ Qwen2.5-7B（任意比較）＋ rinna 等小モデル → **キャッシュだけで 40 GB 超**を見込み、**出力 JSONL・マニフェスト・図**用に **さらに 20 GB 級の余裕**があると安全。
- **13B 審判（任意）**: 追加で **おおむね 25–30 GB 級**（単体の目安；実測は `du -sh "$HF_HOME"` で確認）。

**一言**: 非公開・ゲート付きを使うなら **トークンはシェル／シークレットのみ**、キャッシュは **`HF_HOME` で速いディスクに固定**、本番前に **`du` と空き容量**で足りることを確認する。

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

#### Phase 3 本番 — オペレータ手順（実走の順序）

ここは **マシン上の実データ実行**（リポにコードを足す作業ではない）。順序だけ固定しておくと取りこぼしが減る。

1. **MRMP 生コーパス**  
   `experiments/logs/mrmp_repo/multi_relational_multi_party_chat_corpus/` が無ければ [`fetch_mrmp_corpus.py`](../../experiments/fetch_mrmp_corpus.py)（手順の正本: [`v7_corpus_MRMP.md`](v7_corpus_MRMP.md)）。

2. **`windows.jsonl` 生成**  
   ```bash
   python experiments/v7_mrmp_prepare.py
   ```  
   既定で `experiments/logs/mrmp_prepared/windows.jsonl`・`manifest.json`・`dialogue_eval.jsonl`。デバッグは `--max-dialogues N`・窓は `--window W`。

3. **注意パイプライン（チャンク）** — いずれか一方  
   - **rinna 既定の薄ラッパ**: [`run_local_slm_phase3_mrmp_chunk.sh`](../../experiments/run_local_slm_phase3_mrmp_chunk.sh)（内部で [`run_phase2a_mrmp_tau.sh`](../../experiments/run_phase2a_mrmp_tau.sh) を実行）。**`WINDOWS` はシェルに渡さない**ため、審判済み JSONL 等で置き換えるときは  
     `WINDOWS=path/to/judged.jsonl bash experiments/run_local_slm_phase3_mrmp_chunk.sh` のように **先頭で環境変数を付ける**。  
   - **フル制御**: `WINDOWS`・`OFFSET`・`MAX_ROWS`・`MODEL`・`REVISION`・`OUT_PREFIX`・`CPU=1` を [`run_phase2a_mrmp_tau.sh`](../../experiments/run_phase2a_mrmp_tau.sh) に直接渡す。  
   **`WINDOWS`** は手順 2 の `windows.jsonl`（未設定時の既定は `experiments/logs/mrmp_prepared/windows.jsonl`）。**MRMP 窓本線**では `DAY0_CHECK=1` でパイプライン先頭に [`rvt_exp_2026_008_day0_checks.py`](../../experiments/rvt_exp_2026_008_day0_checks.py)（`--strict-manifest`）を挟める（**審判済み JSONL だけ**に `WINDOWS` を振り替えた実行では **付けない**）。本番の **revision 固定**は `REVISION=<HF commit SHA>`（補遺に転記）。

4. **（任意）図**  
   論文用なら同一実行フローで `GENERATE_PAPER_PLOTS=1`（[`experiments/README.md`](../../experiments/README.md) の Phase II-A 節参照）。

5. **厳密検証**  
   `VALIDATE_STRICT=1` を付けて bundle 検証、または終了後に  
   `python experiments/v7_phase2a_bundle_validate.py --strict --out-prefix "$OUT_PREFIX"`。

6. **凍結・マニフェスト（§11）**  
   成果物の種類に合わせて **`BUNDLE_JSON`** を選ぶ（注意のみなら [`v7_phase2a_mrmp_tau_n3146_bundle_v1.json`](../../experiments/baselines/v7_phase2a_mrmp_tau_n3146_bundle_v1.json)、6 軸審判済み入力なら [`v7_phase2a_mrmp_tau_n3146_judge10k_bundle_v1.json`](../../experiments/baselines/v7_phase2a_mrmp_tau_n3146_judge10k_bundle_v1.json) 等）。  
   ```bash
   WRITE_REPRO_MANIFEST=1 REPRO_MANIFEST_STRICT=1 \
   BUNDLE_JSON=experiments/baselines/v7_phase2a_mrmp_tau_n3146_bundle_v1.json \
   OUT_PREFIX=<あなたのプレフィックス> bash experiments/run_phase2a_mrmp_tau.sh
   ```  
   既に JSON が揃っているなら `python experiments/v7_phase2a_repro_manifest.py --bundle … --out-prefix … --out "${OUT_PREFIX}_repro_manifest.json"` のみでも可。

7. **verify**  
   `python experiments/v7_phase2a_repro_manifest.py --verify "${OUT_PREFIX}_repro_manifest.json"`（または `v7_phase2a_bundle_validate.py --verify-manifest …`）。

#### Phase 3 補足 — 審判本番（hf_local チャンク）

**いつ使うか**: 6 軸（`trust_ab` … `history_ba`）を MRMP 窓に付けたうえで、注意パイプラインの **補助系列**や judge 同梱 bundle を使うとき（手順 3 の **`WINDOWS`** に審判済み JSONL を渡す）。

**前提**: 手順 2 まで済み、`SRC` となる `windows.jsonl` が存在すること（[`run_mrmp_llm_judge_chunks_hf_local.sh`](../../experiments/run_mrmp_llm_judge_chunks_hf_local.sh) は **`SRC` 未存在なら exit 1** とヒント表示）。

**環境変数（主なもの）**

| 変数 | 既定 | 意味 |
|------|------|------|
| `SRC` | `experiments/logs/mrmp_prepared/windows.jsonl` | 審判入力（`v7_mrmp_prepare` 出力） |
| `OUT` | `experiments/logs/v7_judge_hf_local_chunks.jsonl` | 審判付き JSONL（追記・`--resume` でチャンク連結） |
| `CHUNK` | `1000` | 1 チャンクあたり最大行数 |
| `N_CHUNKS` | `10` | チャンク本数（合計上限 ≈ `CHUNK × N_CHUNKS`） |
| `LOG` | `${OUT%.jsonl}_hf_local_run.log` | `tee` 先の実行ログ |
| `HF_MODEL` | `tokyotech-llm/Swallow-7b-instruct-hf` | 審判モデル |
| `HF_REVISION` | 省略 | 本番固定時は **SHA**（補遺・マニフェストに転記） |
| `HF_MAX_NEW_TOKENS` | `256` | 生成上限 |
| `TEMPERATURE` | `0` | 決定論寄り |
| `CPU` | `0` | `1` で `--cpu`（MPS 問題時の切り分け用） |

**実行**

```bash
bash experiments/run_mrmp_llm_judge_chunks_hf_local.sh
```

スモーク例: `CHUNK=100 N_CHUNKS=2 CPU=1 bash experiments/run_mrmp_llm_judge_chunks_hf_local.sh`  
本番は `OUT`・`CHUNK`・`N_CHUNKS` を補遺に書いた値に合わせる（途中失敗時は **同一 `OUT`** で再実行し、`v7_phase1a_llm_judge_six_axes.py` の **`--resume`** が続きから追記する）。

**その後の Phase II-A**

```bash
WINDOWS="$OUT" OFFSET=0 MAX_ROWS=3146 MODEL=rinna/japanese-gpt2-medium \
OUT_PREFIX=experiments/logs/v7_… bash experiments/run_phase2a_mrmp_tau.sh
```

（`MAX_ROWS`・`MODEL`・`OUT_PREFIX`・`REVISION` は本番設計に合わせる。審判のみ変えた再現なら **`HF_REVISION` と `OUT` のペア**を補遺に残す。）

**bundle / マニフェスト**: 補助解析を成果物に含める凍結では [`v7_phase2a_mrmp_tau_n3146_judge10k_bundle_v1.json`](../../experiments/baselines/v7_phase2a_mrmp_tau_n3146_judge10k_bundle_v1.json) のような **審判同梱版**を `BUNDLE_JSON` に選ぶ（手順 6）。プロンプトや手続きを変えたら **§12** のとおり `prereg_revision` を検討。

**主結果用の一括（審判 → 注意 → 図・strict・マニフェスト verify）**: [`run_mrmp_judge_hf_local_then_phase2a_judge10k.sh`](../../experiments/run_mrmp_judge_hf_local_then_phase2a_judge10k.sh) — 既定で `OUT_JUDGE=experiments/logs/v7_judge_10k.jsonl`・`OUT_PREFIX=experiments/logs/v7_phase2a_mrmp_tau_n3146_judge10k`・`REPRO_MANIFEST_STRICT=1`。審判のみ済みなら **`SKIP_MRMP_JUDGE=1`**。`run_phase2a_mrmp_tau.sh` は **`BUNDLE_JSON` 付きで `bundle_validate --strict`** を呼ぶ（図パスは `v7_phase2a_bundle_validate.py` が `figures` / `figures_paper` も検証）。

**最短（リポジトリ直下・カレント不要）**: `bash run_mrmp_judge10k_pipeline.sh`（内部で `experiments/run_mrmp_judge_hf_local_then_phase2a_judge10k.sh` を実行）。

**コピペ実行（角括弧 `<` `>` はシェルでリダイレクトになるので使わない）**: ルートで  
`bash experiments/run_mrmp_judge_hf_local_then_phase2a_judge10k.sh`  
でも可。revision 固定は **実 SHA 文字列**をそのまま:  
`HF_REVISION=実際のSwallowのSHA REVISION=実際のrinnaのSHA bash run_mrmp_judge10k_pipeline.sh`

### Phase 4 — 拡張・任意

- [x] **Swallow 7B vs 13B** 審判 A/B（実走は任意）: [`run_local_slm_phase4_swallow_7b_13b.sh`](../../experiments/run_local_slm_phase4_swallow_7b_13b.sh)。内部で **A→B の順に 2 回** `v7_phase1a_llm_judge_six_axes.py` を呼び、最後に **§10** と同型の一致 JSON/MD を出力。品質と所要時間はログに残し**主結果にしない**。
- [x] **Qwen2.5-7B-Instruct** 等との審判ブレ（実走は任意）: [`run_local_slm_phase4_judge_pair.sh`](../../experiments/run_local_slm_phase4_judge_pair.sh)（既定 A=Swallow-7B-Instruct・B=Qwen2.5-7B-Instruct）。**手順の全文は下記「Phase 4 補足」**。探索のみ・主結果にしない。
- [ ] **MLX** への移行評価: **§13** の手順・合格基準。リポジトリに MLX 推論コードは未同梱（別環境・別マニフェスト）。Phase 4 の PyTorch 審判パイプラインとは**別マニフェスト**で切り離す。

**完了基準（Phase 4）**: 探索として「同一 `SRC`・同一 `OFFSET`/`MAX_ROWS`・同一 `SEED`/`TEMPERATURE` で 2 モデル審判 → `v7_llm_judge_slm_pair_agreement` 出力」が、**環境変数と 1 コマンド**で説明できること（本節および §9）。

#### Phase 4 補足 — オペレータ手順（審判ペア・探索）

**本体**: [`run_local_slm_phase4_judge_pair.sh`](../../experiments/run_local_slm_phase4_judge_pair.sh) が **モデル A → モデル B** の順で審判し、[`v7_llm_judge_slm_pair_agreement.py`](../../experiments/v7_llm_judge_slm_pair_agreement.py) を実行する。チャンク用の `run_mrmp_llm_judge_chunks_hf_local.sh` とは別経路（こちらは **`v7_phase1a_llm_judge_six_axes.py` を直接**、`--offset` / `--max-rows` でスライス）。

**環境変数（主なもの）**

| 変数 | 既定 | 意味 |
|------|------|------|
| `SRC` | `experiments/data/v7_mrmp_sample.jsonl` | 入力 JSONL（本番寄りなら `experiments/logs/mrmp_prepared/windows.jsonl` 等に上書き） |
| `OFFSET` | `0` | スキップ行数 |
| `MAX_ROWS` | `3`（スモーク） | 最大処理行数。**本番探索でもまず小さく**上げる |
| `SEED` | `0` | 審判スクリプトの seed |
| `TEMPERATURE` | `0` | 審判の温度 |
| `HF_MODEL_A` | `tokyotech-llm/Swallow-7b-instruct-hf` | 審判 A |
| `HF_MODEL_B` | `Qwen/Qwen2.5-7B-Instruct` | 審判 B（任意の HF ID に差し替え可） |
| `HF_REVISION_A` / `HF_REVISION_B` | 省略 | 本番固定時は各モデルの **SHA**（補遺 **§14**） |
| `OUT_A` / `OUT_B` | `experiments/logs/phase4_judge_hf_{a,b}.jsonl` | 審判付き JSONL |
| `OUT_JSON` / `OUT_MD` | `phase4_slm_pair_agreement.{json,md}` | 一致集計の出力 |
| `CPU` | `0` | `1` で `--cpu` |

**スモーク（既定どおり・サンプル JSONL）**

```bash
bash experiments/run_local_slm_phase4_judge_pair.sh
```

**MRMP 窓の先頭 N 行だけ（探索例）**

```bash
SRC=experiments/logs/mrmp_prepared/windows.jsonl OFFSET=0 MAX_ROWS=20 CPU=1 \
  bash experiments/run_local_slm_phase4_judge_pair.sh
```

（`MAX_ROWS`・`CPU`・`OUT_*` は環境に合わせる。13B を含む場合は **メモリと時間**に余裕を見る。）

**Swallow 7B × 13B（薄ラッパ）**

[`run_local_slm_phase4_swallow_7b_13b.sh`](../../experiments/run_local_slm_phase4_swallow_7b_13b.sh) は `HF_MODEL_B` を **`tokyotech-llm/Swallow-13b-instruct-hf`** にし、`OUT_*` を 7b/13b 用に分けたうえで上記 `judge_pair` を `exec` する。

```bash
MAX_ROWS=5 CPU=1 bash experiments/run_local_slm_phase4_swallow_7b_13b.sh
```

**§10 との使い分け**: **チャンク＋再開**で大量に審判したいときは `run_mrmp_llm_judge_chunks_hf_local.sh` と §10 のワンショット。**1 スライスを 2 モデルで即比較**するときは本 Phase 4 シェルが短い。

**MLX（Phase 4 の別枠）**: 高速化の評価だけ **§13** を参照。審判の数値一致は §10 手続きで PyTorch 側と突き合わせ、主パイプラインに載せる前に文書化する。

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
- [ ] SLM 同士一致の補遺ログ（任意・**§10「補遺用ワンショット例」**・主結果に含めない）

---

## 8. 関連ドキュメント

- [Phase I-A / III-A 本番手順設計](Phase_IA_IIIA_本番手順設計_v7.md)
- [v7_phase2a_theory_bridge.md](v7_phase2a_theory_bridge.md)（コーパス指標と理論 τ の区別）
- [v7_corpus_MRMP.md](v7_corpus_MRMP.md)
- [Resonanceverse v7.0 実証実験設計書](../v7/Resonanceverse_v7.0_Experimental_Design.md)（本番値の転記先・**§14**）
- [v7_phase2a_prereg_v1.json](v7_phase2a_prereg_v1.json)（Phase II-A 事前登録・**§14**）

## 9. 実装済みコマンド（リポジトリ）

| フェーズ | コマンド |
|----------|----------|
| Phase 0 | `python experiments/v7_local_env_check.py`（torch / MPS / transformers を JSON 出力）。**HF_TOKEN / HF_HOME / ディスク**は本書 §4「Phase 0 補足」 |
| Phase 1 | `bash experiments/run_local_slm_phase1_smoke.sh`（`rinna/japanese-gpt2-medium`・サンプル JSONL・1 対話）。`CPU=1` で `--cpu` |
| Phase 1 検証 | `python experiments/v7_phase2a_bundle_validate.py --bundle experiments/baselines/v7_local_slm_phase1_smoke_bundle_v1.json`（非 strict・成果物は `with_contrib` のみ） |
| Phase 2 | `bash experiments/run_local_slm_phase2_smoke.sh` または `python experiments/v7_phase1a_llm_judge_six_axes.py --provider hf_local …`（`--cpu` / `--hf-revision` / `--hf-max-new-tokens` 可） |
| Phase 1+2 連続 | `bash experiments/run_local_slm_smoke_all.sh` |
| Phase 1 英日比較表 | `bash experiments/run_local_slm_phase1_compare_en_ja.sh`（`BASE`・`JSONL`・`CPU=1` 可） |
| MRMP → `windows.jsonl` | `python experiments/v7_mrmp_prepare.py`（`--out-dir`・`--window`・`--max-dialogues`）→ **Phase 3 本番**の手順 1–2 |
| Phase 3 チャンク（rinna 既定） | `bash experiments/run_local_slm_phase3_mrmp_chunk.sh`（`WINDOWS`・`OFFSET` / `MAX_ROWS` / `OUT_PREFIX`・親シェルは `run_phase2a_mrmp_tau.sh`） |
| MRMP 審判 hf_local チャンク | `bash experiments/run_mrmp_llm_judge_chunks_hf_local.sh`（`SRC`・`OUT`・`CHUNK`・`N_CHUNKS`・`HF_MODEL`・`HF_REVISION`・`CPU=1`）→ 詳細は **§4「Phase 3 補足 — 審判本番」** |
| 審判→主結果（補助 6 軸）一括 | `bash experiments/run_mrmp_judge_hf_local_then_phase2a_judge10k.sh`（**§4 審判本番**末尾・`SKIP_MRMP_JUDGE` 可） |
| run_phase2a に OFFSET/MODEL | `OFFSET=0 MODEL=gpt2 bash experiments/run_phase2a_mrmp_tau.sh`（既定は従来どおり gpt2） |
| 注意 run の revision 固定 | `python experiments/v7_phase2a_empirical_run.py --model rinna/japanese-gpt2-medium --revision <SHA> …` |
| 審判プロンプト正本 | `experiments/prompts/v7_llm_judge_prompt_v1.json` |
| SLM 同士の審判一致（探索） | `python experiments/v7_llm_judge_slm_pair_agreement.py`（`--out-json` / `--out-md`）または `bash experiments/run_local_slm_judge_pair_agreement.sh`（任意 `OUT_MD`） |
| Phase 4 審判 A/B + 一致（探索・重い） | `bash experiments/run_local_slm_phase4_judge_pair.sh`（`SRC`・`OFFSET`・`MAX_ROWS`・`HF_MODEL_A` / `HF_MODEL_B`・`HF_REVISION_*`・`OUT_*`・`CPU=1`）→ **§4「Phase 4 補足」** |
| Phase 4 Swallow 7B vs 13B（探索・重い） | `bash experiments/run_local_slm_phase4_swallow_7b_13b.sh`（内部は `judge_pair`・既定 B=`Swallow-13b-instruct-hf`）→ **§4「Phase 4 補足」** |
| チャンク後の再現マニフェスト | **§11**（`WRITE_REPRO_MANIFEST=1`・`BUNDLE_JSON`） |
| 論文・事前登録への転記チェック | **§14**（`OUT_PREFIX`・bundle・revision・prereg） |
| **次に実走するときの順序** | **§15**（環境 → prepare → 注意スモーク → 本番チャンク → 図・validate → manifest → §14） |

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

### 補遺用ワンショット例（小さな `CHUNK`）

**方針**: 主結果にしない探索ログとして、**同一 `SRC`・同一先頭 N 行**でモデル A/B の審判 JSONL を 2 本出し、直後に集計する（`CHUNK=50`・`N_CHUNKS=1` なら先頭 50 窓のみ）。

**Swallow 7B × Qwen2.5-7B-Instruct**（例: `SRC` は既定の `windows.jsonl`）:

```bash
SRC=experiments/logs/mrmp_prepared/windows.jsonl
OUT_A=experiments/logs/_supp_judge_swallow7b_n50.jsonl
OUT_B=experiments/logs/_supp_judge_qwen25_7b_n50.jsonl

HF_MODEL=tokyotech-llm/Swallow-7b-instruct-hf CHUNK=50 N_CHUNKS=1 OUT="$OUT_A" \
  bash experiments/run_mrmp_llm_judge_chunks_hf_local.sh

HF_MODEL=Qwen/Qwen2.5-7B-Instruct CHUNK=50 N_CHUNKS=1 OUT="$OUT_B" \
  bash experiments/run_mrmp_llm_judge_chunks_hf_local.sh

python experiments/v7_llm_judge_slm_pair_agreement.py \
  --jsonl-a "$OUT_A" --jsonl-b "$OUT_B" \
  --out-json experiments/logs/_supp_slm_agreement_swallow7b_qwen25_n50.json \
  --out-md experiments/logs/_supp_slm_agreement_swallow7b_qwen25_n50.md
```

**Swallow 7B × Swallow 13B**（13B の HF ID は当時のカードに合わせる）: 上と同様に **`HF_MODEL` だけ差し替え**、`OUT_*` と `--out-json` / `--out-md` のファイル名を `_13b` 等で区別する。

**ラッパのみ**: 既に 2 本あるなら  
`JSONL_A=… JSONL_B=… OUT_JSON=… OUT_MD=… bash experiments/run_local_slm_judge_pair_agreement.sh`

補遺には集計 JSON 内の **`prompt_fingerprint_sha256`**・各 JSONL の **`llm_judge_meta`（model / hf_revision / prompt_template_id）**・`SRC` の行数スライスを短く書く（手順 5 どおり）。

---

## 11. Phase 3 — 再現マニフェスト運用（確定手順）

`run_phase2a_mrmp_tau.sh` または `run_local_slm_phase3_mrmp_chunk.sh` で **`OUT_PREFIX`** を決めたチャンク（または本番一括）のあと、次を推奨する。

**`VALIDATE_STRICT=1` の準備（bundle に `pipeline_log` がある場合）**: [`run_phase2a_mrmp_tau.sh`](../../experiments/run_phase2a_mrmp_tau.sh) は実行開始時に **標準出力・標準エラー全体**を **`PIPELINE_LOG`**（既定: `experiments/logs/run_phase2a_mrmp_tau_gpu.log`）へ **tee 上書き**する（ファイル名の `_gpu` は bundle との整合用の歴史的名称）。**並列チャンク**ではチャンクごとに `PIPELINE_LOG=experiments/logs/run_mrmp_chunk_o500.log` のように **パスを分ける**。**順序**: プロットのあと **`bundle_validate --strict`**（`tee` 継続）→ **`tee` を閉じて端末に復帰** → **終了行を `tee -a` でログに追記** → **`WRITE_REPRO_MANIFEST`**（マニフェスト本体の stdout はログに載せない。`--verify` と `pipeline_log` の SHA256 が整合）。**ログ本文と成果物が同一実行のもの**にそろえるには、その実行で `VALIDATE_STRICT=1` を付けて最後まで通す。

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

---

## 14. 論文・事前登録への転記テンプレ（補遺用）

本番を走らせたあと、**実証実験設計書**や**事前登録 JSON**と数を突き合わせるときのチェックリスト。ここに無い項目は設計書の該当節に合わせて増やす。

**Resonanceverse v7.0 実証実験設計書（補遺・再現節）**

- 実行日（ローカル）とリポジトリの **コミット SHA**（任意・マニフェストに含まれるならそこから転記可）。
- **注意モデル**: HF ID・**`REVISION`**（`with_contrib.json` 内メタ）・`inference_device`（`mps` / `cpu`）。
- **審判モデル**（使った場合）: HF ID・**`HF_REVISION`**・`temperature`・プロンプト **`prompt_template_id`**（`v7_llm_judge_prompt_v1.json`）。
- **入力コーパス**: `windows.jsonl` の由来（`v7_mrmp_prepare`・`--window`・MRMP の取得方法）。
- **スライス**: `OFFSET`・`MAX_ROWS`（または審判チャンクの `CHUNK`×`N_CHUNKS` と再開の有無）。
- **成果物の根**: **`OUT_PREFIX`**（`*_with_contrib.json` 等の共通プレフィックス）。
- **凍結ポインタ**: 採用した **`BUNDLE_JSON`** のファイル名（例: `v7_phase2a_mrmp_tau_n3146_bundle_v1.json` vs `…_judge10k_…`）。
- **マニフェスト**: `${OUT_PREFIX}_repro_manifest.json` のパスと、`--verify` 済みか。

**`docs/planning/v7_phase2a_prereg_v1.json`**

- Phase II-A の定義（span・τ レンジ等）を変えた場合は、**本体 prereg の改訂手続き**に従い、ローカル SLM 手順との差分を補遺に 1 段落書く。

**`v7_local_slm_llm_judge_prereg_stub_v1.json`**

- プロンプト・既定モデル・シェル手順を変えたら **`prereg_revision`** を **§12** に従って増やす。

**§10 / Phase 4 補遺ログ（任意）**

- `v7_llm_judge_slm_pair_agreement` の **JSON/MD** のパスと、比較した **2 モデル名＋ revision**・**スライス行数**（チャンク経路なら §10、**Phase 4 シェル**ならさらに `SRC`・`OFFSET`・`MAX_ROWS`・`OUT_A`・`OUT_B` を併記）。

---

## 15. 次回実行プラン（推奨順・事前に決めること）

文書化済みの Phase 0–4 を、**いまから実走するとき**の順序に並べたテンプレである。**主戦場は注意パイプラインの本番チャンク → マニフェスト凍結**；審判・Phase 4・§10 は任意。

### 実行前に決める（メモしてから走らせる）

| 項目 | 例・メモ |
|------|-----------|
| `HF_HOME` / 空き容量 | Phase 0 補足の目安どおりか |
| `HF_TOKEN` | ゲート付きモデルを使うときのみ |
| `WINDOWS` | 通常は `experiments/logs/mrmp_prepared/windows.jsonl`（未生成なら下記 1） |
| 注意モデル | 例: `rinna/japanese-gpt2-medium` と **`REVISION`（SHA）** |
| チャンク分割 | 各チャンクの `OFFSET`・`MAX_ROWS`・**一意の `OUT_PREFIX`** |
| `CPU` | 既定は MPS 利用。**切り分け・安定優先**なら `CPU=1`（遅い） |
| 審判を載せるか | 要るなら `SRC`/`OUT`/`CHUNK`/`N_CHUNKS`・`HF_REVISION`（§4 Phase 3 補足） |
| 凍結 bundle | 注意のみ `v7_phase2a_mrmp_tau_n3146_bundle_v1.json`、審判補助込みなら `…_judge10k_…` 等 |

### 推奨ステップ（順序固定）

1. **環境**: `python experiments/v7_local_env_check.py`（MPS/バージョンの記録用）。venv と `requirements.txt` は既存どおり。
2. **MRMP → 窓**: `python experiments/v7_mrmp_prepare.py`（必要なら `--window` を設計と揃える）。`windows.jsonl` の行数を把握する。必要なら続けて `python experiments/rvt_exp_2026_008_day0_checks.py --strict-manifest` で必須キーと `manifest.json` 件数を検証。
3. **注意パイプライン・スモーク**: 小さな `MAX_ROWS`（例: 20–100）で  
   `WINDOWS=… OFFSET=0 MAX_ROWS=<小> MODEL=rinna/japanese-gpt2-medium OUT_PREFIX=experiments/logs/_smoke_rinna … bash experiments/run_phase2a_mrmp_tau.sh`  
   または `run_local_slm_phase3_mrmp_chunk.sh` 相当。**MPS で落ちるときだけ** `CPU=1` を付ける。
4. **注意パイプライン・本番チャンク**: 行数に合わせて `OFFSET`/`MAX_ROWS` を積み上げ、**チャンクごとに別 `OUT_PREFIX`**。同一チャンク内では `REVISION` を固定。
5. **（任意）図**: 凍結に PNG/PDF を含めるなら `GENERATE_PAPER_PLOTS=1`（または事後 `v7_phase2a_tau_plots.py --paper`）。
6. **厳密検証**: `VALIDATE_STRICT=1` または `v7_phase2a_bundle_validate.py --strict --out-prefix "$OUT_PREFIX"`。
7. **マニフェスト**: `WRITE_REPRO_MANIFEST=1`・成果物に合う `BUNDLE_JSON`・論文図までハッシュするなら `REPRO_MANIFEST_STRICT=1`（§11）。続けて `--verify`。
8. **§14**: 設計書・prereg スタブへ **OUT_PREFIX・revision・bundle 名**を転記。

**審判本番が次のとき**（手順 3–4 と独立してよいが、`WINDOWS` は同じ正本を使う）: §4「Phase 3 補足 — 審判本番」どおり `run_mrmp_llm_judge_chunks_hf_local.sh` → 完了後 `WINDOWS=<審判OUT>` で注意パイプライン → bundle は **judge 同梱版**を選ぶ。

**任意（時間が残ったら）**: §10 ワンショット（小 `CHUNK`）→ 補遺；Phase 4 補足（`judge_pair` / Swallow 7B×13B）；MLX は §13 のみ別評価。

*この § は「次に何を実行するか」のロードマップであり、数値（3146 等）は設計書・事前登録に合わせて置き換える。*
