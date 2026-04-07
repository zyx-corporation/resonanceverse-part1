# 話者ペア軌道 `v7_pair_w_trajectory`（設計・テスト対応）

`core/v7_pair_w_trajectory.py` と `tests/test_v7_pair_w_trajectory.py` の内容を、人間が読む Markdown に整理したもの。

## 目的

各ターンの隠れ状態 `(S, H)` と話者ごとのトークン index を平均プールし、有向ペアベクトル `w_ij` / `w_ji` の系列 `(T, d)` を得る。続けて `core.v7_awai_metrics.omega_awai_series_from_w` により Ω 時系列 `(T,)` を計算する（MRMP との接続は experiments 側の index 取得に委ね、core は experiments に依存しない）。

## API 摘要

| シンボル | 役割 |
|----------|------|
| `pool_hidden_mean(hidden_sh, token_indices)` | `(S,H)` を index で平均プール。空 index は零ベクトル。 |
| `concat_truncate_pair_vector(h_i, h_j, d)` | `concat` を長さ `d` に切り詰め／パディング（学習なし既定）。 |
| `PairRelationLinear` | `Linear(2H → d)` で有向 `(h_utterer, h_responder)` を射影。 |
| `directed_pair_w_ij_w_ji(..., d, pair_projector=)` | `w_ij` / `w_ji` を返す。projector 未指定時は concat_truncate。 |
| `series_from_turn_hiddens(...)` | ターン列から `w_ij`, `w_ji`, `omega` を一括構築。 |
| `batch_series_from_dialogues(...)` | 複数対話を同手続きで処理。 |

MRMP 窓から index を得るには、呼び出し側で `experiments.v7_phase2a_empirical.speaker_token_indices_mrmp_window` を用いる（モジュール docstring どおり）。

## テスト ↔ 検証内容（`test_v7_pair_w_trajectory.py`）

| テスト関数 | 検証していること |
|------------|------------------|
| `test_pool_hidden_mean_empty_and_nonempty` | 空 index は零ベクトル。非空は `index_select` 後の平均と一致。 |
| `test_directed_pair_w_ij_w_ji` | `concat_truncate` 経由の `w_ij` / `w_ji` の形状と、`a`/`b` の入れ替え。 |
| `test_series_from_turn_hiddens_matches_manual_omega` | `series_from_turn_hiddens` が返す Ω が、`omega_awai_series_from_w` の手計算と一致。 |
| `test_pair_relation_linear_matches_concat_when_identity` | `PairRelationLinear` の重みを単位行列にしたとき、`2H==d` で concat_truncate と一致。 |
| `test_series_from_turn_hiddens_with_pair_projector_grad` | `pair_projector` 使用時に、Ω の和を逆伝播して `proj.weight.grad` が立つ。 |
| `test_directed_pair_d_mismatch_raises` | `d` と `pair_projector.d` の不一致で `ValueError`（メッセージ `pair_projector.d`）。 |
| `test_delay_and_backward_diff_shapes` | `v7_awai_metrics.delay_series_torch` と `backward_diff_torch` の形状・境界（τ=0/1）。 |

## 関連パス

- 実装: [`core/v7_pair_w_trajectory.py`](../../core/v7_pair_w_trajectory.py)
- Ω 定義: [`core/v7_awai_metrics.py`](../../core/v7_awai_metrics.py)
- テスト: [`tests/test_v7_pair_w_trajectory.py`](../../tests/test_v7_pair_w_trajectory.py)
