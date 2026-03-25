# 貢献ガイド（Resonanceverse）

英語サマリー: [Contributing (English summary)](#contributing-english-summary) at the bottom.

## 方針の全体像（中間案）

本リポジトリでは次を組み合わせます。

| 要素 | 役割 |
|------|------|
| **[ETHICAL.md](ETHICAL.md)** / **[ETHICAL_ja.md](ETHICAL_ja.md)** | 倫理・利用に関する宣言・要請 |
| **[CLA.md](CLA.md)** / **[CLA_ja.md](CLA_ja.md)** | **軽量 CLA** — 貢献物を AGPL-3.0 の下でプロジェクトが配布できる**ライセンス付与**（著作権譲渡ではない）。**初回貢献時に一度**同意（下記）。 |
| **DCO**（Developer Certificate of Origin） | **すべてのコミット**に `Signed-off-by` を付与し、出所とライセンス適合を表明する |

---

## 1. 貢献前に読むもの

1. [ETHICAL_ja.md](ETHICAL_ja.md)（または [ETHICAL.md](ETHICAL.md)）— 本書「読み方」節を含む  
2. [CLA_ja.md](CLA_ja.md)（または [CLA.md](CLA.md)）— 初回 PR 前に内容を確認する  

---

## 2. 軽量 CLA（初回のみ）

- **初めて Pull Request を送る前**に、[CLA_ja.md](CLA_ja.md) を読み、内容に同意できること。  
- リポジトリ管理者が [CLA assistant](https://github.com/cla-assistant/cla-assistant) 等を有効にしている場合、**GitHub 上で一度**同意画面に従う（同一アカウントでは再同意不要）。  
- ボット未導入の間は、PR 本文に次の一文を含めてください（テンプレートに含める）:  
  `I have read and agree to CLA.md / CLA_ja.md version 1.0.`

---

## 3. DCO（すべてのコミット）

[Developer Certificate of Origin](https://developercertificate.org/) に基づき、**各コミット**に次の行を含めます（`git commit -s` で自動付与）。

```
Signed-off-by: Random J Developer <random@example.com>
```

- 実名または活動名と、有効なメールアドレスを用いてください。  
- 複数コミットの PR では、**すべてのコミット**に `Signed-off-by` がある必要があります。欠けている場合は `git rebase` 後に `git commit -s --amend` 等で修正してください。

### 採用ポリシー（DCO・ブランチ）

本リポジトリでは次を**正式な運用**とする。

| 項目 | 内容 |
|------|------|
| **必須チェック** | PR のジョブ **`dco`**（および通常は **`test`**（`.github/workflows/ci.yml`））を、ブランチ保護の「必須ステータスチェック」に含める。 |
| **`main` への変更** | **Pull Request 経由のマージ**を原則とする。`main` への直接 push は、メンテナも**行わない**（緊急時のみ例外とする場合は別途合意）。 |
| **`dco-push`** | `main` / `master` への push 後にも実行されるが、**必須チェックには含めない**（任意の安全網。push は止められないため）。直接 push を許す運用のときの検知用。 |

### CI とマージ拒否

- PR ではジョブ **`dco`** が、対象コミットに `Signed-off-by` 行があるか検査する。**欠けていると失敗**し、必須チェックにしていれば**マージ不可**になる。
- **`dco-push`** は push **後**に走る。悪いコミットをリモートに載せないためには、上表どおり **PR 必須**が効く。
- ローカルで PR 範囲と同じ検査をする例:

```bash
git fetch origin main
bash scripts/check_dco.sh "$(git merge-base origin/main HEAD)" "$(git rev-parse HEAD)"
# push イベント相当（リモートの before/after を知っている場合）
# bash scripts/check_dco_push.sh "$BEFORE_SHA" "$AFTER_SHA"
```

---

## 4. 開発手順（簡易）

```bash
pip install -r requirements.txt
pip install pytest
pytest tests/ -v
```

- 変更は**トピックブランチ**から PR を送ることを推奨します。  
- `core/` の変更は [docs/api/modules/](docs/api/modules/README.md) の再現手順・CI と整合するよう、可能な範囲で `pytest` を通してください。

---

## 5. Pull Request

- PR テンプレートのチェックリストに従ってください（[.github/pull_request_template.md](.github/pull_request_template.md)）。  
- 説明に「何を」「なぜ」変更したかを簡潔に書いてください。

---

## 6. メンテナ向け（ブランチ保護の設定例）

GitHub: **Settings → Branches → Branch protection rules** で `main`（および `master` を使う場合は同様）に対し、例として次を設定する。

1. **Require a pull request before merging** を有効にする（レビュー人数はチームで決定）。  
2. **Require status checks to pass before merging** を有効にし、必須ジョブに少なくとも **`dco`** と **`test`**（CI ワークフロー名に合わせる）を追加する。  
3. **Do not allow bypassing the above settings** を、可能ならメンテナも含め有効にする（組織ポリシーに従う）。  
4. **`dco-push` は必須チェックに含めない**（任意実行のまま）。  

直接 push を技術的に禁止するには、**Restrict who can push to matching branches** で許可ユーザーを空に近づける、または Organization の **rulesets** で同等の制限をかける。

---

## 7. メンテナ向け（CLA Assistant の例）

1. GitHub App「CLA assistant」を組織またはリポジトリにインストールする。  
2. 同意文書の URL として本リポジトリの **`CLA.md`**（英語）を登録する（GitHub raw URL 等）。  
3. 初回 PR 時のみ同意を求める設定にする。  

詳細は CLA assistant の公式ドキュメントに従ってください。

---

## Contributing (English summary)

- Read **[ETHICAL.md](ETHICAL.md)** and the lightweight **[CLA.md](CLA.md)** before your first contribution.  
- Agree to the CLA once per GitHub account (e.g. via **CLA assistant** when enabled).  
- Use **`git commit -s`** so every commit has a **`Signed-off-by`** line (DCO).  
- Run **`pytest tests/`** before submitting a PR when possible.
- **Adopted policy:** require status checks **`dco`** and **`test`** on `main`; land changes via **Pull Request** (no direct pushes to `main` in normal operation). The **`dco-push`** job is **not** a required check (optional safety net after push).
