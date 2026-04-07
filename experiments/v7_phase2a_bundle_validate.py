"""
Phase II-A ベースライン JSON（bundle pointer）の artifacts を検証する。

- 各パスが存在するか（--strict なら欠落で非ゼロ終了）
- bundle に ``figures`` / ``figures_paper`` があれば **PNG/PDF も**存在検査（``--out-prefix`` でパス置換）
- 主要 JSON の schema_version と最小キー

例::

    python experiments/v7_phase2a_bundle_validate.py
    python experiments/v7_phase2a_bundle_validate.py --strict
    OUT_PREFIX=experiments/logs/my_run python experiments/v7_phase2a_bundle_validate.py --strict
    python experiments/v7_phase2a_bundle_validate.py --strict --out-prefix experiments/logs/my_run \\
      --verify-manifest experiments/logs/my_run_repro_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

_SUFFIX_WITH_CONTRIB = "_with_contrib.json"


def default_prefix_from_artifacts(artifacts: dict[str, str]) -> str | None:
    """with_contributions_jsonl から bundle 既定の OUT_PREFIX 相当（拡張子手前）を得る。"""
    wc = artifacts.get("with_contributions_jsonl")
    if not isinstance(wc, str) or not wc.endswith(_SUFFIX_WITH_CONTRIB):
        return None
    return wc[: -len(_SUFFIX_WITH_CONTRIB)]


def remap_artifact_paths(
    artifacts: dict[str, str],
    *,
    bundle_default_prefix: str,
    out_prefix: str,
) -> dict[str, str]:
    """
    bundle 内のパスで ``bundle_default_prefix`` で始まるものを ``out_prefix`` に置換。
    pipeline_log などプレフィックス外はそのまま。
    """
    bd = bundle_default_prefix.strip().rstrip("/")
    op = out_prefix.strip().rstrip("/")
    if not bd or not op or bd == op:
        return dict(artifacts)
    out: dict[str, str] = {}
    for k, rel in artifacts.items():
        if not isinstance(rel, str):
            out[k] = rel
            continue
        if rel == bd or rel.startswith(bd + "_"):
            out[k] = op + rel[len(bd) :]
        else:
            out[k] = rel
    return out


def _remap_path_dict(
    d: dict[str, Any], *, bundle_default_prefix: str, out_prefix: str
) -> dict[str, Any]:
    """figures / figures_paper を artifacts と同じ規則で out_prefix に合わせる。"""
    bd = bundle_default_prefix.strip().rstrip("/")
    op = out_prefix.strip().rstrip("/")
    if not bd or not op or bd == op:
        return dict(d)
    out: dict[str, Any] = {}
    for k, rel in d.items():
        if not isinstance(rel, str):
            out[k] = rel
            continue
        if rel == bd or rel.startswith(bd + "_"):
            out[k] = op + rel[len(bd) :]
        else:
            out[k] = rel
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_with_contrib(data: dict[str, Any], path: Path) -> list[str]:
    errs: list[str] = []
    if data.get("schema_version") != "v7_phase2a_empirical.v1":
        errs.append(f"{path}: schema_version が v7_phase2a_empirical.v1 ではない")
    if not data.get("by_tau"):
        errs.append(f"{path}: by_tau が空")
    if "inference_device" not in data:
        errs.append(f"{path}: inference_device なし（再現ログ推奨）")
    return errs


def _check_tau_summary(data: dict[str, Any], path: Path) -> list[str]:
    errs: list[str] = []
    if data.get("schema_version") != "v7_phase2a_tau_summary.v1":
        errs.append(f"{path}: schema_version が v7_phase2a_tau_summary.v1 ではない")
    if not isinstance(data.get("summary"), dict):
        errs.append(f"{path}: summary オブジェクトなし")
    aux = data.get("auxiliary_summary")
    if aux is not None and not isinstance(aux, dict):
        errs.append(f"{path}: auxiliary_summary が dict ではない")
    return errs


def _check_bootstrap(data: dict[str, Any], path: Path) -> list[str]:
    errs: list[str] = []
    if data.get("schema_version") != "v7_phase2a_bootstrap.v1":
        errs.append(f"{path}: schema_version が v7_phase2a_bootstrap.v1 ではない")
    if not data.get("by_tau"):
        errs.append(f"{path}: by_tau が空")
    return errs


def validate_artifacts(
    repo: Path,
    artifacts: dict[str, str],
    *,
    strict: bool,
) -> tuple[list[str], list[str]]:
    """
    Returns (errors, warnings).
    strict 時は欠落ファイルも errors。非 strict 時は欠落は warnings のみ。
    """
    errors: list[str] = []
    warnings: list[str] = []

    for key, rel in sorted(artifacts.items()):
        path = (repo / rel).resolve()
        if not path.is_file():
            msg = f"欠落: {key} -> {rel}"
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)
            continue

        if path.suffix.lower() != ".json":
            continue

        try:
            data = _load_json(path)
        except (OSError, json.JSONDecodeError) as e:
            errors.append(f"{path}: 読み込み失敗 {e}")
            continue

        if key == "with_contributions_jsonl" or "with_contrib" in key:
            errors.extend(_check_with_contrib(data, path))
        elif key == "tau_summary_json":
            errors.extend(_check_tau_summary(data, path))
        elif key == "bootstrap_json":
            errors.extend(_check_bootstrap(data, path))

    return errors, warnings


def main() -> None:
    p = argparse.ArgumentParser(description="Phase II-A bundle artifacts 検証")
    p.add_argument(
        "--bundle",
        type=Path,
        default=_ROOT / "experiments" / "baselines" / "v7_phase2a_mrmp_tau_n3146_bundle_v1.json",
        help="bundle pointer JSON",
    )
    p.add_argument("--repo-root", type=Path, default=_ROOT, help="リポジトリルート")
    p.add_argument(
        "--strict",
        action="store_true",
        help="成果物欠落でも失敗させる（再現実行後の CI 向け）",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help=(
            "bundle 既定の OUT_PREFIX を置換（例: experiments/logs/my_run）。"
            "省略時は環境変数 OUT_PREFIX。未設定なら bundle のパスをそのまま使う。"
        ),
    )
    p.add_argument(
        "--verify-manifest",
        type=Path,
        default=None,
        help="再現マニフェスト JSON（v7_phase2a_repro_manifest.py 出力）の SHA256 を検証",
    )
    args = p.parse_args()

    bundle_path = args.bundle.resolve()
    if not bundle_path.is_file():
        print(json.dumps({"error": "bundle_not_found", "path": str(bundle_path)}), file=sys.stderr)
        raise SystemExit(2)

    bundle = _load_json(bundle_path)
    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict):
        print(json.dumps({"error": "bundle_missing_artifacts"}), file=sys.stderr)
        raise SystemExit(2)

    out_prefix = (args.out_prefix or os.environ.get("OUT_PREFIX") or "").strip()
    bd = default_prefix_from_artifacts(artifacts)
    if out_prefix and bd:
        artifacts = remap_artifact_paths(artifacts, bundle_default_prefix=bd, out_prefix=out_prefix)

    combined: dict[str, str] = dict(artifacts)
    fig = bundle.get("figures") if isinstance(bundle.get("figures"), dict) else {}
    figp = bundle.get("figures_paper") if isinstance(bundle.get("figures_paper"), dict) else {}
    if out_prefix and bd:
        fig = _remap_path_dict(fig, bundle_default_prefix=bd, out_prefix=out_prefix)
        figp = _remap_path_dict(figp, bundle_default_prefix=bd, out_prefix=out_prefix)
    for fk, rel in fig.items():
        if isinstance(rel, str) and rel.strip():
            combined[f"figures:{fk}"] = rel.strip()
    for fk, rel in figp.items():
        if isinstance(rel, str) and rel.strip():
            combined[f"figures_paper:{fk}"] = rel.strip()

    repo = args.repo_root.resolve()
    errors, warnings = validate_artifacts(repo, combined, strict=args.strict)

    for w in warnings:
        print(f"v7_phase2a_bundle_validate_warn {w}", file=sys.stderr)
    if errors:
        print("v7_phase2a_bundle_validate_failed", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        raise SystemExit(1)

    if args.verify_manifest is not None:
        from experiments.v7_phase2a_repro_manifest import verify_manifest

        mp = args.verify_manifest.resolve()
        v_errs = verify_manifest(mp, repo=repo)
        if v_errs:
            print("v7_phase2a_bundle_validate_manifest_mismatch", file=sys.stderr)
            for e in v_errs:
                print(e, file=sys.stderr)
            raise SystemExit(1)

    print(
        "v7_phase2a_bundle_validate_ok",
        json.dumps(
            {
                "warnings": len(warnings),
                "strict": args.strict,
                "out_prefix_applied": bool(out_prefix and bd),
                "manifest_verified": str(args.verify_manifest.resolve()) if args.verify_manifest else None,
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
