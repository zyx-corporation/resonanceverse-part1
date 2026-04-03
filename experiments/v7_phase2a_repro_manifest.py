"""
Phase II-A の再現・査読用マニフェスト（SHA256・環境メタ）を生成・検証する。

- **--pin-code-only**: リポジトリ同梱の事前登録・パイプラインスクリプトのみハッシュ（CI・ログ未生成でも可）
- 省略時: bundle の artifacts / figures / figures_paper のパスを解決し、存在するファイルをハッシュ

例::

    python experiments/v7_phase2a_repro_manifest.py \\
      --bundle experiments/baselines/v7_phase2a_mrmp_tau_n3146_bundle_v1.json --pin-code-only
    python experiments/v7_phase2a_repro_manifest.py \\
      --bundle experiments/baselines/v7_phase2a_mrmp_tau_n3146_judge10k_bundle_v1.json \\
      --out-prefix experiments/logs/v7_phase2a_mrmp_tau_n3146_judge10k \\
      --out experiments/logs/v7_phase2a_mrmp_tau_n3146_judge10k_repro_manifest.json
    python experiments/v7_phase2a_repro_manifest.py --verify path/to/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.v7_phase2a_bundle_validate import (  # noqa: E402
    default_prefix_from_artifacts,
    remap_artifact_paths,
)

_SCHEMA = "v7_phase2a_repro_manifest.v1"

# パイプラインの「コード正本」パス（git 管理下想定）
_PINNED_CODE_RELPATHS = (
    "experiments/v7_phase2a_empirical_run.py",
    "experiments/v7_phase2a_empirical.py",
    "experiments/v7_phase2a_tau_summary.py",
    "experiments/v7_phase2a_tau_bootstrap.py",
    "experiments/v7_phase2a_tau_plots.py",
    "experiments/v7_phase2a_bundle_validate.py",
    "experiments/v7_phase2a_repro_manifest.py",
    "experiments/run_phase2a_mrmp_tau.sh",
    "experiments/v7_phase1a_llm_judge_six_axes.py",
    "experiments/v7_llm_judge_slm_pair_agreement.py",
    "experiments/v7_local_env_check.py",
    "experiments/run_local_slm_phase1_smoke.sh",
    "experiments/run_local_slm_phase2_smoke.sh",
    "experiments/run_local_slm_smoke_all.sh",
    "experiments/run_local_slm_phase3_mrmp_chunk.sh",
    "experiments/run_local_slm_phase1_compare_en_ja.sh",
    "experiments/run_mrmp_llm_judge_chunks_hf_local.sh",
    "experiments/run_local_slm_judge_pair_agreement.sh",
    "experiments/prompts/v7_llm_judge_prompt_v1.json",
)


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _entry(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {"sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _remap_path_dict(d: dict[str, Any], *, bundle_default_prefix: str, out_prefix: str) -> dict[str, Any]:
    """remap_artifact_paths と同じ規則で、相対パス辞書を置換する。"""
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


def _collect_path_dicts(
    artifacts: dict[str, Any],
    figures: dict[str, Any],
    figures_paper: dict[str, Any],
) -> list[str]:
    rels: list[str] = []
    for block in (artifacts, figures, figures_paper):
        for v in block.values():
            if isinstance(v, str) and v.strip():
                rels.append(v.strip())
    return sorted(set(rels))


def build_manifest(
    *,
    repo: Path,
    bundle_path: Path,
    bundle: dict[str, Any],
    out_prefix: str | None,
    pin_code_only: bool,
    strict_artifacts: bool,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Returns (manifest_dict, fatal_errors, warnings).
    fatal: 事前登録・コード・strict 時の成果物欠落。
    warnings: 非 strict 時の成果物欠落など。
    """
    fatal: list[str] = []
    warnings: list[str] = []
    files: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    # 1) bundle JSON 自身
    be = _entry(bundle_path.resolve())
    if be:
        rel_self = str(bundle_path.resolve().relative_to(repo.resolve()))
        files[rel_self] = be

    prereg = bundle.get("prereg")
    if isinstance(prereg, str) and prereg.strip():
        p = (repo / prereg.strip()).resolve()
        e = _entry(p)
        if e:
            files[str(p.relative_to(repo.resolve()))] = e
        else:
            fatal.append(f"欠落（事前登録）: {prereg}")

    rs = bundle.get("reproduce_shell")
    if isinstance(rs, str) and rs.strip():
        p = (repo / rs.strip()).resolve()
        e = _entry(p)
        if e:
            files[str(p.relative_to(repo.resolve()))] = e
        else:
            fatal.append(f"欠落（reproduce_shell）: {rs}")

    for rel in _PINNED_CODE_RELPATHS:
        p = (repo / rel).resolve()
        e = _entry(p)
        if e:
            files[rel] = e
        else:
            fatal.append(f"欠落（ピン留めコード）: {rel}")

    if not pin_code_only:
        artifacts = bundle.get("artifacts")
        if not isinstance(artifacts, dict):
            fatal.append("bundle に artifacts がありません")
        else:
            art = dict(artifacts)
            bd = default_prefix_from_artifacts(art) or ""
            op = (out_prefix or "").strip()
            if op and bd:
                art = remap_artifact_paths(art, bundle_default_prefix=bd, out_prefix=op)
                fig = bundle.get("figures") if isinstance(bundle.get("figures"), dict) else {}
                figp = bundle.get("figures_paper") if isinstance(bundle.get("figures_paper"), dict) else {}
                fig = _remap_path_dict(fig, bundle_default_prefix=bd, out_prefix=op)
                figp = _remap_path_dict(figp, bundle_default_prefix=bd, out_prefix=op)
            else:
                fig = bundle.get("figures") if isinstance(bundle.get("figures"), dict) else {}
                figp = bundle.get("figures_paper") if isinstance(bundle.get("figures_paper"), dict) else {}

            for rel in _collect_path_dicts(art, fig, figp):
                p = (repo / rel).resolve()
                e = _entry(p)
                if e:
                    try:
                        key = str(p.relative_to(repo.resolve()))
                    except ValueError:
                        key = str(p)
                    files[key] = e
                else:
                    missing.append(rel)
                    if strict_artifacts:
                        fatal.append(f"欠落（成果物）: {rel}")
                    else:
                        warnings.append(f"欠落（成果物・省略可）: {rel}")

    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "bundle_path": str(bundle_path.resolve().relative_to(repo.resolve())),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pin_code_only": pin_code_only,
        "out_prefix_applied": bool(
            (out_prefix or "").strip() and default_prefix_from_artifacts(bundle.get("artifacts") or {})
        ),
        "files": dict(sorted(files.items())),
        "missing_optional_paths": missing,
    }
    return manifest, fatal, warnings


def verify_manifest(manifest_path: Path, *, repo: Path) -> list[str]:
    """マニフェストに記録された各ファイルの SHA256 を再計算。不一致・欠落はエラーリストに入れる。"""
    data = _load_json(manifest_path)
    if data.get("schema_version") != _SCHEMA:
        return [f"schema_version が {_SCHEMA} ではない: {manifest_path}"]

    errors: list[str] = []
    files = data.get("files")
    if not isinstance(files, dict):
        return [f"files オブジェクト不正: {manifest_path}"]

    for rel, meta in sorted(files.items()):
        if not isinstance(meta, dict):
            errors.append(f"エントリ不正: {rel}")
            continue
        expected = meta.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            errors.append(f"sha256 不正: {rel}")
            continue
        path = (repo / rel).resolve()
        if not path.is_file():
            errors.append(f"欠落: {rel}")
            continue
        got = _sha256_file(path)
        if got != expected:
            errors.append(f"SHA256 不一致: {rel} expected={expected[:12]}... got={got[:12]}...")
    return errors


def main() -> None:
    p = argparse.ArgumentParser(description="Phase II-A 再現マニフェスト（SHA256）")
    p.add_argument("--bundle", type=Path, default=None, help="bundle pointer JSON（--verify 時は省略可）")
    p.add_argument("--repo-root", type=Path, default=_ROOT, help="リポジトリルート")
    p.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="v7_phase2a_bundle_validate と同様に artifacts パスを置換（環境変数 OUT_PREFIX も可）",
    )
    p.add_argument(
        "--pin-code-only",
        action="store_true",
        help="コード・事前登録・bundle 本体のみハッシュ（experiments/logs 不要）",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="成果物欠落時は非ゼロ終了（pin-code-only では無視）",
    )
    p.add_argument("--out", type=Path, default=None, help="マニフェスト JSON の出力先")
    p.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="既存マニフェストを検証（一致しなければ非ゼロ終了）",
    )
    args = p.parse_args()

    repo = args.repo_root.resolve()

    if args.verify is not None:
        mp = args.verify.resolve()
        if not mp.is_file():
            print(json.dumps({"error": "manifest_not_found", "path": str(mp)}), file=sys.stderr)
            raise SystemExit(2)
        errs = verify_manifest(mp, repo=repo)
        if errs:
            print("v7_phase2a_repro_manifest_verify_failed", file=sys.stderr)
            for e in errs:
                print(e, file=sys.stderr)
            raise SystemExit(1)
        print("v7_phase2a_repro_manifest_verify_ok", json.dumps({"path": str(mp)}, ensure_ascii=False))
        return

    if args.bundle is None:
        print(json.dumps({"error": "bundle_required_without_verify"}), file=sys.stderr)
        raise SystemExit(2)

    bundle_path = args.bundle.resolve()
    if not bundle_path.is_file():
        print(json.dumps({"error": "bundle_not_found", "path": str(bundle_path)}), file=sys.stderr)
        raise SystemExit(2)

    bundle = _load_json(bundle_path)
    out_prefix = (args.out_prefix or os.environ.get("OUT_PREFIX") or "").strip()

    manifest, fatal, warns = build_manifest(
        repo=repo,
        bundle_path=bundle_path,
        bundle=bundle,
        out_prefix=out_prefix or None,
        pin_code_only=bool(args.pin_code_only),
        strict_artifacts=bool(args.strict),
    )

    if fatal:
        print("v7_phase2a_repro_manifest_failed", file=sys.stderr)
        for e in fatal:
            print(e, file=sys.stderr)
        raise SystemExit(1)

    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print("v7_phase2a_repro_manifest_ok", json.dumps({"written": str(out_path)}, ensure_ascii=False))
    else:
        sys.stdout.write(text)

    for w in warns:
        print(f"v7_phase2a_repro_manifest_warn {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
