"""
v7 Phase I-A: 写像 φ の有効性 — アテンション非対称成分 S_asym と 6 軸ラベル（または代理特徴）の相関。

- --demo: 合成アテンション（transformers 不要）でパイプライン検証。
- 実モデル: --model gpt2 等で output_attentions から S_asym を計算（ラベル無し時は層別 S_asym 統計のみ）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

AXES = ("trust", "authority", "proximity", "intent", "affect", "history")


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(x, -50, 50))
    return e / e.sum(axis=-1, keepdims=True)


def s_asym_from_attention(a: np.ndarray) -> np.ndarray:
    """a: (L, L) row-softmax 済み想定。S_asym = A - A^T。"""
    return a - a.T


def frobenius_s_asym(s: np.ndarray) -> float:
    return float(np.linalg.norm(s, ord="fro"))


def run_synthetic_demo(*, seed: int, n_samples: int) -> dict[str, Any]:
    """
    合成: ラベル y_k = c_k * ||S_asym||_F + ノイズ → ピアソン相関が高いことを確認。
    """
    rng = np.random.default_rng(seed)
    f_list: list[float] = []
    y_lists: list[list[float]] = [[] for _ in range(6)]
    for _ in range(n_samples):
        seq_len = int(rng.integers(6, 14))
        logits = rng.standard_normal((seq_len, seq_len)).astype(np.float64)
        a = _softmax_rows(logits)
        s = s_asym_from_attention(a)
        f = frobenius_s_asym(s)
        f_list.append(f)
        for k in range(6):
            y = (0.3 + 0.1 * k) * f + rng.normal(0.0, 0.02 * f + 1e-6)
            y_lists[k].append(float(y))

    f_arr = np.array(f_list, dtype=np.float64)
    correlations: dict[str, dict[str, float]] = {}
    for k, name in enumerate(AXES):
        y_arr = np.array(y_lists[k], dtype=np.float64)
        if f_arr.std() < 1e-12 or y_arr.std() < 1e-12:
            r = float("nan")
            rho = float("nan")
        else:
            r = float(np.corrcoef(f_arr, y_arr)[0, 1])
            rho = float("nan")
            try:
                from scipy.stats import spearmanr

                rho = float(spearmanr(f_arr, y_arr).statistic)
            except ImportError:
                pass
        correlations[name] = {"pearson_r": r, "spearman_rho": rho}

    return {
        "schema_version": "v7_phase1a.v1",
        "mode": "synthetic_demo",
        "n_samples": n_samples,
        "seed": seed,
        "feature": "frobenius_norm_S_asym",
        "note": "合成ラベルは ||S_asym||_F の線形＋ノイズ。実コーパスでは 6 軸人手アノテとペアリング。",
        "correlations_vs_scalar_feature": correlations,
    }


def extract_hf_attention_layer_stats(
    *,
    text: str,
    model_name: str,
    cpu: bool,
    seed: int,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, int]:
    """
    層ごとの S_asym 統計を返す。

    Returns
    -------
    (layer_stats, error_payload, num_tokens)
        error_payload が None でないときは layer_stats は None。
    """
    import torch

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        err = {"schema_version": "v7_phase1a.v1", "error": f"transformers: {e}", "skipped": True}
        return None, err, 0

    from core.inference_device import select_inference_device
    from core.reproducibility import set_experiment_seed

    set_experiment_seed(seed)
    device = select_inference_device(force_cpu=cpu)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
    ).to(device)
    m.eval()
    ids = tok.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = m(ids, output_attentions=True)
    attns = out.attentions
    if not attns:
        ntok = int(ids.shape[1])
        err = {"schema_version": "v7_phase1a.v1", "error": "no attentions", "skipped": True}
        return None, err, ntok

    layer_stats: list[dict[str, Any]] = []
    for li, layer_attn in enumerate(attns):
        if layer_attn is None:
            continue
        a = layer_attn[0].mean(dim=0).cpu().numpy()
        s = s_asym_from_attention(a)
        layer_stats.append(
            {
                "layer": li,
                "frobenius_S_asym": frobenius_s_asym(s),
                "mean_abs_S_asym": float(np.mean(np.abs(s))),
            }
        )

    return layer_stats, None, int(ids.shape[1])


def hf_forward_hidden_last_layer(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    text: str,
    layer_index: int = -1,
) -> tuple[Any | None, dict[str, Any] | None, int]:
    """1 回前向きし、指定層の隠れ状態 ``(S, H)`` を返す（話者プール・w_ij 系列用）。"""
    import torch

    batch = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = batch["input_ids"].to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states
    ntok = int(ids.shape[1])
    if not hs:
        err = {"schema_version": "v7_phase1a.v1", "error": "no_hidden_states"}
        return None, err, ntok
    n_layers = len(hs)
    li = layer_index if layer_index >= 0 else n_layers - 1
    li = max(0, min(int(li), n_layers - 1))
    h = hs[li][0]
    return h, None, ntok


def hf_forward_attention_layer_matrix(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    text: str,
    layer_index: int,
) -> tuple[np.ndarray | None, dict[str, Any] | None, int]:
    """
    ロード済み Causal LM で 1 回前向きし、指定層のヘッド平均・行ソフトマックス注意行列
    (L, L) を返す（output_attentions の通常解釈）。
    """
    import torch

    batch = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = batch["input_ids"].to(device)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    attns = out.attentions
    ntok = int(ids.shape[1])
    if not attns:
        return None, {"schema_version": "v7_phase1a.v1", "error": "no attentions"}, ntok
    n_layers = len(attns)
    li = layer_index if layer_index >= 0 else n_layers - 1
    li = max(0, min(int(li), n_layers - 1))
    layer_attn = attns[li]
    if layer_attn is None:
        return None, {"schema_version": "v7_phase1a.v1", "error": "layer_attn_none"}, ntok
    a = layer_attn[0].mean(dim=0).cpu().numpy().astype(np.float64)
    return a, None, ntok


def run_hf_no_labels(
    *,
    model_name: str,
    text: str,
    cpu: bool,
    seed: int,
) -> dict[str, Any]:
    """ラベル無し: 層ごとの S_asym のノルム統計。"""
    layer_stats, err, ntok = extract_hf_attention_layer_stats(
        text=text,
        model_name=model_name,
        cpu=cpu,
        seed=seed,
    )
    if err is not None:
        return err
    assert layer_stats is not None
    return {
        "schema_version": "v7_phase1a.v1",
        "mode": "hf_attention_stats",
        "model": model_name,
        "text_len_tokens": ntok,
        "layers": layer_stats,
        "labels_available": False,
    }


def frobenius_s_asym_demo_from_text(text: str) -> float:
    """text のハッシュで決定論的な合成アテンションの ||S_asym||_F（HF 無し）。"""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    seq_len = int(min(max(len(text) // 4 + 3, 6), 24))
    logits = rng.standard_normal((seq_len, seq_len)).astype(np.float64)
    a = _softmax_rows(logits)
    s = s_asym_from_attention(a)
    return frobenius_s_asym(s)


def main() -> None:
    p = argparse.ArgumentParser(
        description="v7 Phase I-A: S_asym と 6 軸相関（demo / HF 統計）",
    )
    p.add_argument("--demo", action="store_true", help="合成のみ（CI 向け）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--text",
        default="Hello, this is a short test for attention asymmetry.",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.demo:
        payload = run_synthetic_demo(seed=args.seed, n_samples=args.n_samples)
    else:
        payload = run_hf_no_labels(
            model_name=args.model,
            text=args.text,
            cpu=args.cpu,
            seed=args.seed,
        )

    js = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(js, encoding="utf-8")
    print("v7_phase1a_ok", json.dumps({"mode": payload.get("mode")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
