"""
創発の作動的指標（混合基準線との KL、埋め込み新規性）とアブレーション結果のログ出力。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from core.lightweight_resonance import LightweightResonanceFacade
from core.reproducibility import set_experiment_seed


def softmax_rows(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.softmax(x, dim=dim)


def discrete_kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum()


def mixture_distribution(
    components: list[torch.Tensor],
    weights: list[float] | None = None,
) -> torch.Tensor:
    if weights is None:
        w = torch.ones(len(components), device=components[0].device, dtype=components[0].dtype)
    else:
        w = torch.tensor(weights, device=components[0].device, dtype=components[0].dtype)
    w = w / w.sum()
    stacked = torch.stack(components, dim=0)
    return (w.view(-1, *((1,) * (stacked.dim() - 1))) * stacked).sum(dim=0)


def kl_vs_mixture_baseline(
    observed_logits: torch.Tensor,
    baseline_logits_list: list[torch.Tensor],
    weights: list[float] | None = None,
) -> dict[str, float]:
    """観測分布 P と混合 Q の KL(P || Q)。各ベクトルはロジットでも確率でも可（内部で softmax）。"""
    p = softmax_rows(observed_logits.flatten(), dim=-1)
    q_components = [softmax_rows(x.flatten(), dim=-1) for x in baseline_logits_list]
    for c in q_components:
        if c.shape != p.shape:
            m = min(c.numel(), p.numel())
            p = p[:m]
            q_components = [t[:m] for t in q_components]
            break
    q = mixture_distribution(q_components, weights)
    m = min(p.numel(), q.numel())
    kl = discrete_kl_divergence(p[:m], q[:m])
    return {"kl_p_q": float(kl.item())}


def embedding_novelty(
    vectors: torch.Tensor,
    archive: torch.Tensor,
) -> dict[str, float]:
    """
    ドキュメント近似: N_t = 1 - max_j cos(e_t, a_j) の平均。
    vectors: (T, d), archive: (M, d)
    """
    v = F.normalize(vectors, dim=-1)
    a = F.normalize(archive, dim=-1)
    sim = v @ a.t()
    max_cos, _ = sim.max(dim=-1)
    novelty = (1.0 - max_cos).mean()
    return {"mean_novelty_1_minus_max_cos": float(novelty.item())}


@dataclass
class AblationConfig:
    shuffle_resonance_tensor: bool = False
    zero_roi: bool = False


def run_facade_with_ablation(
    facade: LightweightResonanceFacade,
    token_ids: torch.Tensor,
    cfg: AblationConfig,
) -> dict[str, Any]:
    x = facade.embedding(token_ids)
    r = facade.to_resonance(x)
    b, s, _ = r.shape
    r_bt = r.transpose(1, 2)
    r_pooled = F.adaptive_avg_pool1d(r_bt, facade.num_nodes)
    r_pooled = r_pooled.transpose(1, 2)
    resonance_tensor = r_pooled.mean(dim=0)
    if cfg.shuffle_resonance_tensor:
        perm = torch.randperm(resonance_tensor.size(0), device=resonance_tensor.device)
        resonance_tensor = resonance_tensor[perm]
    current_state = r.mean(dim=(0, 1)).unsqueeze(0)
    if cfg.zero_roi:
        out = torch.zeros(0, 1, device=resonance_tensor.device, dtype=resonance_tensor.dtype)
    else:
        out = facade.roi.select_and_compute(current_state, resonance_tensor)
    k = min(64, facade.num_nodes)
    importance = torch.matmul(resonance_tensor, current_state.t()).squeeze()
    top_idx = torch.argsort(importance, descending=True)[:k]
    context_vector = current_state.squeeze(0)
    scores = facade.engine(top_idx, context_vector)
    return {
        "roi_output_norm": float(out.norm().item()) if out.numel() else 0.0,
        "resonance_scores_norm": float(scores.norm().item()),
        "ablation": asdict(cfg),
    }


def write_log(
    preregistered: dict[str, Any],
    metrics: dict[str, Any],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "preregistered": preregistered,
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    yaml_path = out_path.with_name(out_path.stem + "_prereg.yaml")
    yaml_path.write_text(yaml.safe_dump(preregistered, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emergence metrics smoke + JSON log")
    parser.add_argument("--out", type=Path, default=Path("experiments/logs/emergence_run.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_experiment_seed(args.seed)
    pre = {
        "seed": args.seed,
        "vocab_size": 8000,
        "seq_len": 64,
        "num_nodes": 128,
        "tau_N_threshold": 0.01,
    }

    facade = LightweightResonanceFacade(
        vocab_size=8000,
        embed_dim=128,
        resonance_dim=6,
        num_nodes=128,
        tau=1.0,
    )
    tok = torch.randint(0, 8000, (2, 64))

    out_normal = run_facade_with_ablation(facade, tok, AblationConfig())
    out_shuffle = run_facade_with_ablation(facade, tok, AblationConfig(shuffle_resonance_tensor=True))

    obs = facade(tok)["resonance_scores"]
    baselines = [
        obs * 0.1 + torch.randn_like(obs) * 0.5,
        torch.roll(obs, shifts=1, dims=-1),
    ]
    kl_metrics = kl_vs_mixture_baseline(obs.flatten(), [b.flatten() for b in baselines])

    emb = facade.embedding(tok)[0]
    arch = emb[::4].detach()
    nov = embedding_novelty(emb, arch)

    metrics = {
        "ablation_normal": out_normal,
        "ablation_shuffle_tensor": out_shuffle,
        **kl_metrics,
        **nov,
    }
    write_log(pre, metrics, args.out)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
