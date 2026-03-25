"""
軽量共鳴パイプライン: 埋め込み → (N, d) 場 → DynamicROISelector + ResonanceEngine。
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .instrumentation import StageTimer
from .resonance import ResonanceEngine
from .roi_selector import DynamicROISelector


class LightweightResonanceFacade(nn.Module):
    """
    トークン列を固定ノード数 N の共鳴場に写像し、ROI 階層計算と
    ResonanceEngine による部分ノードのスコア化を返す。

    Returns の dict に roi_output, resonance_scores, resonance_tensor, context_vector を含む。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        resonance_dim: int = 6,
        num_nodes: int = 512,
        tau: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.d = resonance_dim
        self.num_nodes = num_nodes

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.to_resonance = nn.Linear(embed_dim, resonance_dim)
        self.roi = DynamicROISelector(num_nodes)
        self.engine = ResonanceEngine(num_nodes, resonance_dim, tau=tau)

    def forward(
        self,
        token_ids: torch.Tensor,
        top_k_engine=None,
        instrument: StageTimer | None = None,
        cultural_scale: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            token_ids: (B, S)
            top_k_engine: ResonanceEngine に渡すノードインデックス数（メモリ節約）。None なら min(64, N)。
            instrument: 与えた場合、各区間の経過時間（および CUDA 時は割り当て差分）を `instrument.records` に蓄積。
            cultural_scale: 任意。(B,1,1) またはブロードキャスト可能。共鳴特徴 r に乗算（Phase 1B）。

        Returns:
            dict with keys: resonance_tensor (N,d), context_vector (d,),
            roi_output (T,1), resonance_scores (K), engine_node_indices (K)
        """
        st = instrument.stage if instrument else lambda _n: nullcontext()

        with st("embedding"):
            x = self.embedding(token_ids)
        with st("to_resonance"):
            r = self.to_resonance(x)
        if cultural_scale is not None:
            r = r * cultural_scale
        b, s, _ = r.shape

        with st("adaptive_pool"):
            r_bt = r.transpose(1, 2)
            r_pooled = F.adaptive_avg_pool1d(r_bt, self.num_nodes)
            r_pooled = r_pooled.transpose(1, 2)
            resonance_tensor = r_pooled.mean(dim=0)
            current_state = r.mean(dim=(0, 1)).unsqueeze(0)

        with st("roi_select"):
            roi_out = self.roi.select_and_compute(current_state, resonance_tensor)

        k = top_k_engine if top_k_engine is not None else min(64, self.num_nodes)
        with st("engine_topk"):
            importance = torch.matmul(resonance_tensor, current_state.t()).squeeze()
            top_idx = torch.argsort(importance, descending=True)[:k]
            context_vector = current_state.squeeze(0)
            resonance_scores = self.engine(top_idx, context_vector)

        return {
            "resonance_tensor": resonance_tensor,
            "context_vector": context_vector,
            "roi_output": roi_out,
            "resonance_scores": resonance_scores,
            "engine_node_indices": top_idx,
        }
