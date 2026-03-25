from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cultural_modulation import CulturalModulationAdapter
from .instrumentation import StageTimer


class ResonantCore(nn.Module):
    """
    Resonanceverse の中核：動的共鳴と自己創生更新。

    - 入力 x: (Batch, Seq, Hidden) — 例：SLM の最終隠れ状態
    - 場 W: (1, num_nodes, d) — register_buffer。更新は copy_ でインプレース
    """

    def __init__(
        self,
        hidden_size: int,
        num_nodes: int = 512,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        tau: float = 1.0,
        stability_bound: float = 5.0,
        drift_scale: float = 0.01,
        initial_obscurity: float = 0.15,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.N = num_nodes
        self.d = 6

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.stability_bound = stability_bound
        self.drift_scale = drift_scale

        self.dimension_projector = nn.Linear(hidden_size, self.d)
        self.theta = nn.Parameter(torch.tensor(float(initial_obscurity)))

        w0 = torch.randn(1, num_nodes, self.d) * 0.02
        self.register_buffer("W", w0)

    def forward(
        self,
        x: torch.Tensor,
        context_mask=None,
        instrument: StageTimer | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, hidden_size)
            context_mask: 未使用（将来拡張用）
            instrument: 与えた場合、区間計測を `instrument.records` に蓄積。

        Returns:
            resonance_weights: (B, S, d) — 朧度でスケールされた共鳴重み
        """
        st = instrument.stage if instrument else lambda _n: nullcontext()

        with st("project"):
            raw_resonance = self.dimension_projector(x)
        with st("softmax"):
            resonance_weights = F.softmax(raw_resonance / self.tau, dim=-1)

        with st("field_update"):
            ci = resonance_weights.mean(dim=(0, 1))
            current_interaction = ci.view(1, 1, self.d).expand(1, self.N, self.d).contiguous()

            drift = torch.randn(1, self.N, self.d, device=x.device, dtype=x.dtype)
            drift = drift * self.drift_scale

            updated = (
                self.alpha * self.W
                + self.beta * current_interaction
                + self.gamma * drift
            )
            updated = torch.clamp(updated, -self.stability_bound, self.stability_bound)
            self.W.copy_(updated)

        with st("obscurity_output"):
            scale = 1.0 - torch.sigmoid(self.theta)
            output = resonance_weights * scale
        return output


class AwaiIntegratedSLM(nn.Module):
    """Hugging Face 系 SLM の隠れ状態に ResonantCore を重ね、語彙ロジットを出す。"""

    def __init__(self, base_slm_model, *, cultural_modulation: bool = False):
        super().__init__()
        self.base_model = base_slm_model
        h = int(base_slm_model.config.hidden_size)
        self.resonance_layer = ResonantCore(h)
        self.out_head = nn.Linear(6, base_slm_model.config.vocab_size)
        self.cultural_adapter: CulturalModulationAdapter | None
        if cultural_modulation:
            self.cultural_adapter = CulturalModulationAdapter(h)
        else:
            self.cultural_adapter = None

    def forward(self, input_ids: torch.Tensor):
        outputs = self.base_model(input_ids, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        resonant_features = self.resonance_layer(last_hidden)
        if self.cultural_adapter is not None:
            resonant_features = resonant_features * self.cultural_adapter(last_hidden)
        logits = self.out_head(resonant_features)
        return logits
