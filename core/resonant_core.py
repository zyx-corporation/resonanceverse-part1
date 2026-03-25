from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cultural_modulation import CulturalModulationAdapter
from .instrumentation import StageTimer


def _masked_mean_over_batch_seq(
    resonance_weights: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """(B,S,d) をマスク付きで (d,) に平均。マスクが無い・全ゼロは従来どおり全平均。"""
    if attention_mask is None:
        return resonance_weights.mean(dim=(0, 1))
    m = attention_mask.to(dtype=resonance_weights.dtype)
    if m.numel() == 0 or m.sum() < 1e-6:
        return resonance_weights.mean(dim=(0, 1))
    num = (resonance_weights * m.unsqueeze(-1)).sum(dim=(0, 1))
    den = m.sum().clamp(min=1e-6)
    return num / den


class ResonantCore(nn.Module):
    """
    Resonanceverse の中核：動的共鳴と自己創生更新。

    - 入力 x: (Batch, Seq, Hidden) — 例：SLM の最終隠れ状態
    - 場 W: (1, num_nodes, d) — register_buffer。更新は copy_ でインプレース
    - **学習モード時のみ**場を更新（``self.training``）。``eval()`` では ``W`` は不変。
    - ``attention_mask`` があればパディング位置を除いて場の平均相互作用を計算する。
    - 場更新のドリフトは ``torch.Generator``（``field_drift_seed``、省略時は ``torch.initial_seed()`` 由来）で生成し、**同一初期化・同一入力列なら再現可能**。
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
        field_drift_seed: int | None = None,
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

        self._field_rng = torch.Generator()
        _ds = field_drift_seed
        if _ds is None:
            _ds = int(torch.initial_seed()) % (2**31 - 1)
        self._field_rng.manual_seed(int(_ds))

        self.dimension_projector = nn.Linear(hidden_size, self.d)
        self.theta = nn.Parameter(torch.tensor(float(initial_obscurity)))

        w0 = torch.randn(1, num_nodes, self.d) * 0.02
        self.register_buffer("W", w0)

    def forward(
        self,
        x: torch.Tensor,
        context_mask=None,
        attention_mask: torch.Tensor | None = None,
        instrument: StageTimer | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, S, hidden_size)
            context_mask: ``attention_mask`` の別名（後方互換）。1=有効トークン。
            attention_mask: (B, S) 省略時は全位置を有効とみなす。パディングを除いて場の平均を取る。
            instrument: 与えた場合、区間計測を `instrument.records` に蓄積。

        Returns:
            resonance_weights: (B, S, d) — 朧度でスケールされた共鳴重み
        """
        pad_mask = attention_mask if attention_mask is not None else context_mask
        st = instrument.stage if instrument else lambda _n: nullcontext()

        with st("project"):
            raw_resonance = self.dimension_projector(x)
        with st("softmax"):
            resonance_weights = F.softmax(raw_resonance / self.tau, dim=-1)

        with st("field_update"):
            if self.training:
                ci = _masked_mean_over_batch_seq(resonance_weights, pad_mask)
                current_interaction = ci.view(1, 1, self.d).expand(1, self.N, self.d).contiguous()

                drift = torch.randn(
                    1,
                    self.N,
                    self.d,
                    generator=self._field_rng,
                    dtype=torch.float32,
                )
                drift = drift.to(device=x.device, dtype=x.dtype) * self.drift_scale

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
    """Hugging Face 系 SLM の隠れ状態に ResonantCore を重ね、語彙ロジットを出す。

    語彙ヘッドは **最終隠れ状態と共鳴特徴（6 次元）を連結**して写像する（下流 `slm_downstream` の
    ``dual`` 読み出しと同型）。6 次元のみを ``Linear(6, V)`` に通す旧経路は情報が細りすぎ、
    因果 LM 学習が成立しなかった。
    """

    def __init__(self, base_slm_model, *, cultural_modulation: bool = False):
        super().__init__()
        self.base_model = base_slm_model
        h = int(base_slm_model.config.hidden_size)
        self.resonance_layer = ResonantCore(h)
        self.out_head = nn.Linear(h + 6, base_slm_model.config.vocab_size)
        self.cultural_adapter: CulturalModulationAdapter | None
        if cultural_modulation:
            self.cultural_adapter = CulturalModulationAdapter(h)
        else:
            self.cultural_adapter = None

    def forward(self, input_ids: torch.Tensor):
        outputs = self.base_model(input_ids, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        pad_mask = self._attention_mask_from_input_ids(input_ids)
        resonant_features = self.resonance_layer(last_hidden, attention_mask=pad_mask)
        if self.cultural_adapter is not None:
            resonant_features = resonant_features * self.cultural_adapter(last_hidden)
        z = torch.cat([last_hidden, resonant_features], dim=-1)
        logits = self.out_head(z)
        return logits

    def _attention_mask_from_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """pad_token_id が無いモデルでは全位置を有効とする。"""
        cfg = self.base_model.config
        pad_id = getattr(cfg, "pad_token_id", None)
        if pad_id is None:
            return torch.ones(
                input_ids.shape,
                dtype=torch.long,
                device=input_ids.device,
            )
        return (input_ids != pad_id).long()
