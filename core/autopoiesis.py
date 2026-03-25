import torch


class AutopoieticInference:
    """自己創生的推論ループ（場の離散更新）。HF モデルまたは任意の (B,S,H) を返すモジュールに対応。"""

    def __init__(
        self,
        model,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        stability_bound: float = 5.0,
        drift_scale: float = 0.01,
    ):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.stability_bound = stability_bound
        self.drift_scale = drift_scale

    def get_context_drift(self, reference: torch.Tensor) -> torch.Tensor:
        """文脈ドリフト項。reference と同じ形状・デバイス・dtype。"""
        return torch.randn_like(reference) * self.drift_scale

    def step(
        self,
        input_ids: torch.Tensor,
        prev_resonance_field: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(input_ids)
        if hasattr(outputs, "last_hidden_state"):
            current_interaction = outputs.last_hidden_state
        else:
            current_interaction = outputs

        drift = self.get_context_drift(current_interaction)
        new_field = (
            self.alpha * prev_resonance_field
            + self.beta * current_interaction
            + self.gamma * drift
        )
        return torch.clamp(new_field, -self.stability_bound, self.stability_bound)
