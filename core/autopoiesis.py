class AutopoieticInference:
    """自己創生的な推論ループ"""
    def __init__(self, model, alpha=0.7, beta=0.2, gamma=0.1):
        self.model = model
        self.alpha = alpha  # 慣性
        self.beta = beta   # 相互作用
        self.gamma = gamma # ドリフト

    def step(self, input_ids, prev_resonance_field):
        # 現在のSLM出力を取得
        outputs = self.model(input_ids)
        current_interaction = outputs.last_hidden_state
        
        # オートポイエティック更新式 (論文4.1の完全開示式)
        # W_new = alpha * W_old + beta * interaction + gamma * drift
        new_field = (self.alpha * prev_resonance_field + 
                     self.beta * current_interaction + 
                     self.gamma * self.get_context_drift())
        
        return torch.clamp(new_field, -5.0, 5.0) # 安定性制約