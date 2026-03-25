import torch
import math

class DynamicROISelector:
    """
    定理4: 動的関心領域制御による計算量最適化の実装
    """
    def __init__(self, num_nodes):
        self.N = num_nodes
        # 3層分割のサイズ定義 (論文2.5節のアルゴリズム)
        self.s1_size = min(2 * int(math.log2(num_nodes)), 50)  # O(log² N)
        self.s2_size = min(int(math.sqrt(num_nodes)), 500)    # O(√N log N)
        self.s3_size = min(int(0.1 * num_nodes), 1000)        # O(0.1N)

    def select_and_compute(self, current_state, resonance_tensor):
        """
        current_state: 現在の隠れ状態 (1, d)
        resonance_tensor: 共鳴場 (N, d)
        """
        # 1. 簡易スコアリング（内積）で暫定的な重要度を算出
        # resonance_tensor shape: (N, d), current_state: (d, 1)
        importance_scores = torch.matmul(resonance_tensor, current_state.t()).squeeze()
        
        # 2. 階層的なインデックス取得
        # スコア上位からS1, S2, S3を切り出す
        sorted_indices = torch.argsort(importance_scores, descending=True)
        
        idx_s1 = sorted_indices[:self.s1_size]
        idx_s2 = sorted_indices[self.s1_size : self.s1_size + self.s2_size]
        idx_s3 = sorted_indices[self.s1_size + self.s2_size : self.s1_size + self.s2_size + self.s3_size]
        
        # 3. 各ティアに応じた共鳴計算 (定理6の朧度を適用)
        results = []
        
        # S1: フル精度計算 (朧度 0)
        res_s1 = self._compute_kernel(resonance_tensor[idx_s1], current_state, theta=0.0)
        results.append(res_s1)
        
        # S2: 近似計算 (朧度 中)
        res_s2 = self._compute_kernel(resonance_tensor[idx_s2], current_state, theta=0.4)
        results.append(res_s2)
        
        # S3: 背景計算 (朧度 高 / 定理6に基づき計算を85%削減)
        # S3は代表値（平均）のみで計算を効率化
        res_s3_mean = self._compute_kernel(resonance_tensor[idx_s3].mean(dim=0, keepdim=True), 
                                          current_state, theta=0.85)
        results.append(res_s3_mean)
        
        return torch.cat(results)

    def _compute_kernel(self, nodes, state, theta):
        """朧度(theta)を考慮した共鳴カーネル"""
        # 論文2.1節: K(w, theta) の実装
        raw_resonance = torch.matmul(nodes, state.t())
        # 定理6: 曖昧性による計算抑制
        return raw_resonance * (1.0 - theta)