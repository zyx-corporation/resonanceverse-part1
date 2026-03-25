import torch
import torch.nn as nn
import torch.nn.functional as F

class ResonanceEngine(nn.Module):
    """
    動的共鳴テンソルの演算と共鳴スコアの算出を担う中核クラス
   
    """
    def __init__(self, num_nodes, dim, tau=1.0):
        super().__init__()
        self.N = num_nodes  # ノード数
        self.d = dim        # 評価次元数 (基本6次元)
        self.tau = tau      # 温度パラメータ (定理5)
        
        # 共鳴テンソル W(t) ∈ ℝ^(N×N×d) の初期化
        # Xavier初期化に基づき、勾配消失を防ぐ
        limit = torch.sqrt(torch.tensor(2.0 / (2 * num_nodes)))
        self.W = nn.Parameter(torch.randn(num_nodes, num_nodes, dim) * limit)
        
    def forward(self, node_indices, context_vector):
        """
        特定のノードセットと文脈に基づく共鳴計算
       
        """
        # 現在の対象ノードに対するテンソルのスライスを取得
        # shape: (batch_size, num_active_nodes, d)
        w_ij = self.W[node_indices]
        
        # 文脈ベクトルとの相互作用による動的重み付け (定理2)
        # K(w_ij, theta) の簡易実装としてアダマール積とSoftmaxを適用
        resonance_energy = torch.einsum('bnd,d->bn', w_ij, context_vector)
        
        # 最大エントロピー原理に基づく共鳴スコアの正規化 (定理5)
        # p*(r_i) = exp(r_i/τ) / Σ exp(r_j/τ)
        resonance_scores = F.softmax(resonance_energy / self.tau, dim=-1)
        
        return resonance_scores

    def get_dimension_intensity(self):
        """
        各次元（信頼、情動等）の現在の共鳴強度を算出
        """
        return torch.mean(self.W, dim=(0, 1))