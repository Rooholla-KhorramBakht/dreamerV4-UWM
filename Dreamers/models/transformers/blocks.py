import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional

class RopeEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, device = 'cpu'):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.device = device
        # Cache the sin/cos computations to avoid recomputation
        thetas = torch.tensor([10000**(-2*i/dim) for i in range(dim//2)])
        thetas = torch.stack([thetas, thetas], dim=0).transpose(0,1).reshape(-1)
        thetas_all = torch.stack([thetas*i for i in range(max_seq_len)], dim=0)
        cos_cache = thetas_all.cos() # TxD
        sin_cache = thetas_all.sin() # TxD
        self.register_buffer('cos_emb', cos_cache.unsqueeze(0).unsqueeze(0)) #1x1xTxD
        self.register_buffer('sin_emb', sin_cache.unsqueeze(0).unsqueeze(0)) #1x1xTxD

    def forward(self, q: torch.Tensor, k:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] :
        B, H, T, d = q.shape
        cos_emb = self.cos_emb[:, :, :T, :]  # 1x1xTxD
        sin_emb = self.sin_emb[:, :, :T, :]  # 1x1xTxD 
        q_odd, q_even = q[..., ::2], q[..., 1::2]
        qJ = torch.stack([-q_even, q_odd], dim=-1).reshape_as(q)
        k_odd, k_even = k[..., ::2], k[..., 1::2]
        kJ = torch.stack([-k_even, k_odd], dim=-1).reshape_as(k)
        
        q_rot = (q * cos_emb) + (qJ * sin_emb)
        k_rot = (k * cos_emb) + (kJ * sin_emb)
        return q_rot, k_rot
    

class Attention(nn.Module):
    def __init__(self, model_dim, n_heads, n_kv_heads=None, causal = False, dropout_prob = 0, qk_norm=True, max_seq_len=128, rope_embedder = None):
        super().__init__()
        assert model_dim%n_heads==0, 'Model dimension should be devisible by the number of heads'
        self.d = model_dim
        self.dk = model_dim // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.dropout_prob = dropout_prob
        self.rope_embedder = rope_embedder
        self.qk_norm = qk_norm
        self.causal = causal
        self.W_q = nn.Linear(self.d, self.d, bias=False)
        self.W_k = nn.Linear(self.d, self.dk*self.n_kv_heads, bias=False)
        self.W_v = nn.Linear(self.d, self.dk*self.n_kv_heads, bias=False)
        self.W_o = nn.Linear(self.d, self.d, bias=False)
        self.register_buffer("g", torch.tensor(math.log2(float(max_seq_len**2-max_seq_len)), dtype=torch.float32)) # The normalization constant in QK-Norm is active.

    def forward(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 mask: Optional[torch.Tensor] = None)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head (self/cross) attention forward pass.
        Args:
            q: (B, T_q, D) queries
            k: (B, T_k, D) keys
            v: (B, T_k, D) values
            mask: (B, 1, T_q, T_k) or broadcastable mask
        
        Returns:
            y: (B, T_q, D) attention output
            A: (B, n_heads, T_q, T_k) attention weights
        """
        if mask is not None and mask.dtype not in (torch.bool, torch.float32, torch.float16, torch.bfloat16):
            mask = mask.to(torch.bool)
        B, T_q, _ = q.shape
        B, T_k, _ = k.shape
        Q = self.W_q(q) # BxTx3d
        K = self.W_k(k) # BxTx3d
        V = self.W_v(v) # BxTx3d
        Q = Q.view(B, T_q, self.n_heads, self.dk).transpose(1,2) # BxhxTxd
        K = K.view(B, T_k, self.n_kv_heads, self.dk).transpose(1,2) # Bxn_kvxT_kxd
        V = V.view(B, T_k, self.n_kv_heads, self.dk).transpose(1,2) # Bxn_kvxT_kxd
        # Normalize the features per head if qk_norm is active
        if self.qk_norm:
            Q = F.normalize(Q, dim=-1)
            K = F.normalize(K, dim=-1)

        if self.rope_embedder is not None:
            Q, K = self.rope_embedder(Q, K)

        Y = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask, 
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=self.causal,
            scale = self.g,
            enable_gqa= False if self.n_kv_heads == self.n_heads else True,
        )  # [B, n_heads, Tq, dk]
        Y = Y.transpose(1, 2).contiguous().view(B, T_q, self.d)
        Y = self.W_o(Y)
        return Y
    

# class AxialAttentionBlock(nn.Module):
#     def __init__(self, model_dim, n_heads, dropout_prob = 0, qk_norm=True, max_seq_len=128, rope_embedder = None):
#         super().__init__()
#         assert model_dim%n_heads==0, 'Model dimension should be devisible by the number of heads'
#         self.d = model_dim
#         self.n_heads = n_heads
#         self.dropout_prob = dropout_prob
#         self.qk_norm = qk_norm
#         self.max_seq_len = max_seq_len
#         self.rope_embedder = rope_embedder
#         self.mha = MultiHeadAttention(model_dim=model_dim, 
#                                       n_heads=n_heads, 
#                                       dropout_prob=dropout_prob,
#                                       qk_norm=qk_norm, 
#                                       max_seq_len=max_seq_len, 
#                                       rope_embedder=rope_embedder)

#     def forward(self, q, k, v, dim, mask):
#         """
#         Axial attention along dim axis.
#         Args:
#             q: (B, T_q, D) queries
#             k: (B, T_k, D) keys
#             v: (B, T_k, D) values
#             dim: int dimension along which the attention should be applied
#             mask: (B, 1, T_q, T_k) or broadcastable mask
#         Returns:
#             y: (B, T_q, D) attention output
#             A: (B, n_heads, T_q, T_k) attention weights
#         """

