import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

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
    
         
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_heads, dropout_prob = 0, rope = None):
        super().__init__()
        assert model_dim%n_heads==0, 'Model dimension should be devisible by the number of heads'
        self.d = model_dim
        self.dk = model_dim // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(self.d, self.d, bias=False)
        self.W_k = nn.Linear(self.d, self.d, bias=False)
        self.W_v = nn.Linear(self.d, self.d, bias=False)
        self.W_o = nn.Linear(self.d, self.d, bias=False)

        if dropout_prob > 0:
            self.attn_dropout = nn.Dropout(p=dropout_prob)
            self.out_dropout = nn.Dropout(p=dropout_prob)
        else:
            self.attn_dropout = None
            self.out_dropout = None
        self.rope = rope

    def forward(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 mask=None)-> Tuple[torch.Tensor, torch.Tensor]:
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
        B, T_q, _ = q.shape
        B, T_k, _ = k.shape
        Q = self.W_q(q) # BxTx3d
        K = self.W_k(k) # BxTx3d
        V = self.W_v(v) # BxTx3d
        Q = Q.view(B, T_q, self.n_heads, self.dk).transpose(1,2) # BxhxTxd
        K = K.view(B, T_k, self.n_heads, self.dk).transpose(1,2) # BxhxTxd
        V = V.view(B, T_k, self.n_heads, self.dk).transpose(1,2) # BxhxTxd
        if self.rope is not None:
            Q, K = self.rope(Q, K)
        A = Q@K.transpose(-1, -2)             #BxhxTxT 
        A = A/(self.dk**0.5)
        if mask is not None: # Mask should have the same shape as the attention map
            # ensure mask can broadcast to (B, n_heads, T, T)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            A = A.masked_fill(mask==0, float('-inf'))
            A = torch.clamp(A, min=-1e4, max=1e4)
        A = F.softmax(A, dim=-1) #BxhxTxT Softmax applied along the attention matrix rows for each head independently
        if self.attn_dropout is not None:
            A = self.attn_dropout(A)
        Y = torch.matmul(A, V) # BxhxTxdk
        Y = Y.transpose(1,2).contiguous() # BxTxhxdk
        Y = self.W_o(Y.view(B, T_q, self.d))
        if self.out_dropout is not None:
            Y = self.out_dropout(Y)
        return Y, A 