import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint
import numpy as np

class KVCache:
    """
    Simple rolling buffer for KV caching.
    """
    def __init__(self, max_seq_len: int, batch_size: int, num_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len = max_seq_len
        self.curr_len = 0
        # Preallocate buffers [B, H, S, D]
        self.k_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k_new, v_new: [B, H, S_new, D] (usually S_new=1)
        Returns full concatenated history [B, H, S_total, D]
        """
        B, H, S_new, D = k_new.shape
        
        # If buffer is full, roll (shift left)
        if self.curr_len + S_new > self.max_seq_len:
            shift = (self.curr_len + S_new) - self.max_seq_len
            self.k_cache = torch.roll(self.k_cache, shifts=-shift, dims=2)
            self.v_cache = torch.roll(self.v_cache, shifts=-shift, dims=2)
            self.curr_len -= shift
            
        # Insert new tokens
        self.k_cache[:, :, self.curr_len : self.curr_len + S_new, :] = k_new
        self.v_cache[:, :, self.curr_len : self.curr_len + S_new, :] = v_new
        
        self.curr_len += S_new
        
        # Return valid slice
        return self.k_cache[:, :, :self.curr_len, :], self.v_cache[:, :, :self.curr_len, :]
    
    def reset(self):
        self.curr_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

# --------- Utilities ---------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FeedForwardSwiGLU(nn.Module):
    """
    SwiGLU FFN:
      y = down( dropout( SiLU(up(x)) * gate(x) ) )
    Hidden size defaults to ~8/3 * d_model as commonly used in LLaMA/HF.
    """
    def __init__(self, d_model: int, d_hidden: int | None = None, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        if d_hidden is None:
            # 8/3 expansion; round to multiple of 8 for tensor cores if desired
            d_hidden = (8 * d_model) // 3
        self.up = nn.Linear(d_model, d_hidden, bias=bias)
        self.gate = nn.Linear(d_model, d_hidden, bias=bias)
        self.down = nn.Linear(d_hidden, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    #@torch.compile(fullgraph=False, mode="reduce-overhead", dynamic=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.silu(self.up(x))
        g = self.gate(x)
        h = self.drop(u * g)
        return self.down(h)

class MHA_GQA(nn.Module):
    """
    Multi-head attention with optional grouped-query attention (GQA).
    - num_heads_q: number of query heads (Hq)
    - num_heads_kv: number of key/value heads (G), G divides Hq
    - head_dim: Dh
    Expects:
      q_src: [..., batch_size, Sq, d_model]
      kv_src: [..., batch_size, Skv, d_model]
      The leading '...' dimensions must match between q_src and kv_src.
    Returns:
      out: [..., batch_size, Sq, d_model]
    """
    def __init__(self, d_model: int, num_heads_q: int, num_heads_kv: int, head_dim: int, dropout: float = 0.0, max_seq_len: int = 8192, rope_base: float = 10000.0):
        super().__init__()
        assert d_model == num_heads_q * head_dim, "d_model must equal Hq * Dh for simplicity"
        assert num_heads_q % num_heads_kv == 0, "Hq must be divisible by G (num_heads_kv)"
        assert head_dim % 2 == 0  # RoPE rotates 2D pairs
        self.d_model = d_model
        self.Hq = num_heads_q
        self.G = num_heads_kv
        self.Dh = head_dim
        self.Wq = nn.Linear(d_model, self.Hq * self.Dh, bias=False)
        self.Wk = nn.Linear(d_model, self.G * self.Dh, bias=False)
        self.Wv = nn.Linear(d_model, self.G * self.Dh, bias=False)
        self.out = nn.Linear(self.Hq * self.Dh, d_model, bias=False)
        self.dropout = dropout

        # QKNorm scaling parameter g, initialized as in the paper:
        # g0 = log2(L^2 - L), with L ≈ max_seq_len here.
        self.max_seq_len = max_seq_len
        L = float(max_seq_len)
        g0 = math.log2(L * L - L)
        #self.qk_scale = nn.Parameter(torch.tensor([g0], dtype=torch.float32))

        # RoPE: Precompute inverse frequencies for rotations
        # theta_i = base^(-2i/Dh) for i in [0, Dh/2)
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin cache for max_seq_len positions
        self._build_rope_cache(max_seq_len)

    def _build_rope_cache(self, seq_len: int):
        """Precompute cos and sin for positions [0, seq_len)."""
        # positions: [seq_len]
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # freqs: [seq_len, Dh//2] = outer(positions, inv_freq)
        freqs = torch.outer(positions, self.inv_freq)  # [seq_len, Dh//2]
        # Expand to full head_dim by repeating each frequency for pairs (x_2i, x_2i+1)
        # cos/sin cache: [seq_len, Dh//2] -> [seq_len, Dh] via repeat_interleave
        cos_cache = freqs.cos().repeat_interleave(2, dim=-1)  # [seq_len, Dh]
        sin_cache = freqs.sin().repeat_interleave(2, dim=-1)  # [seq_len, Dh]
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def _apply_rope(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply RoPE to x of shape [..., num_heads, seq_len, head_dim].
        position_ids: Optional [..., seq_len] to select custom positions from cache.
        Returns rotated x of same shape.
        """
        *prefix, num_heads, seq_len, head_dim = x.shape
        
        # If position_ids not provided, assume positions [0, 1, ..., seq_len-1]
        if position_ids is None:
            cos = self.cos_cache[:seq_len]  # [seq_len, Dh]
            sin = self.sin_cache[:seq_len]  # [seq_len, Dh]
        else:
            # Custom positions: gather from cache
            cos = self.cos_cache[position_ids]  # [..., seq_len, Dh]
            sin = self.sin_cache[position_ids]  # [..., seq_len, Dh]

        # Explicit broadcast over heads (looks useless, but actually can avoid silent issues)
        # + casting to avoid mixed-precision mismatches
        cos = cos.to(x.dtype).unsqueeze(-3)  # [..., 1, S, Dh]
        sin = sin.to(x.dtype).unsqueeze(-3)  # [..., 1, S, Dh]

        # Rotate: x_rot = x * cos + rotate_half(x) * sin
        x_rot = (x * cos) + (self._rotate_half(x) * sin)
        return x_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate interleaved pairs: [x0, x1, x2, x3, x4, x5, ...] -> [-x1, x0, -x3, x2, -x5, x4, ...]
        For each pair (x_{2i}, x_{2i+1}), produce (-x_{2i+1}, x_{2i}).
        """
        # Reshape to isolate pairs: [..., seq, head_dim] -> [..., seq, head_dim//2, 2]
        x = x.unflatten(-1, (-1, 2))  # [..., seq, head_dim//2, 2]
        # Reverse pairs and negate the first element: (x0, x1) -> (-x1, x0)
        x_rotated = torch.stack((-x[..., 1], x[..., 0]), dim=-1)  # [..., seq, head_dim//2, 2]
        # Flatten back to original shape
        return x_rotated.flatten(-2)  # [..., seq, head_dim]

    def _proj_q(self, x):  # [..., Sq, d] -> [..., Hq, Sq, Dh]
        q = self.Wq(x)  # [..., Sq, Hq * Dh]
        q = q.view(*q.shape[:-1], self.Hq, self.Dh)  # [..., Sq, Hq, Dh]
        q = q.transpose(-3, -2)  # [..., Hq, Sq, Dh]
        return q

    def _proj_kv(self, x):  # [..., Skv, d] -> [..., G, Skv, Dh] for both k and v
        k = self.Wk(x)  # [..., Skv, G * Dh]
        k = k.view(*k.shape[:-1], self.G, self.Dh)  # [..., Skv, G, Dh]
        k = k.transpose(-3, -2)  # [..., G, Skv, Dh]
        v = self.Wv(x)  # [..., Skv, G * Dh]
        v = v.view(*v.shape[:-1], self.G, self.Dh)  # [..., Skv, G, Dh]
        v = v.transpose(-3, -2)  # [..., G, Skv, Dh]
        return k, v

    def forward(self, q_src, kv_src, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False,
                q_position_ids: Optional[torch.Tensor] = None, kv_position_ids: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None, update_cache: bool = True):
        # Compute projections (handle arbitrary prefix)
        q = self._proj_q(q_src)  # [..., Hq, Sq, Dh]
        k, v = self._proj_kv(kv_src)  # [..., G, Skv, Dh]

        # --- KV Caching Logic ---
        if kv_cache is not None:
            # If caching, we assume q_src is the *new* query (Sq=1) 
            # and kv_src is the *new* KV (Skv=1)
            
            # Apply RoPE to the NEW k only
            # NOTE: RoPE needs the correct absolute position index for the new token.
            # The caller must pass correct kv_position_ids for the current step.
            k = self._apply_rope(k, position_ids=kv_position_ids)
            
            if update_cache:
                # Standard behavior: Add new keys to cache
                k, v = kv_cache.update(k, v)
            else:
                # Read-only behavior: 
                # We need to attend to [Cache, New_K].
                # So we manually concat without updating the ring buffer.
                
                # 1. Retrieve existing cache
                # cache_k: [B, H, S_curr, D]
                cache_k = kv_cache.k_cache[:, :, :kv_cache.curr_len, :]
                cache_v = kv_cache.v_cache[:, :, :kv_cache.curr_len, :]
                
                # 2. Concat with current step's K/V
                # k: [B, H, 1, D] -> [B, H, S_curr + 1, D]
                k = torch.cat([cache_k, k], dim=2)
                v = torch.cat([cache_v, v], dim=2)
            
            # RoPE for Q (current step)
            q = self._apply_rope(q, position_ids=q_position_ids)
            
        else:
            # Standard behavior (uncached)
            q = self._apply_rope(q, position_ids=q_position_ids)
            k = self._apply_rope(k, position_ids=kv_position_ids)

        # --- QKNorm: ℓ2-normalize along head_dim, then scale Q by learnable g ---
        # Normalize per (head, position) vector across Dh
        """q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)
        # Apply scalar scale g to Q (equivalent to scaling logits by g)
        q = q * self.qk_scale"""

        if q.shape[-2] == 1:
             effective_causal = False
        else:
             effective_causal = is_causal

        # Merge prefix dims if >1 (e.g., [B, T, Hq, Sq, Dh] -> [B*T, Hq, Sq, Dh]) for Flash Attention 4D
        prefix_shape = q.shape[:-3]
        if len(prefix_shape) > 1:
            prefix_prod = math.prod(prefix_shape)
            q = q.reshape(prefix_prod, *q.shape[-3:])  # [prefix_prod, Hq, Sq, Dh]
            k = k.reshape(prefix_prod, *k.shape[-3:])  # [prefix_prod, G, Skv, Dh]
            v = v.reshape(prefix_prod, *v.shape[-3:])  # [prefix_prod, G, Skv, Dh]
            # SDPA on merged 4D
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    query=q, key=k, value=v,
                    attn_mask=attn_mask, is_causal=effective_causal,
                    dropout_p=(self.dropout if self.training else 0.0),
                    enable_gqa=(self.G != self.Hq),
                    #scale=1.0,  # disable 1/sqrt(Dh) since QKNorm already controls scale
                )  # [prefix_prod, Hq, Sq, Dh]
            # Unmerge: restore prefix
            out = out.view(*prefix_shape, *out.shape[-3:])  # [..., Hq, Sq, Dh]
        else:
            # Direct 4D (standard [B, Hq, Sq, Dh])
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    query=q, key=k, value=v,
                    attn_mask=attn_mask, is_causal=effective_causal,
                    dropout_p=(self.dropout if self.training else 0.0),
                    enable_gqa=(self.G != self.Hq),
                    #scale=1.0,
                )  # [..., Hq, Sq, Dh]
        
        # Reshape for output projection: [..., Hq, Sq, Dh] -> [..., Sq, Hq, Dh] -> [..., Sq, Hq * Dh]
        out = out.transpose(-3, -2).contiguous()
        out = out.view(*out.shape[:-2], self.Hq * self.Dh)
        out = self.out(out)  # [..., Sq, d_model]

        return out

class BlockCausalEncoderLayer(nn.Module):
    """
    One encoder layer with either spatial-only or temporal-only attention.
    Routing (encoder):
      - Latent queries attend to {latents ⊕ patches}
      - Patch queries attend to {patches} only
    """
    def __init__(
        self,
        d_model: int,
        num_heads_q: int,
        num_heads_kv: int,
        seq_len: int,
        num_latents: int,
        num_patches: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        temporal: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        Dh = d_model // num_heads_q
        self.temporal = temporal
        self.is_last = is_last
        # Separate attention blocks for latent and patch streams
        self.attn_latent = MHA_GQA(d_model, num_heads_q, num_heads_kv, Dh, dropout=dropout, max_seq_len=seq_len if temporal else num_patches+num_latents)
        if not is_last:
            self.attn_patch = MHA_GQA(d_model, num_heads_q, num_heads_kv if False else num_heads_q, Dh, dropout=dropout, max_seq_len=seq_len if temporal else num_patches)
        # Note: for patches we usually set G = Hq (i.e., standard MHA) since routing is within-modality
        # LayerNorms
        self.ln_L_q = nn.RMSNorm(d_model)
        self.ln_L_kv = nn.RMSNorm(d_model)
        if not is_last:
            self.ln_P_q = nn.RMSNorm(d_model)
            self.ln_P_kv = nn.RMSNorm(d_model)
        # MLPs
        #self.ffn_L = FeedForward(d_model, int(mlp_ratio * d_model), dropout)
        #self.ffn_P = FeedForward(d_model, int(mlp_ratio * d_model), dropout)
        self.ffn_L = FeedForwardSwiGLU(d_model, None, dropout)
        if not is_last:
            self.ffn_P = FeedForwardSwiGLU(d_model, None, dropout)
        self.ln_L_ff = nn.RMSNorm(d_model)
        if not is_last:
            self.ln_P_ff = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        P: torch.Tensor,  # [B, T, Np, d]
        L: torch.Tensor,  # [B, T, Nl, d]
        causal_mask_temporal: Optional[torch.Tensor] = None,  # unused
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_last:
            _, _, Np, _ = P.shape
        B, T, Nl, d = L.shape

        # Wrap expensive operations in checkpoint
        def _forward_spatial(P, L):
            union_sp = torch.cat([L, P], dim=2)
            L_att = self.attn_latent(self.ln_L_q(L), self.ln_L_kv(union_sp), None, False)
            L_out = L + self.dropout(L_att)
            L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))

            if not self.is_last:
                P_att = self.attn_patch(self.ln_P_q(P), self.ln_P_kv(P), attn_mask=None, is_causal=False)
                P_out = P + self.dropout(P_att)
                P_out = P_out + self.dropout(self.ffn_P(self.ln_P_ff(P_out)))
            else:
                P_out = None

            return P_out, L_out
        
        def _forward_temporal(P, L):
            """union_LP = torch.cat([L, P], dim=2)  # [B, T, Np + Nl, d]
            union_LP = union_LP.permute(0, 2, 1, 3)  # [B, Np + Nl, T, d]

            union_LP_att = self.attn_latent(self.ln_L_q(union_LP), self.ln_L_kv(union_LP), attn_mask=None, is_causal=True)
            union_LP_out = union_LP + self.dropout(union_LP_att)
            union_LP_out = union_LP_out + self.dropout(self.ffn_L(self.ln_L_ff(union_LP_out)))

            union_LP_out = union_LP_out.permute(0, 2, 1, 3)  # [B, T, Np + Nl, d]
            L_out = union_LP_out[:, :, :Nl]
            P_out = union_LP_out[:, :, Nl:]"""

            L = L.permute(0, 2, 1, 3)  # [B, Nl, T, d]

            if not self.is_last:
                P = P.permute(0, 2, 1, 3)  # [B, Np, T, d]
                P_att = self.attn_patch(self.ln_P_q(P), self.ln_P_kv(P), attn_mask=None, is_causal=True)
                P_out = P + self.dropout(P_att)
                P_out = P_out + self.dropout(self.ffn_P(self.ln_P_ff(P_out)))
                P_out = P_out.permute(0, 2, 1, 3)  # [B, T, Np, d]
            else:
                P_out = None

            L_att = self.attn_latent(self.ln_L_q(L), self.ln_L_kv(L), attn_mask=None, is_causal=True)
            L_out = L + self.dropout(L_att)
            L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
            L_out = L_out.permute(0, 2, 1, 3)  # [B, T, Nl, d]

            return P_out, L_out
        
        # Use checkpointing for training
        if self.training and False:
            if not self.temporal:
                P_out, L_out = checkpoint(_forward_spatial, P, L, use_reentrant=False)
            else:
                P_out, L_out = checkpoint(_forward_temporal, P, L, use_reentrant=False)
        else:
            if not self.temporal:
                P_out, L_out = _forward_spatial(P, L)
            else:
                P_out, L_out = _forward_temporal(P, L)
        
        return P_out, L_out

class DreamerV4Encoder(nn.Module):
    """
    Block-causal tokenizer encoder:
      - Alternates spatial-only and temporal-only layers
      - Encoder routing:
          Latents attend to {latents ⊕ patches}; patches attend to {patches} only.
      - Produces latent bottleneck Z via linear down-projection + tanh.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        d_model: int,
        n_layers: int,
        num_heads_q: int,
        num_heads_kv_latent: int,
        seq_len: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        n_latents: int = 256,
        bottleneck_dim: int = 16,
        temporal_every: int = 1,  # 1 => alternate: spatial, temporal, spatial, temporal, ...
        in_channels: int = 3,
        mae_max_mask_prob: float = 0.9,
        activate_masking: bool = False
    ):
        super().__init__()
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.H, self.W = H, W
        self.P = patch_size
        self.Np = (H // patch_size) * (W // patch_size)
        self.Nl = n_latents
        self.d = d_model
        self.mae_max_mask_prob = mae_max_mask_prob
        self.activate_masking = activate_masking

        # Patch embedding via Conv2d (kernel=stride=patch_size)
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        # Learned mask token for MAE-style patch dropout
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Learned initial latent tokens per frame
        self.latent_tokens = nn.Parameter(torch.randn(n_latents, d_model) / math.sqrt(d_model))

        # Stack of layers
        layers = []
        for i in range(n_layers):
            temporal = (i % 2 == 1) if temporal_every == 1 else (((i + 1) % temporal_every) == 0)
            layers.append(
                BlockCausalEncoderLayer(
                    d_model=d_model,
                    num_heads_q=num_heads_q,
                    num_heads_kv=num_heads_kv_latent,  # used by latent stream; patch stream uses Hq internally
                    seq_len=seq_len,
                    num_latents=self.Nl,
                    num_patches=self.Np,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    temporal=temporal,
                    is_last=(i == n_layers-1) or (i == n_layers-2 and (n_layers % temporal_every) == 0)
                )
            )
        self.layers = nn.ModuleList(layers)

        # Bottleneck readout from latents: d -> d_b with tanh
        self.down_proj = nn.Linear(d_model, bottleneck_dim)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W] -> patch tokens [B, T, Np, d]
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        tok = self.patch_embed(x)  # [B*T, d, H/P, W/P]
        tok = tok.flatten(2).transpose(1, 2)  # [B*T, Np, d]
        tok = tok.view(B, T, self.Np, self.d)
        return tok

    def forward(self, video: torch.Tensor, mask = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        video: [B, T, C, H, W]
        Returns:
          P_enc: [B, T, Np, d]   (encoded patch tokens)
          L_enc: [B, T, Nl, d]   (encoded latent tokens)
          Z:     [B, T, Nl, d_b] (tanh bottleneck from latents)
        """
        B, T, C, H, W = video.shape
        assert (H, W) == (self.H, self.W)

        # Prepare tokens
        P = self.patchify(video)  # [B,T,Np,d]

        # --- Masked Autoencoding (MAE) patch dropout, as in DreamerV4 ---
        if (self.training or self.activate_masking) and self.mae_max_mask_prob > 0.0:
            B, T, Np, D = P.shape

            # Sample per-(B,T) dropout probability p ~ U(0, mae_max_mask_prob)
            # Shape: [B, T]
            drop_p = torch.rand(B, T, device=P.device, dtype=P.dtype) * self.mae_max_mask_prob

            # For each patch in each frame, sample Bernoulli with prob=drop_p[b, t]
            # rand: [B, T, Np]
            rand = torch.rand(B, T, Np, device=P.device, dtype=P.dtype)
            mask = rand < drop_p.unsqueeze(-1) if mask is None else mask  # bool [B, T, Np]

            # Replace masked patches with learned mask_token
            # mask.unsqueeze(-1): [B, T, Np, 1] -> broadcast over channel dimension
            P = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.to(P.dtype),  # [1, 1, 1, D] broadcasts to [B,T,Np,D]
                P,
            )
        # --- end MAE ---

        L0 = self.latent_tokens[None, None, :, :].expand(B, T, self.Nl, self.d).contiguous()  # [B,T,Nl,d]
        P_enc, L_enc = P, L0

        # Apply layers
        for layer in self.layers:
            P_enc, L_enc = layer(P_enc, L_enc, causal_mask_temporal=None)

        # Bottleneck readout from latents: d -> d_b + tanh
        Z = torch.tanh(self.down_proj(L_enc))  # [B,T,Nl,d_b]
        if self.activate_masking:
            return P_enc, L_enc, Z, mask
        else:
            return P_enc, L_enc, Z

class BlockCausalDecoderLayer(nn.Module):
    """
    One decoder layer with either spatial-only or temporal-only attention.
    Routing (decoder):
      - Latent queries attend to {latents} only
      - Patch-readout queries attend to {latents ⊕ patches}
    """
    def __init__(
        self,
        d_model: int,
        num_heads_q: int,
        num_heads_kv_latent: int,   # G for latent stream (GQA optional if < Hq)
        seq_len: int,
        num_latents: int,
        num_patches: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        temporal: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        Dh = d_model // num_heads_q
        self.temporal = temporal
        self.is_last = is_last

        # Attention blocks
        # Latent stream: K/V are latents only; allow GQA via num_heads_kv_latent
        if not is_last:
            self.attn_latent = MHA_GQA(d_model, num_heads_q, num_heads_kv_latent, Dh, dropout=dropout, max_seq_len=seq_len if temporal else num_latents)
        # Patch stream: K/V are latents ⊕ patches; use MHA on the union (set G=Hq)
        self.attn_patch = MHA_GQA(d_model, num_heads_q, num_heads_q, Dh, dropout=dropout, max_seq_len=seq_len if temporal else num_latents+num_patches)

        # Norms and MLPs
        if not is_last:
            self.ln_L_q = nn.RMSNorm(d_model)
            self.ln_L_kv = nn.RMSNorm(d_model)

        self.ln_P_q = nn.RMSNorm(d_model)
        self.ln_P_kv = nn.RMSNorm(d_model)

        #self.ffn_L = FeedForward(d_model, int(mlp_ratio * d_model), dropout)
        #self.ffn_P = FeedForward(d_model, int(mlp_ratio * d_model), dropout)
        if not is_last:
            self.ffn_L = FeedForwardSwiGLU(d_model, None, dropout)
        self.ffn_P = FeedForwardSwiGLU(d_model, None, dropout)

        if not is_last:
            self.ln_L_ff = nn.RMSNorm(d_model)
        self.ln_P_ff = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        R: torch.Tensor,  # [B, T, Np, d] patch readout tokens
        L: torch.Tensor,  # [B, T, Nl, d] projected latents
        kv_cache_l: Optional[KVCache] = None,
        kv_cache_r: Optional[KVCache] = None,
        position_ids: Optional[torch.Tensor] = None,
        update_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, Np, d = R.shape
        if not self.is_last:
            _, _, Nl, _ = L.shape

        # Wrap expensive operations in checkpoint
        def _forward_spatial(R, L):
            union_sp = torch.cat([L, R], dim=2)  # [B, T, Nl + Np, d]

            if not self.is_last:
                L_att = self.attn_latent(self.ln_L_q(L), self.ln_L_kv(L), attn_mask=None, is_causal=False)
                L_out = L + self.dropout(L_att)
                L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
            else:
                L_out = None

            R_att = self.attn_patch(self.ln_P_q(R), self.ln_P_kv(union_sp), attn_mask=None, is_causal=False)
            R_out = R + self.dropout(R_att)
            R_out = R_out + self.dropout(self.ffn_P(self.ln_P_ff(R_out)))

            return R_out, L_out

        def _forward_temporal(R, L, kv_cache_l, kv_cache_r, position_ids, update_cache):
            # --- 1. Patch Stream (R) ---
            B, T, Np, d = R.shape
            R_flat = R.transpose(1, 2).reshape(B * Np, T, d) # [B*Np, T, d]

            R_att = self.attn_patch(
                self.ln_P_q(R_flat),
                self.ln_P_kv(R_flat),
                is_causal=True,
                kv_cache=kv_cache_r,
                q_position_ids=position_ids,
                kv_position_ids=position_ids,
                update_cache=update_cache
            )

            R_out = R_flat + self.dropout(R_att)
            R_out = R_out + self.dropout(self.ffn_P(self.ln_P_ff(R_out)))
            R_out = R_out.view(B, Np, T, d).transpose(1, 2) # Restore [B, T, Np, d]

            # --- 2. Latent Stream (L) ---
            if not self.is_last:
                _, _, Nl, _ = L.shape
                L_flat = L.transpose(1, 2).reshape(B * Nl, T, d) # [B*Nl, T, d]

                L_att = self.attn_latent(
                    self.ln_L_q(L_flat),
                    self.ln_L_kv(L_flat),
                    is_causal=True,
                    kv_cache=kv_cache_l,
                    q_position_ids=position_ids,
                    kv_position_ids=position_ids,
                    update_cache=update_cache
                )

                L_out = L_flat + self.dropout(L_att)
                L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
                L_out = L_out.view(B, Nl, T, d).transpose(1, 2) # Restore [B, T, Nl, d]
            else:
                L_out = None

            return R_out, L_out

        # Use checkpointing for training
        if self.training and False:
            if not self.temporal:
                R_out, L_out = checkpoint(_forward_spatial, R, L, use_reentrant=False)
            else:
                R_out, L_out = checkpoint(_forward_temporal, R, L, use_reentrant=False)
        else:
            if not self.temporal:
                R_out, L_out = _forward_spatial(R, L)
            else:
                R_out, L_out = _forward_temporal(R, L, kv_cache_l, kv_cache_r, position_ids, update_cache)

        return R_out, L_out


class DreamerV4Decoder(nn.Module):
    """
    Block-causal tokenizer decoder:
      - Projects latent bottleneck Z (d_b) back to model width d.
      - Alternates spatial-only and temporal-only layers.
      - Routing (decoder): latents -> latents; patches -> patches ⊕ latents.
      - Outputs per-patch predictions and reconstructed frames.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        d_model: int,
        n_layers: int,
        num_heads_q: int,
        num_heads_kv_latent: int,  # G for latents (GQA if < Hq)
        bottleneck_dim: int,
        seq_len: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        n_latents: int = 256,
        in_channels: int = 3,
        temporal_every: int = 1,
    ):
        super().__init__()
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.H, self.W = H, W
        self.P = patch_size
        self.Np = (H // patch_size) * (W // patch_size)
        self.Nl = n_latents
        self.d = d_model
        self.d_b = bottleneck_dim
        self.C = in_channels

        # Project bottleneck up to model width
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        # Learned patch readout tokens per frame
        self.readout_tokens = nn.Parameter(torch.randn(self.Np, d_model) / math.sqrt(d_model))

        # Stack of decoder layers
        layers = []
        for i in range(n_layers):
            temporal = (i % 2 == 1) if temporal_every == 1 else (((i+1) % temporal_every) == 0)
            layers.append(
                BlockCausalDecoderLayer(
                    d_model=d_model,
                    num_heads_q=num_heads_q,
                    num_heads_kv_latent=num_heads_kv_latent,
                    seq_len=seq_len,
                    num_latents=self.Nl,
                    num_patches=self.Np,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    temporal=temporal,
                    is_last=(i == n_layers-1) or (i == n_layers-2 and (n_layers % temporal_every) == 0)
                )
            )
        self.layers = nn.ModuleList(layers)

        # Per-patch regression head: d -> P^2 * C
        self.patch_head = nn.Linear(d_model, (patch_size ** 2) * in_channels)

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Z: [B, T, Nl, d_b]  (latent bottleneck from encoder)
        Returns:
          R_dec: [B, T, Np, d]        (decoded readout tokens)
          x_hat: [B, T, C, H, W]      (reconstructed frames)
        """
        B, T, Nl, d_b = Z.shape
        assert Nl == self.Nl and d_b == self.d_b

        # Project latents up
        L = self.up_proj(Z)  # [B,T,Nl,d]

        # Initialize per-frame readout tokens
        R0 = self.readout_tokens[None, None, :, :].expand(B, T, self.Np, self.d).contiguous()

        # Apply decoder layers
        R_dec, L_dec = R0, L
        for layer in self.layers:
            R_dec, L_dec = layer(R_dec, L_dec)

        # Map readout tokens to patch pixels
        B_, T_, Np_, d_ = R_dec.shape
        patches = self.patch_head(R_dec).view(B_, T_, Np_, self.P, self.P, self.C)  # [B,T,Np,P,P,C]

        # Reassemble patches into images: (H/P, W/P) grid
        Hp, Wp = self.H // self.P, self.W // self.P
        patches = patches.view(B_, T_, Hp, Wp, self.P, self.P, self.C).permute(0,1,2,4,3,5,6).contiguous()
        x_hat = patches.view(B_, T_, self.H, self.W, self.C).permute(0,1,4,2,3).contiguous()  # [B,T,C,H,W]
        return R_dec, x_hat

    def init_cache(self, batch_size: int, device: torch.device, max_seq_len: int):
        """Initializes separate KV caches for Latent (L) and Patch (R) streams."""
        self.caches_l = []
        self.caches_r = []
        
        for layer in self.layers:
            if layer.temporal:
                # Cache L: Batch * Num_Latents sequences
                cache_l = KVCache(
                    max_seq_len=max_seq_len,
                    batch_size=batch_size * self.Nl,
                    num_heads=layer.attn_latent.Hq,
                    head_dim=layer.attn_latent.Dh,
                    device=device,
                    dtype=self.up_proj.weight.dtype
                ) if not layer.is_last else None
                
                # Cache R: Batch * Num_Patches sequences
                cache_r = KVCache(
                    max_seq_len=max_seq_len,
                    batch_size=batch_size * self.Np,
                    num_heads=layer.attn_patch.Hq,
                    head_dim=layer.attn_patch.Dh,
                    device=device,
                    dtype=self.up_proj.weight.dtype
                )
                self.caches_l.append(cache_l)
                self.caches_r.append(cache_r)
            else:
                self.caches_l.append(None)
                self.caches_r.append(None)

    def forward_step(self, Z: torch.Tensor, start_step_idx: int, update_cache: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive forward that handles both single-step generation (T=1)
        and multi-step prefilling (T>1).
        
        Z: [B, T, Nl, d_b]
        """
        B, T, Nl, d_b = Z.shape
        
        # 1. Project latents up to model width
        L = self.up_proj(Z) # [B, T, Nl, d]
        
        # 2. Get readout tokens for the sequence length T
        # Expand R0 to match the input time dimension T
        R0 = self.readout_tokens[None, None, :, :].expand(B, T, self.Np, self.d).contiguous()
        
        # 3. Generate Position IDs for RoPE
        # Create a range [start, start+1, ..., start+T-1]
        # Shape [1, T] -> Broadcasts to [Batch * Tokens, T] inside MHA
        pos_ids = torch.arange(start_step_idx, start_step_idx + T, device=Z.device, dtype=torch.long)
        pos_ids = pos_ids.unsqueeze(0) 

        R_dec, L_dec = R0, L
        
        # 4. Apply layers (Spatial layers handle T natively; Temporal layers use cache)
        for i, layer in enumerate(self.layers):
            if layer.temporal:
                R_dec, L_dec = layer(
                    R_dec, L_dec, 
                    kv_cache_l=self.caches_l[i], 
                    kv_cache_r=self.caches_r[i],
                    position_ids=pos_ids,
                    update_cache=update_cache
                )
            else:
                R_dec, L_dec = layer(R_dec, L_dec)
                
        # 5. Output projection
        patches = self.patch_head(R_dec).view(B, T, self.Np, self.P, self.P, self.C)
        Hp, Wp = self.H // self.P, self.W // self.P
        patches = patches.view(B, T, Hp, Wp, self.P, self.P, self.C).permute(0,1,2,4,3,5,6).contiguous()
        x_hat = patches.view(B, T, self.H, self.W, self.C).permute(0,1,4,2,3).contiguous()
        
        return R_dec, x_hat

class BlockCausalDynamicsLayer(nn.Module):
    """
    A single layer of the Dynamics Transformer.
    Alternates between Spatial Attention (mixing tokens within a frame) 
    and Temporal Attention (mixing history for a specific token position).
    """
    def __init__(self, d_model, num_heads, seq_len, num_tokens, dropout=0.0, mlp_ratio=4.0, temporal=False):
        super().__init__()
        self.d_model = d_model
        self.temporal = temporal
        
        self.norm1 = nn.RMSNorm(d_model, eps=1e-6)
        # Dynamics has a single stream of tokens, so we use Self-Attention (Hq=Hkv)
        self.attn = MHA_GQA(
            d_model=d_model,
            num_heads_q=num_heads,
            num_heads_kv=num_heads,
            head_dim=d_model // num_heads,
            dropout=dropout,
            max_seq_len=seq_len if temporal else num_tokens
        )
        
        self.norm2 = nn.RMSNorm(d_model, eps=1e-6)
        self.ffn = FeedForwardSwiGLU(d_model, int(d_model * mlp_ratio * 2 / 3), dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache: Optional[KVCache] = None, position_ids: Optional[torch.Tensor] = None, update_cache=True):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        
        skip = x
        x = self.norm1(x)
        
        if self.temporal:
            # Temporal Attention: Independent per token index 'n', causal over 't'
            # Reshape to (B*N, T, D) to treat each token position as a separate sequence
            x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
            x = self.attn(x, x, is_causal=True, kv_cache=kv_cache, 
                          q_position_ids=position_ids, kv_position_ids=position_ids,
                          update_cache=update_cache)
            x = x.view(B, N, T, D).permute(0, 2, 1, 3)
        else:
            # Spatial Attention: Independent per timestep 't', dense over 'n'
            # Reshape to (B*T, N, D) to treat each frame as a separate sequence
            x = x.view(B * T, N, D)
            x = self.attn(x, x, is_causal=False)
            x = x.view(B, T, N, D)
            
        x = self.dropout(x)
        x = x + skip
        
        skip = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + skip
            
        return x

class DreamerV4Dynamics(nn.Module):
    def __init__(
        self,
        action_dim: int,
        num_latents: int,      # Number of spatial tokens (z) per frame (e.g., 256)
        latent_dim: int,       # Dimension of the bottleneck (db) (e.g., 32)
        d_model: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        num_registers: int = 4,
        seq_len: int = 96,    # Context length
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        num_tau_levels: int = 128,  # For discrete noise level embedding
        temporal_every: int = 4       # Apply temporal attention every K layers
    ):
        super().__init__()
        num_step_levels = int(np.log2(num_tau_levels)) + 1
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.num_registers = num_registers
        
        # --- Embeddings ---
        # 1. Action Embedding (Continuous vector)
        self.act_proj = nn.Linear(action_dim, d_model)
        
        # 2. Noise/Step Embedding
        # Paper: "encode each with a discrete embedding lookup and concatenate their channels"
        # We embed sigma and step to d_model/2 and concat to get d_model.
        self.sigma_embed = nn.Embedding(num_tau_levels, d_model // 2)
        self.step_embed = nn.Embedding(num_step_levels, d_model // 2)
        
        # 3. Latent Projection (db -> d_model)
        self.z_proj = nn.Linear(latent_dim, d_model)
        
        # 4. Register Tokens (Learned)
        self.register_tokens = nn.Parameter(torch.randn(num_registers, d_model) * 0.02)
        
        # --- Transformer ---
        layers = []
        # Total tokens per timestep: 1 (Action) + 1 (Sigma/Step) + Registers + Latents
        self.num_tokens_per_step = 1 + 1 + num_registers + num_latents
        
        for i in range(num_layers):
            # Follows paper's logic: Temporal only every 4 layers (default), otherwise Spatial
            # Note: This simple logic puts temporal at indices 3, 7, 11...
            is_temporal = ((i + 1) % temporal_every == 0)
            
            layers.append(BlockCausalDynamicsLayer(
                d_model=d_model,
                num_heads=num_heads,
                seq_len=seq_len,
                num_tokens=self.num_tokens_per_step,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                temporal=is_temporal,
            ))
        self.layers = nn.ModuleList(layers)
        
        # --- Output Head ---
        # Predict clean representations (x-prediction)
        self.out_proj = nn.Linear(d_model, latent_dim)

    def forward(self,
                action: torch.Tensor,
                noisy_z: torch.Tensor,
                sigma_idx: torch.Tensor,
                step_idx: torch.Tensor):
        """
        Args:
            action: (B, T, action_dim) - Actions taken at each step
            noisy_z: (B, T, num_latents, latent_dim) - Corrupted representations
            sigma_idx: (B, T) - Indices for noise level
            step_idx: (B, T) - Indices for step size
        Returns:
            pred_z: (B, T, num_latents, latent_dim) - Predicted clean representations
        """
        B, T, _, _ = noisy_z.shape
        
        # 1. Prepare Embeddings
        # Action: (B, T, 1, D)
        emb_act = self.act_proj(action).unsqueeze(2)
        
        # Sigma/Step: (B, T, 1, D)
        emb_sigma = self.sigma_embed(sigma_idx)
        emb_step = self.step_embed(step_idx)
        emb_noise = torch.cat([emb_sigma, emb_step], dim=-1).unsqueeze(2)
        
        # Registers: (B, T, Nr, D)
        emb_reg = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        
        # Latents: (B, T, Nl, D)
        emb_z = self.z_proj(noisy_z)
        
        # 2. Interleave / Concatenate Inputs
        # Order: [Action, Noise, Registers, Latents]
        # Shape: (B, T, 1+1+Nr+Nl, D)
        x = torch.cat([emb_act, emb_noise, emb_reg, emb_z], dim=2)
        
        # 3. Apply Transformer Layers
        for layer in self.layers:
            x = layer(x)
            
        # 4. Extract and Project Outputs
        # We only need the predictions corresponding to the latent tokens (the last Nl tokens)
        out_z_tokens = x[:, :, -self.num_latents:, :]
        pred_z = self.out_proj(out_z_tokens)
        
        return pred_z

    def init_cache(self, batch_size: int, device: torch.device, max_seq_len: int = 96):
        """Initializes KV caches for all temporal layers."""
        self.caches = []
        for layer in self.layers:
            if layer.temporal:
                # Temporal cache needs to store (Batch * Num_Tokens) streams
                # Batch size for cache = B * num_tokens_per_step
                cache_bs = batch_size * self.num_tokens_per_step
                
                cache = KVCache(
                    max_seq_len=max_seq_len,
                    batch_size=cache_bs,
                    num_heads=self.layers[0].attn.Hq, # Assuming uniform heads
                    head_dim=self.layers[0].attn.Dh,
                    device=device,
                    dtype=self.act_proj.weight.dtype
                )
                self.caches.append(cache)
            else:
                self.caches.append(None) # Spatial layers don't cache time

    def forward_step(self, 
                     action: torch.Tensor, 
                     noisy_z: torch.Tensor, 
                     sigma_idx: torch.Tensor, 
                     step_idx: torch.Tensor,
                     start_step_idx: int,  # Absolute position for RoPE
                     update_cache: bool = True):
        """
        Autoregressive forward for a single timestep T=1.
        Must call init_cache() before starting loop.
        """
        # Inputs are [B, 1, ...]
        B, T, _, _ = noisy_z.shape
        
        # 1. Embeddings (Same as forward)
        emb_act = self.act_proj(action).unsqueeze(2) # [B, 1, 1, D]
        emb_sigma = self.sigma_embed(sigma_idx)
        emb_step = self.step_embed(step_idx)
        emb_noise = torch.cat([emb_sigma, emb_step], dim=-1).unsqueeze(2)
        emb_reg = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        emb_z = self.z_proj(noisy_z)
        
        # Concatenate: [B, 1, N_total, D]
        x = torch.cat([emb_act, emb_noise, emb_reg, emb_z], dim=2)
        
        # Position IDs for RoPE
        # We need [T] indices starting from start_step_idx
        # Shape [1, T] to broadcast over batch
        pos_ids = torch.arange(start_step_idx, start_step_idx + T, device=x.device, dtype=torch.long)
        pos_ids = pos_ids.unsqueeze(0) # [1, T]

        # 2. Apply Layers with Caching
        for i, layer in enumerate(self.layers):
            if layer.temporal:
                x = layer(x, kv_cache=self.caches[i], position_ids=pos_ids, update_cache=update_cache)
            else:
                x = layer(x)

        # 3. Output
        out_z_tokens = x[:, :, -self.num_latents:, :]
        pred_z = self.out_proj(out_z_tokens)
        
        return pred_z
