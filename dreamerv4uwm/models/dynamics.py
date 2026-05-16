import math
from dataclasses import dataclass
from typing import Optional, List
from .blocks import create_temporal_mask
import torch
import torch.nn as nn
from .blocks import EfficientTransformerBlock, LayerType
from omegaconf import DictConfig, OmegaConf

import torch


class SymlogTwoHotHead(nn.Module):
    """Symlog-space two-hot classification head over a fixed bucket grid.

    Predicts a scalar via classification: logits over `num_buckets` evenly spaced
    in symlog space across [min_val, max_val]. `get_targets` returns the
    (low_idx, low_weight, high_idx, high_weight) tuple for the two-hot CE target.
    """

    def __init__(self, input_dim: int, num_buckets: int = 255, min_val: float = -20.0, max_val: float = 20.0):
        """Build the head.

        Args:
            input_dim:   feature dim of the activations fed to `forward`.
            num_buckets: number of bins; logits have this width.
            min_val, max_val: symlog-space range covered by the bucket grid.
                Targets outside this range are clipped, not extrapolated.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.min_val = min_val
        self.max_val = max_val
        self.linear = nn.Linear(input_dim, num_buckets)
        buckets = torch.linspace(min_val, max_val, num_buckets)
        self.register_buffer("buckets", buckets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to per-bucket logits.

        Args:
            x: feature tensor of shape (..., input_dim). Leading dims are
               preserved; in this codebase typically (B, T, D).

        Returns:
            logits of shape (..., num_buckets).
        """
        return self.linear(x)

    @staticmethod
    def to_symlog(x: torch.Tensor) -> torch.Tensor:
        """Compress real values into symlog space: sign(x) * log(|x| + 1).

        Used to handle a wide reward range without dedicating most buckets to
        the tails. Linear near zero, log-scaled away from zero.
        """
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    @staticmethod
    def from_symlog(x: torch.Tensor) -> torch.Tensor:
        """Inverse of `to_symlog`: sign(x) * (exp(|x|) - 1)."""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    def get_targets(self, rewards: torch.Tensor):
        """Build two-hot targets from raw rewards.

        Maps each reward into symlog space, locates it on the bucket grid, and
        returns the (low_idx, low_weight, high_idx, high_weight) tuple that
        scatters unit mass linearly between adjacent buckets.

        Out-of-range rewards are clamped to the edge buckets — the head cannot
        represent values beyond [min_val, max_val] in symlog space.

        Edge case: when a target lands exactly on a bucket index, naive
        `high - low` weights collapse to 0/0. We force all mass onto `low` in
        that case (also covers the post-clamp case where low == high == edge).

        Args:
            rewards: real-space reward tensor of any shape S (commonly (B, T, L)
                in MTP usage, or (B, T) for a single-step head).

        Returns:
            tuple (low, low_weight, high, high_weight), each of shape S:
                low, high       — long tensors, bucket indices in [0, num_buckets-1].
                low_weight, high_weight — float tensors, weights summing to 1
                                           per element (two-hot mass).
        """
        y = self.to_symlog(rewards)
        width = (self.max_val - self.min_val) / (self.num_buckets - 1)
        indices = (y - self.min_val) / width
        indices = indices.clamp(0, self.num_buckets - 1)

        low = indices.floor().long()
        high = indices.ceil().long()
        low_weight = high.float() - indices
        high_weight = indices - low.float()

        mask = (low == high)
        low_weight[mask] = 1.0
        high_weight[mask] = 0.0

        return low, low_weight, high, high_weight


class RewardMTPHead(nn.Module):
    """Multi-token prediction head: L parallel reward predictions per timestep.

    A shared MLP trunk maps each task embedding to a hidden vector, then L
    independent `SymlogTwoHotHead` heads each emit logits for one future
    horizon offset (predicting r_{t+0}, r_{t+1}, ..., r_{t+L-1}).

    The L heads do not share output weights — each learns its own decoder over
    the bucket grid — but they share the trunk so the per-step cost stays
    bounded as L grows.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        mtp_length: int = 8,
        num_buckets: int = 255,
    ):
        """Build the MTP head.

        Args:
            input_dim:   feature dim of the per-timestep embedding fed to `forward`
                         (in this codebase, the agent-token slice of the denoiser
                         output, dim = `model_dim`).
            hidden_dim:  width of the shared trunk.
            mtp_length:  number of future-reward heads L.
            num_buckets: bucket-grid width passed to each `SymlogTwoHotHead`.
        """
        super().__init__()
        self.mtp_length = mtp_length
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([
            SymlogTwoHotHead(hidden_dim, num_buckets) for _ in range(mtp_length)
        ])

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """Predict L bucketized reward distributions per timestep.

        Args:
            h_t: per-timestep agent embedding of shape (B, T, input_dim).

        Returns:
            logits of shape (B, T, L, num_buckets), where dim 2 indexes the
            future horizon offset and dim 3 indexes bucket logits. Apply
            log-softmax over the last dim before computing two-hot CE.
        """
        x = self.hidden(h_t)
        outputs = [head(x) for head in self.heads]
        return torch.stack(outputs, dim=2)


def build_agent_isolation_mask(n_world: int) -> torch.Tensor:
    """Spatial-attention mask that isolates a trailing read-only agent token.

    The token sequence per frame is laid out as `[world_tokens..., agent]`,
    where `world_tokens` is the existing `[Z, Reg, IC, AC, A]` block of length
    `n_world`. The agent token sits at index `n_world` (the last position).

    Polarity (matches the float-mask convention used by AxialAttention: 0.0 =
    allowed, -inf = blocked):
        - Agent row (last): can attend to everything → all zeros.
        - World rows: cannot attend to the agent column → -inf at the last col.
        - All other cells: 0.0 — i.e. inside the world block we leave attention
          fully open, matching the `spatial_mask=None` path used when the
          reward head is disabled. So toggling `train_reward_model` adds
          *only* agent isolation, no other change to spatial attention.

    Args:
        n_world: number of world (non-agent) tokens per frame.

    Returns:
        float mask of shape (n_world + 1, n_world + 1).
    """
    n_total = n_world + 1
    mask = torch.zeros(n_total, n_total, dtype=torch.float32)
    mask[:n_world, n_world] = float('-inf')  # block agent column for world rows
    return mask


class DiscreteEmbedder(nn.Module):
    def __init__(self, n_states, n_dim):
        super().__init__()
        self.n_states = n_states

        # (n_states, n_dim) — each row = embedding for one discrete state
        self.embeddings = nn.Parameter(torch.zeros(n_states, n_dim))

        # good idea: initialize like nn.Embedding
        nn.init.normal_(self.embeddings, std=0.02)

    def forward(self, x):
        """
        x: LongTensor of shape (B,) or (B, T) containing indices in [0, n_states)
        returns: embeddings of shape (B, n_dim) or (B, T, n_dim)
        """
        x = x.long()
        return self.embeddings[x]  # fancy indexing works

@dataclass
class DreamerV4DenoiserCfg:
    num_action_tokens: int          # S_a
    num_latent_tokens: int
    num_register_tokens: int
    max_sequence_length: int
    context_length: int
    model_dim: int
    latent_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    dropout_prob: float = 0.0
    qk_norm: bool = True
    num_noise_levels: int = 32                # finest grid size for τ (must be power of 2)
    n_actions: int = 0  # number of action components
    dual_stream: bool = False
    is_causal: bool = False  # whether to use causal masking in the transformer (should be False for standard denoising, True for stepwise inference)
    layer_types: Optional[List[str]] = None  # list of layer types, e.g. ["spatial", "temporal", "spatial", "temporal"]; defaults to alternating spatial/temporal
    # Reward / MTP head. Off by default → bit-equivalent to runs without these
    # fields. When on, a read-only agent token is appended per frame and an MTP
    # head produces L=`mtp_length` future-reward predictions per timestep.
    train_reward_model: bool = False
    mtp_length: int = 8
    reward_hidden_dim: int = 512
    reward_num_buckets: int = 255

class DreamerV4Denoiser(nn.Module):
    """
    Dynamics model / denoiser with:
      - per-frame latent tokens
      - per-frame discrete τ index (0..num_noise_levels-1)
      - per-frame discrete step index (0..max_pow2), where 0 ↔ d_min, max_pow2 ↔ 1
      - action conditioning via ActionTokenizer
    """

    def __init__(self, cfg: DreamerV4DenoiserCfg, max_num_forward_steps=None):
        super().__init__()
        self.cfg = cfg
        # --- Discrete embeddings for diffusion τ and shortcut step index ---
        self.obs_diffusion_embedder = DiscreteEmbedder(cfg.num_noise_levels, cfg.model_dim)  # τ index: 0..num_noise_levels-1
        self.obs_shortcut_embedder = DiscreteEmbedder(int(math.log2(cfg.num_noise_levels)) + 1, cfg.model_dim)  # step index: 0..max_pow2
        self.act_diffusion_embedder = DiscreteEmbedder(cfg.num_noise_levels, cfg.model_dim)  # τ index: 0..num_noise_levels-1
        self.act_shortcut_embedder = DiscreteEmbedder(int(math.log2(cfg.num_noise_levels)) + 1, cfg.model_dim)  # step index: 0..max_pow2
        # Frame-identity embedder: 0 = context frame, 1 = horizon frame. Summed
        # into IC and AC after their projections. Zero-init keeps the network
        # bit-equivalent to the pre-existing arch when is_horizon is None or
        # all-zeros (i.e., WM mode and any caller that doesn't supply it).
        self.frame_id_embedder = DiscreteEmbedder(2, cfg.model_dim)
        nn.init.zeros_(self.frame_id_embedder.embeddings)
        
        # --- Register tokens: (1, 1, S_r, D) ---
        self.register_tokens = nn.Parameter(
            torch.zeros(1, 1, cfg.num_register_tokens, cfg.model_dim)
        )

        self.num_modality_tokens = cfg.num_action_tokens + \
                                   cfg.num_latent_tokens + \
                                   cfg.num_register_tokens + \
                                    2 # noise level + shortcut tokens (obs + act) that are combined into a single control token each
        # World-token count BEFORE the optional agent token. Used to size the
        # agent-isolation mask. Keep this value sourced from
        # num_modality_tokens so future modality-count changes flow through.
        self.num_world_tokens = self.num_modality_tokens
        if cfg.train_reward_model:
            self.num_modality_tokens += 1  # appended agent token
        if cfg.layer_types is not None:
            self.layer_types = [LayerType(t) for t in cfg.layer_types]
        else:
            self.layer_types = [
                LayerType.SPATIAL if i % 2 == 0 else LayerType.TEMPORAL
                for i in range(4)
            ]
        # --- Transformer layers ---
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(
                model_dim=cfg.model_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                dropout_prob=cfg.dropout_prob,
                qk_norm=cfg.qk_norm,
                modality_dim_max_seq_len=self.num_modality_tokens,
                temporal_dim_max_seq_len= (max_num_forward_steps if max_num_forward_steps is not None else cfg.max_sequence_length),   
                context_length=cfg.context_length,    
                is_causal=cfg.is_causal,   
                layer_types=self.layer_types     
            )
            for _ in range(cfg.n_layers)
        ])

        # --- Latent projections ---
        self.latent_projector = nn.Linear(cfg.latent_dim, cfg.model_dim, bias=False)
        self.obs_projector = nn.Linear(cfg.model_dim, cfg.latent_dim, bias=False)
        self.action_projector = nn.Linear(cfg.model_dim, cfg.n_actions)

        # --- Combine shortcut + diffusion embeddings into a single control token ---
        self.obs_diff_control_proj = nn.Linear(cfg.model_dim * 2, cfg.model_dim, bias=False)
        self.act_diff_control_proj = nn.Linear(cfg.model_dim * 2, cfg.model_dim, bias=False)
        self.action_input_proj = nn.Linear(cfg.n_actions, cfg.model_dim)
        # Initialize learnable tokens
        nn.init.normal_(self.register_tokens, std=0.02)

        # --- Optional reward head + read-only agent token ---
        if cfg.train_reward_model:
            # (1, 1, 1, D): broadcasts to (B, T, 1, D) at forward time. Init
            # scale matches the reference implementation.
            self.agent_token = nn.Parameter(
                torch.randn(1, 1, 1, cfg.model_dim) * 0.02
            )
            self.reward_head = RewardMTPHead(
                input_dim=cfg.model_dim,
                hidden_dim=cfg.reward_hidden_dim,
                mtp_length=cfg.mtp_length,
                num_buckets=cfg.reward_num_buckets,
            )
            agent_spatial_mask = build_agent_isolation_mask(self.num_world_tokens)
            self.register_buffer("agent_spatial_mask", agent_spatial_mask, persistent=False)
        else:
            self.register_parameter("agent_token", None)
            self.reward_head = None
            self.agent_spatial_mask = None

    def forward(
        self,
        noisy_act: torch.Tensor,  # (B, T, n_c)
        noisy_obs: torch.Tensor,          # (B, T, N_latent, D_latent)
        obs_sigma_idx: torch.Tensor,         # (B, T) long, τ index on finest grid
        obs_step_idx: torch.Tensor,        # (B, T) long, step index (0..max_pow2; 0 ↔ d_min)
        act_sigma_idx: torch.Tensor,  # (B, T) long, τ index for actions (if separate from obs)
        act_step_idx: torch.Tensor,   # (B, T) long, step index for actions (if separate from obs
        is_horizon: Optional[torch.Tensor] = None,  # (T,) long {0,1}; None ↔ all context (purely causal)


    ) -> torch.Tensor:
        B, T, N_lat, D_latent = noisy_obs.shape

        # --- Encode diffusion τ and shortcut d into single control token ---
        # diff_step_token: (B, T, 1, D_model)
        obs_diff_step_token = self.obs_diffusion_embedder(obs_sigma_idx).unsqueeze(-2)
        obs_shortcut_token = self.obs_shortcut_embedder(obs_step_idx).unsqueeze(-2)
        act_diff_step_token = self.act_diffusion_embedder(act_sigma_idx).unsqueeze(-2)
        act_shortcut_token = self.act_shortcut_embedder(act_step_idx).unsqueeze(-2)

        # concat along channels: (B, T, 1, 2*D_model) -> (B, T, 1, D_model)
        obs_diff_control_token = torch.cat([obs_shortcut_token, obs_diff_step_token], dim=-1)
        obs_diff_control_token = self.obs_diff_control_proj(obs_diff_control_token)  # (B, T, 1, D_model)
        act_diff_control_token = torch.cat([act_shortcut_token, act_diff_step_token], dim=-1)
        act_diff_control_token = self.act_diff_control_proj(act_diff_control_token)  # (B, T, 1, D_model)

        # --- Frame-identity embedding summed into IC and AC ---
        # is_horizon is per-frame and shared across the batch. Zero-init means
        # the no-arg path is bit-equivalent to the pre-existing model.
        if is_horizon is not None:
            frame_id_emb = self.frame_id_embedder(is_horizon)        # (T, D)
            frame_id_token = frame_id_emb.view(1, T, 1, self.cfg.model_dim)
            obs_diff_control_token = obs_diff_control_token + frame_id_token
            act_diff_control_token = act_diff_control_token + frame_id_token

        # --- Register tokens replicated per time step ---
        # reg_tokens: (1, 1, S_r, D) -> (B, T, S_r, D)
        reg_tokens = self.register_tokens.expand(B, T, -1, -1)

        # --- Project observations latents to model dim ---
        # obs_proj: (B, T, N_lat, D_model)
        obs_tokens = self.latent_projector(noisy_obs)
        act_tokens = self.action_input_proj(noisy_act).unsqueeze(-2)  # (B, T, 1, D_model)

        # --- Concatenate tokens:
        #[obs_tokens : register_tokens : obs_diff_control_token : act_diff_control_token : action_tokens (: agent_token)]
        # obs_tokens       : (B, T, N_lat,   D)
        # reg_tokens       : (B, T, S_r,     D)
        # obs_diff_control_token : (B, T, 1,       D)
        # act_diff_control_token : (B, T, 1,       D)
        # act_tokens       : (B, T, S_a,     D)
        # agent_token (opt): (B, T, 1,       D)
        x = torch.cat(
            [obs_tokens, reg_tokens, obs_diff_control_token, act_diff_control_token, act_tokens],
            dim=-2,  # token dimension
        )  # x: (B, T, N_lat + S_r + 1 + S_a, D_model)

        if self.cfg.train_reward_model:
            agent_part = self.agent_token.expand(B, T, -1, -1)  # (B, T, 1, D)
            x = torch.cat([x, agent_part], dim=-2)              # (..., +1)

        # --- Transformer dynamics ---
        # Spatial layers consume `spatial_mask`; temporal layers ignore it
        # (see EfficientTransformerLayer.forward — the mask kwarg is only
        # threaded into the spatial branch). So passing the agent-isolation
        # mask to every block gives the right "spatial-only" semantics for
        # free, and `None` preserves today's exact behavior.
        # Cast the mask to the activation dtype so SDPA kernels don't fault
        # under bf16 autocast.
        if self.cfg.train_reward_model:
            spatial_mask = self.agent_spatial_mask.to(dtype=x.dtype)
        else:
            spatial_mask = None
        for layer in self.layers:
            x = layer(x, spatial_mask=spatial_mask, is_horizon=is_horizon)

        # --- Project back to latent dim, return only latent slice ---
        if self.cfg.train_reward_model:
            world_x = x[:, :, :-1, :]                                # drop agent col
            agent_x = x[:, :, -1, :]                                  # (B, T, D)
            obs_output = self.obs_projector(world_x[:, :, :self.cfg.num_latent_tokens, :])
            act_output = self.action_projector(world_x[:, :, -self.cfg.num_action_tokens:, :])
            pred_rewards = self.reward_head(agent_x)                  # (B, T, L, K)
            return obs_output, act_output, pred_rewards
        else:
            obs_output = self.obs_projector(x[:, :, :self.cfg.num_latent_tokens, :])  # (B, T, N_lat, D_latent)
            act_output = self.action_projector(x[:, :, -self.cfg.num_action_tokens:, :])  # (B, T, S_a, n_actions)
            return obs_output, act_output, None

    def forward_step(
        self,
        noisy_act: torch.Tensor,           # (B, T, n_actions)
        noisy_obs: torch.Tensor,           # (B, T, N_latent, D_latent)
        obs_sigma_idx: torch.Tensor,       # (B, T) long
        obs_step_idx: torch.Tensor,        # (B, T) long
        act_sigma_idx: torch.Tensor,       # (B, T) long
        act_step_idx: torch.Tensor,        # (B, T) long
        start_step_idx: int,
        update_cache: bool = True,
        is_horizon: Optional[torch.Tensor] = None,
    ):
        """KV-cached counterpart to `forward()` for autoregressive sampling.

        Token layout, projections, and outputs mirror `forward()` exactly; only
        the temporal layers are run via their cached `forward_step` path.

        `is_horizon` is summed into IC/AC for distributional consistency with
        training, but the cached temporal path remains purely causal — the
        horizon-aware mask is uncached-only. At deployment the streamed
        context should pass is_horizon = zeros (or None, equivalent at init
        but trained-row-0 thereafter — pass zeros for fidelity).
        """
        B, T, N_lat, D_latent = noisy_obs.shape

        obs_diff_step_token = self.obs_diffusion_embedder(obs_sigma_idx).unsqueeze(-2)
        obs_shortcut_token  = self.obs_shortcut_embedder(obs_step_idx).unsqueeze(-2)
        act_diff_step_token = self.act_diffusion_embedder(act_sigma_idx).unsqueeze(-2)
        act_shortcut_token  = self.act_shortcut_embedder(act_step_idx).unsqueeze(-2)

        obs_diff_control_token = torch.cat([obs_shortcut_token, obs_diff_step_token], dim=-1)
        obs_diff_control_token = self.obs_diff_control_proj(obs_diff_control_token)
        act_diff_control_token = torch.cat([act_shortcut_token, act_diff_step_token], dim=-1)
        act_diff_control_token = self.act_diff_control_proj(act_diff_control_token)

        if is_horizon is not None:
            frame_id_emb = self.frame_id_embedder(is_horizon)        # (T, D)
            frame_id_token = frame_id_emb.view(1, T, 1, self.cfg.model_dim)
            obs_diff_control_token = obs_diff_control_token + frame_id_token
            act_diff_control_token = act_diff_control_token + frame_id_token

        reg_tokens = self.register_tokens.expand(B, T, -1, -1)
        obs_tokens = self.latent_projector(noisy_obs)
        act_tokens = self.action_input_proj(noisy_act).unsqueeze(-2)

        x = torch.cat(
            [obs_tokens, reg_tokens, obs_diff_control_token, act_diff_control_token, act_tokens],
            dim=-2,
        )

        if self.cfg.train_reward_model:
            agent_part = self.agent_token.expand(B, T, -1, -1)
            x = torch.cat([x, agent_part], dim=-2)
            spatial_mask = self.agent_spatial_mask.to(dtype=x.dtype)
        else:
            spatial_mask = None

        for layer in self.layers:
            x = layer.forward_step(
                x,
                start_step_idx=start_step_idx,
                spatial_mask=spatial_mask,
                update_cache=update_cache,
            )

        if self.cfg.train_reward_model:
            world_x = x[:, :, :-1, :]
            agent_x = x[:, :, -1, :]
            obs_output = self.obs_projector(world_x[:, :, :self.cfg.num_latent_tokens, :])
            act_output = self.action_projector(world_x[:, :, -self.cfg.num_action_tokens:, :])
            pred_rewards = self.reward_head(agent_x)
            return obs_output, act_output, pred_rewards
        else:
            obs_output = self.obs_projector(x[:, :, :self.cfg.num_latent_tokens, :])
            act_output = self.action_projector(x[:, :, -self.cfg.num_action_tokens:, :])
            return obs_output, act_output, None
    
    def init_cache(self, batch_size: int, device: torch.device, context_length: int, dtype: torch.dtype):
        """Initializes KV caches for all temporal layers."""
        for layer in self.layers:
            layer.init_cache(batch_size, device, context_length, dtype)

class DenoiserWrapper(nn.Module):
    def __init__(self, cfg: DictConfig, max_num_forward_steps=None):
        super().__init__()
        self.cfg = cfg
        denoiser_cfg = DreamerV4DenoiserCfg(**OmegaConf.to_object(cfg.denoiser))
        self.model = DreamerV4Denoiser(denoiser_cfg, max_num_forward_steps=max_num_forward_steps)

    def forward(
        self,
        noisy_act: torch.Tensor,          # (B, T, n_actions)
        noisy_obs: torch.Tensor,          # (B, T, N_latent, D_latent)
        obs_sigma_idx: torch.Tensor,      # (B, T) long
        obs_step_idx: torch.Tensor,       # (B, T) long
        act_sigma_idx: torch.Tensor,      # (B, T) long
        act_step_idx: torch.Tensor,       # (B, T) long
        is_horizon: Optional[torch.Tensor] = None,
    ):
        return self.model(
            noisy_act=noisy_act,
            noisy_obs=noisy_obs,
            obs_sigma_idx=obs_sigma_idx,
            obs_step_idx=obs_step_idx,
            act_sigma_idx=act_sigma_idx,
            act_step_idx=act_step_idx,
            is_horizon=is_horizon,
        )

    def forward_step(
        self,
        noisy_act: torch.Tensor,          # (B, T, n_actions) or (B, 1, n_actions) depending on your step API
        noisy_obs: torch.Tensor,          # (B, T, N_latent, D_latent) or (B, 1, ...)
        obs_sigma_idx: torch.Tensor,      # (B, T) or (B, 1)
        obs_step_idx: torch.Tensor,       # (B, T) or (B, 1)
        act_sigma_idx: torch.Tensor,      # (B, T) or (B, 1)
        act_step_idx: torch.Tensor,       # (B, T) or (B, 1)
        start_step_idx: int,
        update_cache: bool = True,
        is_horizon: Optional[torch.Tensor] = None,
    ):
        return self.model.forward_step(
            noisy_act=noisy_act,
            noisy_obs=noisy_obs,
            obs_sigma_idx=obs_sigma_idx,
            obs_step_idx=obs_step_idx,
            act_sigma_idx=act_sigma_idx,
            act_step_idx=act_step_idx,
            start_step_idx=start_step_idx,
            update_cache=update_cache,
            is_horizon=is_horizon,
        )

    def init_cache(
        self,
        batch_size: int,
        device: torch.device,
        context_length: int,
        dtype: torch.dtype,
    ):
        self.model.init_cache(batch_size, device, context_length, dtype)
