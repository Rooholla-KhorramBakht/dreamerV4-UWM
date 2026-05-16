import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models.dynamics import DreamerV4Denoiser
import torch.distributed as dist
import random


# Symlog two-hot bucket range. Must match SymlogTwoHotHead's defaults in
# dreamerv4uwm/models/dynamics.py — the loss bucketizes targets onto the same
# grid the model's heads project onto. If those defaults are ever made
# cfg-tunable, plumb the same values here.
_REWARD_SYMLOG_MIN = -20.0
_REWARD_SYMLOG_MAX = 20.0


def _two_hot_targets(rewards: torch.Tensor, num_buckets: int):
    """Compute two-hot bucket targets in symlog space.

    Returns (low, low_w, high, high_w), each the same shape as `rewards`. low
    and high are long tensors of bucket indices; low_w/high_w are floats
    summing to 1 per element. Clamps OOR rewards to the edge buckets and
    handles the bucket-edge `low == high` case (mass goes onto `low`).
    """
    y = torch.sign(rewards) * torch.log(torch.abs(rewards) + 1.0)  # symlog
    width = (_REWARD_SYMLOG_MAX - _REWARD_SYMLOG_MIN) / (num_buckets - 1)
    indices = (y - _REWARD_SYMLOG_MIN) / width
    indices = indices.clamp(0, num_buckets - 1)
    low = indices.floor().long()
    high = indices.ceil().long()
    low_w = high.float() - indices
    high_w = indices - low.float()
    edge = (low == high)
    low_w = torch.where(edge, torch.ones_like(low_w), low_w)
    high_w = torch.where(edge, torch.zeros_like(high_w), high_w)
    return low, low_w, high, high_w


def compute_reward_mtp_loss(
    pred_rewards: torch.Tensor,    # (B, T, L, K) logits
    rewards: torch.Tensor,          # (B, T) raw scalar rewards
) -> torch.Tensor:
    """Multi-token-prediction two-hot CE on bucketized symlog rewards.

    Builds targets target_{t, l} = rewards[t + l] for l in [0, L), with
    out-of-window positions masked out.

    Casts logits to fp32 before log-softmax — bf16 underflows on the
    one-bucket-wide weights. Returns a scalar mean over valid (t, l) pairs.

    Returns 0.0 when T == 0.
    """
    B, T, L, K = pred_rewards.shape
    if T == 0:
        return pred_rewards.new_zeros(())

    # Roll rewards into per-(t, l) targets via right-padded unfold.
    padded = F.pad(rewards, (0, L - 1))                          # (B, T + L - 1)
    targets = padded.unfold(dimension=1, size=L, step=1)[:, :T]  # (B, T, L)

    # Validity: drop (t, l) pairs that overshoot the window.
    t_idx = torch.arange(T, device=rewards.device).view(1, T, 1)
    l_idx = torch.arange(L, device=rewards.device).view(1, 1, L)
    valid = ((t_idx + l_idx) < T).expand(B, T, L).float()

    low, low_w, high, high_w = _two_hot_targets(targets, num_buckets=K)
    logp = F.log_softmax(pred_rewards.float(), dim=-1)            # fp32 cast
    nll = -(
        low_w * logp.gather(-1, low.unsqueeze(-1)).squeeze(-1)
        + high_w * logp.gather(-1, high.unsqueeze(-1)).squeeze(-1)
    )
    return (nll * valid).sum() / valid.sum().clamp_min(1.0)


def loss_weight(tau: torch.Tensor, scheme: str = 'ramp') -> torch.Tensor:
    """
    Per-element loss weighting over the noise schedule.

    Convention: τ=1 is clean, τ=0 is pure noise (see forward-process classes).

    Schemes:
      - 'ramp':    w(τ) = 0.9 τ + 0.1   (flow-matching default; down-weights
                   the noisy end 10×, favoring easy clean-ish samples).
      - 'uniform': w(τ) = 1             (equal weight everywhere — use when
                   you need strong gradient in the very-noisy regime, e.g.
                   training for masked-context / unconditional generation).

    tau: (B, T) or broadcastable shape
    returns: same shape as tau
    """
    if scheme == 'ramp':
        return 0.9 * tau + 0.1
    elif scheme == 'uniform':
        return torch.ones_like(tau)
    else:
        raise ValueError(f"Unknown loss weighting scheme: {scheme!r}")


def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """Back-compat alias; prefer loss_weight(tau, scheme='ramp')."""
    return loss_weight(tau, scheme='ramp')

class RMSLossScaler:
    """
    Tracks running RMS for named losses and returns normalized losses.
    """
    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = decay
        self.eps = eps
        self.ema_sq = {}  # name -> scalar tensor

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        # value is a scalar loss tensor (per-batch, per-rank)
        with torch.no_grad():
            sq = value.detach().pow(2)
            mean_sq = sq.mean()

            # Optionally average across ranks for a global RMS
            if dist.is_initialized():
                dist.all_reduce(mean_sq, op=dist.ReduceOp.AVG)

            if name not in self.ema_sq:
                self.ema_sq[name] = mean_sq
            else:
                self.ema_sq[name] = (
                    self.decay * self.ema_sq[name] + (1.0 - self.decay) * mean_sq
                )
            rms = (self.ema_sq[name] + self.eps).sqrt()
        # Normalize current loss by running RMS; gradient flows only through value
        return value / rms

class UWMForwardProcess(nn.Module):
    """
    Dyadic shortcut schedule for BOTH:
      - obs latents z: (B, T, N_lat, D_lat)
      - actions a:     (B, T, 1, n_actions)   (we enforce num_action_tokens=1)

    Forward mixing:
      x_tau = (1 - tau) * x0 + tau * x_clean
    """

    def __init__(self,
                 max_diff_steps=32,
                 action_noise_std: float = 1.,
                 mode_weights: Optional[dict] = None,
                 forcing_context_noise_bias: float = 0.0,
                 forcing_context_noise_alpha: float = 0.5,
                 forcing_context_noise_beta: float = 2.0,
                 forcing_mask_actions: bool = False,
                 device='cpu'):
        super().__init__()
        self.max_diff_steps = max_diff_steps
        self.device = device
        self.action_noise_std = action_noise_std
        self.forcing_mask_actions = forcing_mask_actions
        self.modes = ['policy', 'video', 'wm', 'id', 'forcing', 'action_only']
        if mode_weights is not None:
            weights = [float(mode_weights.get(m, 0.0)) for m in self.modes]
        else:
            weights = [1.0] * len(self.modes)
        total = sum(weights)
        assert total > 0, "At least one mode must have a positive weight"
        self.mode_probs = [w / total for w in weights]
        # Biased context-τ sampling in `forcing` mode. With prob `bias` the
        # context (chunk-0) τ for both streams is redrawn from Beta(α, β); the
        # horizon chunk is unaffected. α<β tilts mass toward τ≈0 (pure noise),
        # giving the model explicit training on very-corrupted-context rollouts.
        assert 0.0 <= forcing_context_noise_bias <= 1.0
        self.forcing_context_noise_bias = float(forcing_context_noise_bias)
        self.forcing_context_noise_alpha = float(forcing_context_noise_alpha)
        self.forcing_context_noise_beta = float(forcing_context_noise_beta)

    def _maybe_bias_context_tau(self, ctx_tau: torch.Tensor) -> torch.Tensor:
        """Per-element Bernoulli mixture: with prob `bias`, replace τ with a
        Beta(α, β) draw quantized to the max_diff_steps grid. Used only for
        the context chunk of `forcing` mode."""
        B = ctx_tau.shape[0]
        use_low = torch.rand(B, device=self.device) < self.forcing_context_noise_bias
        beta_dist = torch.distributions.Beta(
            torch.tensor(self.forcing_context_noise_alpha, device=self.device),
            torch.tensor(self.forcing_context_noise_beta, device=self.device),
        )
        beta_samples = beta_dist.sample((B,))
        biased_idx = (beta_samples * self.max_diff_steps).long().clamp(max=self.max_diff_steps - 1)
        biased_tau = biased_idx.float() / self.max_diff_steps
        return torch.where(use_low, biased_tau, ctx_tau)

    def sample_step_noise(self, batch_size, seq_len, force_mode: Optional[str] = None):
        B, T = batch_size, seq_len
        # Diffusion forcing noise level
        state_tau_d = torch.randint(0, self.max_diff_steps, (B,T)).to(self.device)
        state_tau = state_tau_d/self.max_diff_steps

        action_tau_d = torch.randint(0, self.max_diff_steps, (B,T)).to(self.device)
        action_tau = action_tau_d/self.max_diff_steps

        # torch.randint(1, T-1, ...) requires T >= 3. T=1 is the image-mode
        # path (force_mode='video', context_length unused by the video
        # schedule); T=2 is not used today but guarded for safety.
        if T <= 2:
            context_length = max(0, T - 1)
        else:
            context_length = torch.randint(1, T-1, (1,)).item()
        if force_mode is not None:
            assert force_mode in self.modes, (
                f"force_mode {force_mode!r} not in modes {self.modes}"
            )
            mode = force_mode
        else:
            mode = random.choices(self.modes, weights=self.mode_probs, k=1)[0]
        if mode == 'policy':
            # context: clean state, clean action ; chunk: noisy state, noisy action
            state_tau[:, :context_length] =  0.9999                               
            action_tau[:, :context_length] =  0.9999                                
            state_tau[:, context_length:] = state_tau[:, context_length].unsqueeze(-1)    
            action_tau[:, context_length:] = action_tau[:, context_length].unsqueeze(-1) 
        elif mode =='video':
            # Unconditioned video generation: identical state τ across context
            # AND horizon (single τ per batch element), action fully masked
            # (τ=0, pure noise) everywhere. Loss is computed on states across
            # the FULL sequence — see compute_uwm_loss.
            state_tau[:] = state_tau[:, 0].unsqueeze(-1)
            action_tau[:] = 0.
        elif mode=='wm':
            # context: noisy state (random τ broadcast across ctx), clean action;
            # horizon: noisy state (independent random τ broadcast across hor), clean action.
            # Context τ and horizon τ are sampled independently per batch element.
            state_tau[:, :context_length] = state_tau[:, 0].unsqueeze(-1)
            action_tau[:, :context_length] = 0.9999
            state_tau[:, context_length:] = state_tau[:, context_length].unsqueeze(-1)
            action_tau[:, context_length:] = 0.9999
        elif mode=='id':
            # context: clean state, noisy action ; chunk: clean state, noisy action
            state_tau[:, :context_length] =  0.9999                               
            action_tau[:, :context_length] = action_tau[:, context_length].unsqueeze(-1)
            state_tau[:, context_length:] =  0.9999   
            action_tau[:, context_length:] = action_tau[:, context_length].unsqueeze(-1)
        elif mode=='forcing':
            # Progressive Temporal Denoising: linearly decreasing τ across T.
            # Frame 0 = cleanest (highest τ), frame T-1 = noisiest (lowest τ).
            # Causal attention grounds later noisy frames on earlier clean ones,
            # resolving the cold-start problem without breaking causality.
            # Both streams share the same progressive schedule.
            pair = torch.sort(torch.randint(
                0, self.max_diff_steps, (B, 2), device=self.device,
            ), dim=1)[0]
            high_idx = pair[:, 1:2]  # (B, 1) — cleaner end  → frame 0
            low_idx  = pair[:, 0:1]  # (B, 1) — noisier end  → frame T-1
            slope = torch.linspace(0, 1, steps=T, device=self.device).unsqueeze(0)  # (1, T)
            progressive_idx = (high_idx + slope * (low_idx - high_idx)).long()
            progressive_idx = progressive_idx.clamp(0, self.max_diff_steps - 1)
            progressive_tau = progressive_idx.float() / self.max_diff_steps
            state_tau  = progressive_tau
            action_tau = progressive_tau if not self.forcing_mask_actions else torch.zeros_like(progressive_tau)
            context_length = 1
        elif mode=='action_only':
            # Conditional action policy baseline: clean context (state & action),
            # horizon state fully masked (τ=0, pure noise), horizon action sampled.
            # Loss is action-only on the horizon — the model must produce actions
            # from the clean context alone with no state info on the horizon.
            state_tau[:, :context_length] = 0.9999
            action_tau[:, :context_length] = 0.9999
            state_tau[:, context_length:] = 0.0
            action_tau[:, context_length:] = action_tau[:, context_length].unsqueeze(-1)
        else:
            raise NotImplementedError

        state_tau_idx = (state_tau*self.max_diff_steps).to(torch.long)
        action_tau_idx = (action_tau*self.max_diff_steps).to(torch.long)
        obs_diff = dict(tau=state_tau, tau_idx = state_tau_idx.to(self.device))
        act_diff = dict(tau=action_tau, tau_idx = action_tau_idx.to(self.device))
        # Frame-identity flag for the horizon-aware temporal mask.
        # WM stays fully causal (ctx + hor both treated as context).
        # video is deprecated and kept fully causal for parity with the legacy path.
        # All other modes split at `context_length`: hor = bidirectional within
        # the horizon block, ctx = causal-only-attended-from. forcing's
        # context_length is forced to 1 inside the mode's branch above, so its
        # is_horizon labels frame 0 as ctx and 1..T-1 as hor.
        is_horizon = torch.zeros(T, dtype=torch.long, device=self.device)
        if mode in ('policy', 'forcing', 'id', 'action_only'):
            is_horizon[context_length:] = 1
        return obs_diff, act_diff, context_length, mode, is_horizon

    def forward(
        self,
        z_clean: torch.Tensor,     # (B, T, N_lat, D_lat)
        a_clean: torch.Tensor,     # (B, T, N_act_tokens, n_actions)   (raw actions from dataset)
        force_mode: Optional[str] = None,
    ):
        B, T, N_lat, D_lat = z_clean.shape
        device = z_clean.device
        obs_diff, act_diff, context_length, mode, is_horizon = self.sample_step_noise(
            B, T, force_mode=force_mode,
        )

        # observation forward diffusion
        z0 = torch.randn_like(z_clean)
        obs_tau = obs_diff["tau"].unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
        z_tau = (1. - obs_tau) * z0 + obs_tau * z_clean

        # action forward diffusion
        a0 = self.action_noise_std * torch.randn_like(a_clean)
        act_tau = act_diff["tau"].unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
        a_tau = (1. - act_tau) * a0 + act_tau * a_clean

        return {
            # obs
            "x": z_clean,
            "x0": z0,
            "x_tau": z_tau,
            "obs_tau": obs_diff["tau"],
            "obs_tau_idx": obs_diff["tau_idx"],
            # act
            "a": a_clean,
            "a0": a0,
            "a_tau": a_tau,
            "act_tau": act_diff["tau"],
            "act_tau_idx": act_diff["tau_idx"],
            "context_length": context_length,
            "mode": mode,
            "is_horizon": is_horizon,
            "forcing_mask_actions": self.forcing_mask_actions if mode == 'forcing' else False,
        }
    
def compute_uwm_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    device='cpu',
    loss_weighting: str = 'ramp',
    rewards: Optional[torch.Tensor] = None,
):
    """Returns a dict with keys:
        - obs_flow_loss : scalar
        - act_flow_loss : scalar
        - reward_loss   : scalar or None (None when train_reward_model=False)

    `rewards` is a (B, T) tensor of raw scalar rewards. Required when
    `denoiser.cfg.train_reward_model` is True; ignored otherwise. Always
    matches the time axis of `info['x']` — caller's responsibility to slice
    rewards to the same window as the latents/actions when running on the
    image branch (T=1) or when cropping.
    """

    # --- obs ---
    x = info["x"]
    B, T, N_lat, D_lat = x.shape
    x_tau = info["x_tau"]
    obs_tau_idx = info["obs_tau_idx"]

    # --- act (S_a = 1) ---
    a = info["a"]         # (B,T,1,n_actions)
    a_tau = info["a_tau"] # (B,T,1,n_actions)
    act_tau_idx = info["act_tau_idx"]  # (B, T)

    step_idx = torch.zeros((B, T), dtype=torch.long, device=device)

    z_hat, a_hat, pred_rewards = denoiser(
        noisy_act=a_tau.squeeze(-2),  # (B,T,A) — denoiser expects (B,T,n_actions)
        noisy_obs=x_tau,
        obs_sigma_idx=obs_tau_idx,
        obs_step_idx=step_idx,
        act_sigma_idx=act_tau_idx,
        act_step_idx=step_idx,
        is_horizon=info.get("is_horizon"),
    )  # a_hat: (B,T,1,A); pred_rewards: (B,T,L,K) or None

    # X-prediction targets: directly regress clean signal
    obs_x_target = x                    # (B, T, N_lat, D_lat)
    act_x_target = a                    # (B, T, 1, n_actions)

    obs_flow_sq = (z_hat - obs_x_target).pow(2).mean(dim=(-1, -2))  # (B, T)
    act_flow_sq = (a_hat - act_x_target).pow(2).mean(dim=(-1, -2))  # (B, T)
    # Note: no .squeeze() here — info['obs_tau'] is already (B, T). At T=1 a
    # naive squeeze collapses (B, 1) → (B,) and then (B, 1) * (B,) broadcasts
    # to (B, B). Harmless at T>1 today but wrong for image-mode reshapes.
    w_obs      = loss_weight(info['obs_tau'], scheme=loss_weighting)  # (B, T)
    w_act      = loss_weight(info['act_tau'], scheme=loss_weighting)  # (B, T)

    mode = info['mode']
    context_length = info['context_length']
    if mode=='policy':
        obs_flow_loss = (obs_flow_sq*w_obs)[:, context_length:].mean()
        act_flow_loss = (act_flow_sq*w_act)[:, context_length:].mean()
    elif mode=='video':
        # Unconditioned video generation: state loss across the FULL sequence,
        # action stream zeroed out.
        obs_flow_loss = (obs_flow_sq*w_obs).mean()
        act_flow_loss = (act_flow_sq*w_act).mean()*0.

    elif mode=='wm':
        obs_flow_loss = (obs_flow_sq*w_obs).mean()
        act_flow_loss = (act_flow_sq*w_act).mean()*0.

    elif mode=='id':
        obs_flow_loss = (obs_flow_sq*w_obs).mean()*0.
        act_flow_loss = (act_flow_sq*w_act).mean()

    elif mode=='forcing':
        obs_flow_loss = (obs_flow_sq*w_obs).mean()
        if info.get('forcing_mask_actions', False):
            act_flow_loss = (act_flow_sq*w_act).mean()*0.
        else:
            act_flow_loss = (act_flow_sq*w_act).mean()
    elif mode=='action_only':
        obs_flow_loss = (obs_flow_sq*w_obs).mean()*0.
        act_flow_loss = (act_flow_sq*w_act)[:, context_length:].mean()
    else:
        raise NotImplementedError

    # --- reward MTP loss (only when the head exists) ---
    reward_loss = None
    if pred_rewards is not None:
        if rewards is None:
            raise RuntimeError(
                "denoiser was built with train_reward_model=True but "
                "compute_uwm_loss was called without `rewards`."
            )
        reward_loss = compute_reward_mtp_loss(pred_rewards, rewards)

    return {
        "obs_flow_loss": obs_flow_loss,
        "act_flow_loss": act_flow_loss,
        "reward_loss": reward_loss,
    }



class ShortcutUWMForwardProcess(nn.Module):
    """
    Dyadic shortcut schedule for BOTH:
      - obs latents z: (B, T, N_lat, D_lat)
      - actions a:     (B, T, 1, n_actions)   (we enforce num_action_tokens=1)

    Forward mixing:
      x_tau = (1 - tau) * x0 + tau * x_clean
    """

    def __init__(self,
                 max_diff_steps=32,
                 action_noise_std: float = 1.,
                 mode_weights: Optional[dict] = None,
                 flow_bias: float = 0.25,
                 forcing_context_noise_bias: float = 0.0,
                 forcing_context_noise_alpha: float = 0.5,
                 forcing_context_noise_beta: float = 2.0,
                 device='cpu'):
        super().__init__()
        assert (max_diff_steps & (max_diff_steps - 1)) == 0, "max_diff_steps must be a power of 2"
        assert 0.0 <= flow_bias <= 1.0, "flow_bias must lie in [0, 1]"
        self.max_diff_steps = max_diff_steps
        self.num_noise_levels = max_diff_steps
        self.max_pow2 = int(math.log2(max_diff_steps))
        self.d_min = 1.0 / float(max_diff_steps)
        self.device = device
        self.action_noise_std = action_noise_std
        self.flow_bias = float(flow_bias)
        self.modes = ['policy', 'video', 'wm', 'id', 'forcing']
        if mode_weights is not None:
            weights = [float(mode_weights.get(m, 0.0)) for m in self.modes]
        else:
            weights = [1.0] * len(self.modes)
        total = sum(weights)
        assert total > 0, "At least one mode must have a positive weight"
        self.mode_probs = [w / total for w in weights]
        # See UWMForwardProcess for the rationale — identical mechanism on the
        # dyadic grid. Context (chunk-0) τ-index is redrawn from Beta(α, β)
        # with probability `bias`, then quantized to the current step's grid.
        assert 0.0 <= forcing_context_noise_bias <= 1.0
        self.forcing_context_noise_bias = float(forcing_context_noise_bias)
        self.forcing_context_noise_alpha = float(forcing_context_noise_alpha)
        self.forcing_context_noise_beta = float(forcing_context_noise_beta)

    def _sample_biased_dyadic_chunk(self, step_index, num_tau_levels):
        """Draw (tau_idx, tau_half_idx) for a single chunk with τ ~ Beta(α, β),
        quantized to the current dyadic grid. Returns tensors of shape (B, 1)
        matching the layout used in `sample_step_noise`."""
        B = step_index.shape[0]
        beta_dist = torch.distributions.Beta(
            torch.tensor(self.forcing_context_noise_alpha, device=self.device),
            torch.tensor(self.forcing_context_noise_beta, device=self.device),
        )
        beta_samples = beta_dist.sample((B, 1))
        m = torch.floor(beta_samples * 0.9999 * num_tau_levels).long()
        tau_idx = m * (2 ** step_index)
        delta = (2 ** step_index) // 2
        tau_half_idx = torch.clamp(tau_idx + delta, min=0, max=self.num_noise_levels - 1)
        return tau_idx, tau_half_idx

    def _sample_dyadic_tau(self, B, step_index_raw, num_tau_levels):
        """
        Sample tau indices on the dyadic grid for two chunks (context & horizon).

        Returns:
            tau_index:           (B, 2) long, index on finest grid
            tau_plus_half_index: (B, 2) long, tau + d/2 index on finest grid
        """
        step_index = self.max_pow2 - step_index_raw          # (B, 1)
        m = torch.floor(
            torch.rand(B, 2, device=self.device) * 0.9999 * num_tau_levels
        ).long()                                              # (B, 2)
        tau_index = m * (2 ** step_index)                     # (B, 2)
        delta = (2 ** step_index) // 2                        # half-step stride
        tau_plus_half_index = torch.clamp(
            tau_index + delta, min=0, max=self.num_noise_levels - 1
        )
        return tau_index, tau_plus_half_index

    @staticmethod
    def _fill_schedule(target, target_half, ctx_len, ctx_val, hor_val, ctx_half, hor_half):
        """Assign context / horizon slices for one (B, T) schedule tensor and its half-step counterpart."""
        target[:, :ctx_len] = ctx_val
        target[:, ctx_len:] = hor_val
        target_half[:, :ctx_len] = ctx_half
        target_half[:, ctx_len:] = hor_half

    def sample_step_noise(self, batch_size, seq_len):

        B, T = batch_size, seq_len
        CLEAN = self.num_noise_levels - 1   # tau index for "fully clean"
        NOISY = 0                           # tau index for "fully noisy"

        # --- Shared shortcut step size (same for obs & act) ---
        # step_index_raw ∈ {0, ..., max_pow2}; d_t = 1 / 2^step_index_raw.
        # With probability `flow_bias`, force the flow branch (step_index_raw = max_pow2,
        # d = d_min). Remaining mass is spread uniformly over the bootstrap branches
        # {0, ..., max_pow2 - 1}. This keeps the flow objective well-trained so the
        # bootstrap target ladder has a solid base rung.
        if self.max_pow2 == 0:
            step_index_raw = torch.zeros((B, 1), device=self.device, dtype=torch.long)
        else:
            is_flow = torch.rand(B, 1, device=self.device) < self.flow_bias
            bootstrap_raw = torch.randint(
                0, self.max_pow2, (B, 1),
                device=self.device, dtype=torch.long,
            )
            flow_raw = torch.full(
                (B, 1), self.max_pow2,
                device=self.device, dtype=torch.long,
            )
            step_index_raw = torch.where(is_flow, flow_raw, bootstrap_raw)
        step = 1.0 / (2.0 ** step_index_raw.float())         # (B, 1) float, d_t
        # Convention: step_index 0 ↔ d_min (smallest), max_pow2 ↔ 1 (largest)
        step_index = self.max_pow2 - step_index_raw           # (B, 1)
        half_step_index = torch.clamp(step_index - 1, min=0)  # (B, 1)
        num_tau_levels = (2 ** step_index_raw).float()         # (B, 1)

        # --- Sample independent obs / act noise levels for 2 chunks each ---
        obs_tau_idx, obs_tau_half_idx = self._sample_dyadic_tau(B, step_index_raw, num_tau_levels)
        act_tau_idx, act_tau_half_idx = self._sample_dyadic_tau(B, step_index_raw, num_tau_levels)

        # --- Build (B, T) schedule tensors per mode ---
        context_length = torch.randint(1, T - 1, (1,)).item()
        mode = random.choices(self.modes, weights=self.mode_probs, k=1)[0]

        state_idx = torch.empty((B, T), device=self.device, dtype=torch.long)
        action_idx = torch.empty((B, T), device=self.device, dtype=torch.long)
        state_half_idx = torch.empty((B, T), device=self.device, dtype=torch.long)
        action_half_idx = torch.empty((B, T), device=self.device, dtype=torch.long)

        # x_tau = (1 - tau) * x0 + tau * x_clean  →  tau ≈ 1 is clean, tau = 0 is pure noise
        o0 = obs_tau_idx[:, 0].unsqueeze(-1)       # sampled obs tau (chunk 0)
        o1 = obs_tau_idx[:, 1].unsqueeze(-1)       # sampled obs tau (chunk 1)
        oh0 = obs_tau_half_idx[:, 0].unsqueeze(-1)
        oh1 = obs_tau_half_idx[:, 1].unsqueeze(-1)
        a0 = act_tau_idx[:, 0].unsqueeze(-1)
        a1 = act_tau_idx[:, 1].unsqueeze(-1)
        ah0 = act_tau_half_idx[:, 0].unsqueeze(-1)
        ah1 = act_tau_half_idx[:, 1].unsqueeze(-1)

        if mode == 'policy':
            # context: clean obs, clean act | horizon: noisy obs, noisy act
            self._fill_schedule(state_idx, state_half_idx, context_length,
                                CLEAN, o0, CLEAN, oh0)
            self._fill_schedule(action_idx, action_half_idx, context_length,
                                CLEAN, a0, CLEAN, ah0)
        elif mode == 'video':
            # Unconditioned video generation: identical state τ across context
            # AND horizon (single chunk-0 dyadic draw broadcast everywhere);
            # action fully masked (τ_idx=NOISY) throughout. Loss is computed
            # on states across the FULL sequence — see compute_bootstrap_uwm_loss.
            self._fill_schedule(state_idx, state_half_idx, context_length,
                                o0, o0, oh0, oh0)
            self._fill_schedule(action_idx, action_half_idx, context_length,
                                NOISY, NOISY, NOISY, NOISY)
        elif mode == 'wm':
            # context: clean obs, clean act | horizon: noisy obs, clean act
            self._fill_schedule(state_idx, state_half_idx, context_length,
                                CLEAN, o0, CLEAN, oh0)
            self._fill_schedule(action_idx, action_half_idx, context_length,
                                CLEAN, CLEAN, CLEAN, CLEAN)
        elif mode == 'id':
            # clean obs everywhere, noisy act everywhere
            self._fill_schedule(state_idx, state_half_idx, context_length,
                                CLEAN, CLEAN, CLEAN, CLEAN)
            self._fill_schedule(action_idx, action_half_idx, context_length,
                                a0, a0, ah0, ah0)
        elif mode == 'forcing':
            # Progressive Temporal Denoising on the dyadic grid.
            # Frame 0 = cleanest (highest idx), frame T-1 = noisiest (lowest idx).
            # Both streams share the same per-frame progressive schedule; the
            # shortcut step size d (and half-step delta) is still per-batch.
            delta = (2 ** step_index) // 2  # (B, 1)
            pair = torch.sort(torch.randint(
                0, self.num_noise_levels, (B, 2), device=self.device,
            ), dim=1)[0]
            high_idx = pair[:, 1:2]  # (B, 1) — cleaner end  → frame 0
            low_idx  = pair[:, 0:1]  # (B, 1) — noisier end  → frame T-1
            slope = torch.linspace(0, 1, steps=T, device=self.device).unsqueeze(0)
            progressive_idx = (high_idx + slope * (low_idx - high_idx)).long()
            progressive_idx = progressive_idx.clamp(0, self.num_noise_levels - 1)
            progressive_half_idx = (progressive_idx + delta).clamp(
                0, self.num_noise_levels - 1,
            )
            state_idx[:]       = progressive_idx
            state_half_idx[:]  = progressive_half_idx
            action_idx[:]      = progressive_idx
            action_half_idx[:] = progressive_half_idx
            context_length = 1
        else:
            raise NotImplementedError

        obs_diff = dict(
            tau=state_idx.float() / self.num_noise_levels,
            tau_idx=state_idx,
            tau_plus_half_step_idx=state_half_idx,
        )
        act_diff = dict(
            tau=action_idx.float() / self.num_noise_levels,
            tau_idx=action_idx,
            tau_plus_half_step_idx=action_half_idx,
        )
        return obs_diff, act_diff, context_length, mode, step_index.squeeze(-1), half_step_index.squeeze(-1), step.squeeze(-1)

    def forward(
        self,
        z_clean: torch.Tensor,     # (B, T, N_lat, D_lat)
        a_clean: torch.Tensor,     # (B, T, N_act_tokens, n_actions)   (raw actions from dataset)
    ):
        B, T, N_lat, D_lat = z_clean.shape
        obs_diff, act_diff, context_length, mode, step_index, half_step_index, step = self.sample_step_noise(B, T)
        
        # observation forward diffusion
        z0 = torch.randn_like(z_clean)
        obs_tau = obs_diff["tau"].unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
        z_tau = (1. - obs_tau) * z0 + obs_tau * z_clean

        # action forward diffusion
        a0 = self.action_noise_std * torch.randn_like(a_clean)
        act_tau = act_diff["tau"].unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
        a_tau = (1. - act_tau) * a0 + act_tau * a_clean

        return {
            # obs
            "x": z_clean,
            "x0": z0,
            "x_tau": z_tau,
            "obs_tau": obs_diff["tau"],
            "obs_tau_idx": obs_diff["tau_idx"],
            "obs_tau_plus_half_step_idx": obs_diff["tau_plus_half_step_idx"],
            # act
            "a": a_clean,
            "a0": a0,
            "a_tau": a_tau,
            "act_tau": act_diff["tau"],
            "act_tau_idx": act_diff["tau_idx"],
            "act_tau_plus_half_step_idx": act_diff["tau_plus_half_step_idx"],
            # shortcut schedule
            "context_length": context_length,
            "mode": mode,
            "step_index": step_index,       # (B,) long
            "half_step_index": half_step_index,  # (B,) long
            "step": step,                   # (B,) float, d_t
            "d_min": self.d_min,            # scalar float
        }


def compute_bootstrap_uwm_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    device='cpu',
    teacher: Optional[DreamerV4Denoiser] = None,
    loss_weighting: str = 'ramp',
    rewards: Optional[torch.Tensor] = None,
):
    """
    Shortcut / bootstrap loss for the unified world model (two streams: obs + act).

    When step_index == 0 (d = d_min):  flow loss — direct x-prediction against clean target.
    When step_index > 0 (d > d_min):   bootstrap loss — match big-step velocity to
                                        the mean of two no-grad half-step velocities.

    If `teacher` is provided, the two half-step target forwards are evaluated with it
    (typically an EMA copy of the student). If `teacher is None` the student itself is
    used in no-grad / eval mode (self-bootstrapped target).

    Returns a dict: {obs_flow_loss, act_flow_loss, obs_boot_loss, act_boot_loss,
    reward_loss}. `reward_loss` is None unless the denoiser was built with
    `train_reward_model=True`. Reward MTP is computed only on the big-step
    (student) forward — the half-step target forwards discard the third return.
    """
    # --- unpack info from ShortcutUWMForwardProcess.forward() ---
    x = info["x"]                                      # (B, T, N_lat, D_lat) clean obs
    B, T, N_lat, D_lat = x.shape
    x_tau = info["x_tau"]                               # (B, T, N_lat, D_lat) noisy obs
    obs_tau_idx = info["obs_tau_idx"]                   # (B, T) long

    a = info["a"]                                       # (B, T, 1, n_act) clean act
    a_tau = info["a_tau"]                               # (B, T, 1, n_act) noisy act
    act_tau_idx = info["act_tau_idx"]                   # (B, T) long

    obs_tau = info["obs_tau"]                           # (B, T) float
    act_tau = info["act_tau"]                           # (B, T) float

    step_index = info["step_index"]                     # (B,) long
    half_step_index = info["half_step_index"]           # (B,) long
    step = info["step"]                                 # (B,) float, d_t

    obs_tau_plus_half_idx = info["obs_tau_plus_half_step_idx"]  # (B, T) long
    act_tau_plus_half_idx = info["act_tau_plus_half_step_idx"]  # (B, T) long

    mode = info["mode"]
    context_length = info["context_length"]

    # Expand step-level quantities from (B,) to (B, T) for denoiser
    step_idx_bt = step_index.unsqueeze(1).expand(-1, T)          # (B, T)
    half_step_idx_bt = half_step_index.unsqueeze(1).expand(-1, T)  # (B, T)

    # Broadcast shapes for element-wise ops
    obs_tau_b = obs_tau.unsqueeze(-1).unsqueeze(-1)     # (B, T, 1, 1)
    act_tau_b = act_tau.unsqueeze(-1).unsqueeze(-1)     # (B, T, 1, 1)
    step_b = step.view(B, 1, 1, 1)                     # (B, 1, 1, 1)

    x_tau_det = x_tau.detach()
    a_tau_det = a_tau.detach()

    eps = 1e-4  # denominator clamp to avoid 0/0 for tau ≈ 1

    # =================================================
    # 1) Bootstrap target (no-grad, two half-steps)
    # =================================================
    target_net = teacher if teacher is not None else denoiser
    restore_student_train = False
    with torch.no_grad():
        if teacher is None:
            # self-bootstrap: temporarily flip the student to eval for target.
            # (kept for backward-compat; prefer passing an EMA teacher).
            target_net.eval()
            restore_student_train = True

        # --- first half-step: f(x_τ, τ, d/2) ---
        f1_obs, f1_act, _ = target_net(
            noisy_act=a_tau_det.squeeze(-2),
            noisy_obs=x_tau_det,
            obs_sigma_idx=obs_tau_idx,
            obs_step_idx=half_step_idx_bt,
            act_sigma_idx=act_tau_idx,
            act_step_idx=half_step_idx_bt,
        )
        # velocities
        b1_obs = (f1_obs - x_tau) / (1.0 - obs_tau_b).clamp(min=eps)
        b1_act = (f1_act - a_tau) / (1.0 - act_tau_b).clamp(min=eps)

        # advance to τ + d/2
        z_prime = x_tau + b1_obs * (step_b / 2.0)
        a_prime = a_tau + b1_act * (step_b / 2.0)

        # tau indices at τ + d/2  (already computed by forward process)
        # --- second half-step: f(z', τ+d/2, d/2) ---
        f2_obs, f2_act, _ = target_net(
            noisy_act=a_prime.squeeze(-2),
            noisy_obs=z_prime,
            obs_sigma_idx=obs_tau_plus_half_idx,
            obs_step_idx=half_step_idx_bt,
            act_sigma_idx=act_tau_plus_half_idx,
            act_step_idx=half_step_idx_bt,
        )

        denom2_obs = (1.0 - (obs_tau_b + step_b / 2.0)).clamp(min=eps)
        denom2_act = (1.0 - (act_tau_b + step_b / 2.0)).clamp(min=eps)
        b2_obs = (f2_obs - z_prime) / denom2_obs
        b2_act = (f2_act - a_prime) / denom2_act

        v_target_obs = 0.5 * (b1_obs + b2_obs)
        v_target_act = 0.5 * (b1_act + b2_act)

    if restore_student_train:
        denoiser.train()

    # =================================================
    # 2) Big-step prediction (gradient tracked)
    # =================================================
    z_hat, a_hat, pred_rewards = denoiser(
        noisy_act=a_tau_det.squeeze(-2),
        noisy_obs=x_tau_det,
        obs_sigma_idx=obs_tau_idx,
        obs_step_idx=step_idx_bt,
        act_sigma_idx=act_tau_idx,
        act_step_idx=step_idx_bt,
    )

    # =================================================
    # 3) Per-element losses
    # =================================================
    w_obs = loss_weight(obs_tau, scheme=loss_weighting)  # (B, T)
    w_act = loss_weight(act_tau, scheme=loss_weighting)  # (B, T)

    # --- flow branch (step_index == 0): x-prediction MSE ---
    obs_flow_sq = (z_hat - x).pow(2).mean(dim=(-1, -2))           # (B, T)
    act_flow_sq = (a_hat - a).pow(2).mean(dim=(-1, -2))           # (B, T)

    # --- bootstrap branch (step_index > 0): velocity-matching ---
    v_hat_obs = (z_hat - x_tau) / (1.0 - obs_tau_b).clamp(min=eps)
    v_hat_act = (a_hat - a_tau) / (1.0 - act_tau_b).clamp(min=eps)

    obs_boot_sq = ((1.0 - obs_tau_b) ** 2 * (v_hat_obs - v_target_obs).pow(2)).mean(dim=(-1, -2))  # (B, T)
    act_boot_sq = ((1.0 - act_tau_b) ** 2 * (v_hat_act - v_target_act).pow(2)).mean(dim=(-1, -2))  # (B, T)

    # step_index is (B,) — shared across time; expand to (B, T) masks
    mask_small = (step_index == 0).float().unsqueeze(1).expand(-1, T)  # (B, T)
    mask_large = (step_index > 0).float().unsqueeze(1).expand(-1, T)   # (B, T)

    # =================================================
    # 4) Mode-dependent masking & aggregation
    #    Inactive streams use *.mean()*0. to keep the
    #    computation graph alive for DDP / CUDA graphs.
    # =================================================
    if mode == 'policy':
        # loss only on horizon for both streams
        mask_s = mask_small[:, context_length:]
        mask_l = mask_large[:, context_length:]
        n_f = mask_s.sum().clamp_min(1.0)
        n_b = mask_l.sum().clamp_min(1.0)
        obs_flow_loss = ((obs_flow_sq * w_obs)[:, context_length:] * mask_s).sum() / n_f
        act_flow_loss = ((act_flow_sq * w_act)[:, context_length:] * mask_s).sum() / n_f
        obs_boot_loss = ((obs_boot_sq * w_obs)[:, context_length:] * mask_l).sum() / n_b
        act_boot_loss = ((act_boot_sq * w_act)[:, context_length:] * mask_l).sum() / n_b
    elif mode == 'video':
        # Unconditioned video generation: state loss across the FULL sequence
        # (context + horizon, identical τ), action stream zeroed out.
        n_f = mask_small.sum().clamp_min(1.0)
        n_b = mask_large.sum().clamp_min(1.0)
        obs_flow_loss = (obs_flow_sq * w_obs * mask_small).sum() / n_f
        act_flow_loss = (act_flow_sq * w_act).mean() * 0.
        obs_boot_loss = (obs_boot_sq * w_obs * mask_large).sum() / n_b
        act_boot_loss = (act_boot_sq * w_act).mean() * 0.
    elif mode == 'wm':
        # obs on horizon, act zeroed
        mask_s = mask_small[:, context_length:]
        mask_l = mask_large[:, context_length:]
        n_f = mask_s.sum().clamp_min(1.0)
        n_b = mask_l.sum().clamp_min(1.0)
        obs_flow_loss = ((obs_flow_sq * w_obs)[:, context_length:] * mask_s).sum() / n_f
        act_flow_loss = (act_flow_sq * w_act).mean() * 0.
        obs_boot_loss = ((obs_boot_sq * w_obs)[:, context_length:] * mask_l).sum() / n_b
        act_boot_loss = (act_boot_sq * w_act).mean() * 0.
    elif mode == 'id':
        # obs zeroed, act on all timesteps
        n_f = mask_small.sum().clamp_min(1.0)
        n_b = mask_large.sum().clamp_min(1.0)
        obs_flow_loss = (obs_flow_sq * w_obs).mean() * 0.
        act_flow_loss = (act_flow_sq * w_act * mask_small).sum() / n_f
        obs_boot_loss = (obs_boot_sq * w_obs).mean() * 0.
        act_boot_loss = (act_boot_sq * w_act * mask_large).sum() / n_b
    elif mode == 'forcing':
        # both streams, all timesteps
        n_f = mask_small.sum().clamp_min(1.0)
        n_b = mask_large.sum().clamp_min(1.0)
        obs_flow_loss = (obs_flow_sq * w_obs * mask_small).sum() / n_f
        act_flow_loss = (act_flow_sq * w_act * mask_small).sum() / n_f
        obs_boot_loss = (obs_boot_sq * w_obs * mask_large).sum() / n_b
        act_boot_loss = (act_boot_sq * w_act * mask_large).sum() / n_b
    else:
        raise NotImplementedError

    # --- reward MTP loss (only when the head exists) ---
    reward_loss = None
    if pred_rewards is not None:
        if rewards is None:
            raise RuntimeError(
                "denoiser was built with train_reward_model=True but "
                "compute_bootstrap_uwm_loss was called without `rewards`."
            )
        reward_loss = compute_reward_mtp_loss(pred_rewards, rewards)

    return {
        "obs_flow_loss": obs_flow_loss,
        "act_flow_loss": act_flow_loss,
        "obs_boot_loss": obs_boot_loss,
        "act_boot_loss": act_boot_loss,
        "reward_loss": reward_loss,
    }