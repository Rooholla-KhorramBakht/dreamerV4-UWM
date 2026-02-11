import math
from typing import Optional, List
import torch
import torch.nn as nn
from .models.dynamics import DreamerV4Denoiser
import torch.distributed as dist


def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """
    Eq. (8): w(τ) = 0.9 τ + 0.1

    tau: (B, T) or (B, T, 1, 1)
    returns: same shape as tau
    """
    return 0.9 * tau + 0.1


# ------------------------------------------------------------
# 5. Forward diffusion with per-frame (τ_t, d_t)
# ------------------------------------------------------------

class ForwardDiffusionWithShortcut(nn.Module):
    """
    Dyadic shortcut schedule for BOTH:
      - obs latents z: (B, T, N_lat, D_lat)
      - actions a:     (B, T, 1, n_actions)   (we enforce num_action_tokens=1)

    Forward mixing:
      x_tau = (1 - tau) * x0 + tau * x_clean
    """

    def __init__(self, num_noise_levels=32):
        super().__init__()
        assert (num_noise_levels & (num_noise_levels - 1)) == 0, "num_noise_levels must be a power of 2"
        self.num_noise_levels = int(num_noise_levels)
        self.max_pow2 = int(math.log2(self.num_noise_levels))
        self.d_min = 1.0 / float(self.num_noise_levels)

    def sample_step_noise(self, batch_size, seq_len, device):
        B, T = batch_size, seq_len

        step_index_raw = torch.randint(
            low=0,
            high=self.max_pow2 + 1,
            size=(B, T),
            device=device,
            dtype=torch.long,
        )

        step = 1.0 / (2.0 ** step_index_raw.float())     # (B,T)
        step_index = self.max_pow2 - step_index_raw      # (B,T)

        num_levels = (2 ** step_index_raw).float()
        m = torch.floor(torch.rand(B, T, device=device) * 0.9999 * num_levels).long()

        tau = m.float() * step                           # (B,T)
        tau_index = m * (2 ** step_index)                # (B,T)

        stride_full = (2 ** step_index)
        delta_tau_index = stride_full // 2

        tau_plus_half_index = torch.clamp(tau_index + delta_tau_index, 0, self.num_noise_levels - 1)
        half_step_index = torch.clamp(step_index - 1, min=0)

        return dict(
            tau=tau,
            step=step,
            tau_index=tau_index,
            step_index=step_index,
            half_step_index=half_step_index,
            tau_plus_half_index=tau_plus_half_index,
        )

    @staticmethod
    def _mix_tau(x0: torch.Tensor, x1: torch.Tensor, tau_bt: torch.Tensor) -> torch.Tensor:
        expand_shape = [tau_bt.shape[0], tau_bt.shape[1]] + [1] * (x1.ndim - 2)
        tau = tau_bt.view(*expand_shape).to(dtype=x1.dtype)
        return (1.0 - tau) * x0 + tau * x1

    def forward(
        self,
        z_clean: torch.Tensor,     # (B, T, N_lat, D_lat)
        a_clean: torch.Tensor,     # (B, T, n_actions)   (raw actions from dataset)
        *,
        action_noise_std: float = 1.0,
        action_clip: float | None = None,
        separate_action_schedule: bool = False,
    ):
        B, T, N_lat, D_lat = z_clean.shape
        device = z_clean.device

        # enforce token dim = 1 for actions
        a_clean_tok = a_clean.unsqueeze(-2)  # (B, T, 1, n_actions)

        obs_diff = self.sample_step_noise(B, T, device)
        act_diff = self.sample_step_noise(B, T, device) if separate_action_schedule else obs_diff

        # noisy obs
        z0 = torch.randn_like(z_clean)
        z_tau = self._mix_tau(z0, z_clean, obs_diff["tau"])

        # noisy actions (in tokenized shape)
        a0 = action_noise_std * torch.randn_like(a_clean_tok)
        a_tau = self._mix_tau(a0, a_clean_tok, act_diff["tau"])
        if action_clip is not None:
            a_tau = torch.clamp(a_tau, -action_clip, action_clip)

        return {
            # obs
            "x": z_clean,
            "x_tau": z_tau,
            "obs_tau": obs_diff["tau"],
            "obs_step": obs_diff["step"],
            "obs_tau_index": obs_diff["tau_index"],
            "obs_step_index": obs_diff["step_index"],
            "obs_half_step_index": obs_diff["half_step_index"],
            "obs_tau_plus_half_index": obs_diff["tau_plus_half_index"],

            # act (kept as tokens with S_a = 1)
            "a": a_clean_tok,     # (B,T,1,n_actions)
            "a_tau": a_tau,       # (B,T,1,n_actions)
            "act_tau": act_diff["tau"],
            "act_step": act_diff["step"],
            "act_tau_index": act_diff["tau_index"],
            "act_step_index": act_diff["step_index"],
            "act_half_step_index": act_diff["half_step_index"],
            "act_tau_plus_half_index": act_diff["tau_plus_half_index"],

            # constants
            "d_min": self.d_min,
            "num_noise_levels": self.num_noise_levels,
        }

import torch

def compute_bootstrap_diffusion_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
):
    # --- obs ---
    x = info["x"]
    x_tau = info["x_tau"]
    obs_tau = info["obs_tau"]
    obs_step = info["obs_step"]
    obs_tau_index = info["obs_tau_index"]
    obs_step_index = info["obs_step_index"]
    obs_half_step_index = info["obs_half_step_index"]
    obs_tau_plus_half_index = info["obs_tau_plus_half_index"]

    # --- act (S_a = 1) ---
    a = info["a"]         # (B,T,1,n_actions)
    a_tau = info["a_tau"] # (B,T,1,n_actions)
    act_tau = info["act_tau"]
    act_step = info["act_step"]
    act_tau_index = info["act_tau_index"]
    act_step_index = info["act_step_index"]
    act_half_step_index = info["act_half_step_index"]
    act_tau_plus_half_index = info["act_tau_plus_half_index"]

    B, T, N_lat, D_lat = x.shape

    # broadcast for obs
    obs_tau_b = obs_tau.view(B, T, 1, 1)
    obs_step_b = obs_step.view(B, T, 1, 1)

    # broadcast for act (a is (B,T,1,A) => make (B,T,1,1))
    act_tau_b = act_tau.view(B, T, 1, 1)
    act_step_b = act_step.view(B, T, 1, 1)

    x_tau_detached = x_tau.detach()
    a_tau_detached = a_tau.detach()

    # =================================================
    # 1) Bootstrap Target Calculation
    # =================================================
    with torch.no_grad():
        denoiser.eval()

        # pass noisy_act as (B,T,n_actions) because denoiser expects that
        f1_obs, f1_act = denoiser(
            noisy_act=a_tau_detached.squeeze(-2),     # (B,T,A)
            noisy_obs=x_tau_detached,                 # (B,T,N_lat,D_lat)
            obs_sigma_idx=obs_tau_index,
            obs_step_idx=obs_half_step_index,
            act_sigma_idx=act_tau_index,
            act_step_idx=act_half_step_index,
        )  # f1_act: (B,T,1,A)

        # ---- obs bootstrap target ----
        b1_obs = (f1_obs - x_tau) / (1.0 - obs_tau_b)
        z_prime = x_tau + b1_obs * (obs_step_b / 2.0)

        # ---- act bootstrap target ----
        b1_act = (f1_act - a_tau) / (1.0 - act_tau_b)
        a_prime = a_tau + b1_act * (act_step_b / 2.0)

        f2_obs, f2_act = denoiser(
            noisy_act=a_prime.squeeze(-2),            # (B,T,A)
            noisy_obs=z_prime,
            obs_sigma_idx=obs_tau_plus_half_index,
            obs_step_idx=obs_half_step_index,
            act_sigma_idx=act_tau_plus_half_index,
            act_step_idx=act_half_step_index,
        )

        denom2_obs = 1.0 - (obs_tau_b + obs_step_b / 2.0)
        b2_obs = (f2_obs - z_prime) / denom2_obs
        v_target_obs = 0.5 * (b1_obs + b2_obs)

        denom2_act = 1.0 - (act_tau_b + act_step_b / 2.0)
        b2_act = (f2_act - a_prime) / denom2_act
        v_target_act = 0.5 * (b1_act + b2_act)

    denoiser.train()

    # =================================================
    # 2) Big-step prediction (grad tracked)
    # =================================================
    z_hat, a_hat = denoiser(
        noisy_act=a_tau_detached.squeeze(-2).clone().requires_grad_(True),  # (B,T,A)
        noisy_obs=x_tau_detached.clone().requires_grad_(True),
        obs_sigma_idx=obs_tau_index,
        obs_step_idx=obs_step_index,
        act_sigma_idx=act_tau_index,
        act_step_idx=act_step_index,
    )  # a_hat: (B,T,1,A)

    # =================================================
    # 3) Losses (obs)
    # =================================================
    w_obs = ramp_weight(obs_tau)

    obs_flow_sq = (z_hat - x).pow(2).mean(dim=(-1, -2))  # (B,T)
    obs_mask_small = (obs_step_index == 0).float()
    obs_flow_loss = (w_obs * obs_flow_sq * obs_mask_small).sum() / obs_mask_small.sum().clamp_min(1.0)

    v_hat_obs = (z_hat - x_tau) / (1.0 - obs_tau_b)
    obs_boot_sq = ((1.0 - obs_tau_b) ** 2 * (v_hat_obs - v_target_obs).pow(2)).mean(dim=(-1, -2))
    obs_mask_large = (obs_step_index > 0).float()
    obs_bootstrap_loss = (w_obs * obs_boot_sq * obs_mask_large).sum() / obs_mask_large.sum().clamp_min(1.0)

    # =================================================
    # 4) Losses (act)
    # =================================================
    w_act = ramp_weight(act_tau)

    act_flow_sq = (a_hat - a).pow(2).mean(dim=(-1, -2))  # (B,T) since (B,T,1,A)
    act_mask_small = (act_step_index == 0).float()
    act_flow_loss = (w_act * act_flow_sq * act_mask_small).sum() / act_mask_small.sum().clamp_min(1.0)

    v_hat_act = (a_hat - a_tau) / (1.0 - act_tau_b)
    act_boot_sq = ((1.0 - act_tau_b) ** 2 * (v_hat_act - v_target_act).pow(2)).mean(dim=(-1, -2))
    act_mask_large = (act_step_index > 0).float()
    act_bootstrap_loss = (w_act * act_boot_sq * act_mask_large).sum() / act_mask_large.sum().clamp_min(1.0)

    return obs_flow_loss, obs_bootstrap_loss, act_flow_loss, act_bootstrap_loss


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