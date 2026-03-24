import math
from typing import Optional, List
import torch
import torch.nn as nn
from .models.dynamics import DreamerV4Denoiser
import torch.distributed as dist
import random


def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """
    Eq. (8): w(τ) = 0.9 τ + 0.1

    tau: (B, T) or broadcastable shape
    returns: same shape as tau
    """
    return 0.9 * tau + 0.1


class FlowMatchingForwardProcess(nn.Module):
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
                device='cpu'):
        super().__init__()
        self.max_diff_steps = max_diff_steps
        self.device = device
        self.action_noise_std = action_noise_std

    def sample_step_noise(self, batch_size, seq_len, context_length=None):
        B, T = batch_size, seq_len
        # Diffusion forcing noise level
        tau_d = torch.randint(0, self.max_diff_steps, (B,T)).to(self.device)
        tau = tau_d/self.max_diff_steps
        
        if context_length is not None:
            tau[:, :context_length] =  0.9999                     # lownoise for the context 
            tau[:, context_length:] = tau[:, context_length].unsqueeze(-1)  # same noise level across prediction chunk

        
        tau_idx = (tau*self.max_diff_steps).to(torch.long)
        
        return dict(
            tau=tau,
            tau_idx = tau_idx.to(self.device),
        )

    def forward(
        self,
        z_clean: torch.Tensor,     # (B, T, N_lat, D_lat)
        a_clean: torch.Tensor,     # (B, T, N_act_tokens, n_actions)   (raw actions from dataset)
        context_length = None,
    ):
        B, T, N_lat, D_lat = z_clean.shape
        device = z_clean.device
        obs_diff = self.sample_step_noise(B, T, context_length=context_length)
        act_diff = self.sample_step_noise(B, T, context_length=context_length)  
        
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
        }

def compute_flowmatching_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    device='cpu', 
):
    
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

    z_hat, a_hat = denoiser(
        noisy_act=a_tau.squeeze(-2),  # (B,T,A) — denoiser expects (B,T,n_actions)
        noisy_obs=x_tau,
        obs_sigma_idx=obs_tau_idx,
        obs_step_idx=step_idx,
        act_sigma_idx=act_tau_idx,
        act_step_idx=step_idx,
    )  # a_hat: (B,T,1,A)

    # X-prediction targets: directly regress clean signal
    obs_x_target = x                    # (B, T, N_lat, D_lat)
    act_x_target = a                    # (B, T, 1, n_actions)

    obs_flow_sq = (z_hat - obs_x_target).pow(2).mean(dim=(-1, -2))  # (B, T)
    act_flow_sq = (a_hat - act_x_target).pow(2).mean(dim=(-1, -2))  # (B, T)
    w_obs      = ramp_weight(info['obs_tau'].squeeze())           # (B, T)
    w_act      = ramp_weight(info['act_tau'].squeeze())           # (B, T)
    obs_flow_loss = (obs_flow_sq*w_obs).mean()
    act_flow_loss = (act_flow_sq*w_act).mean()

    return obs_flow_loss, act_flow_loss

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
                 device='cpu'):
        super().__init__()
        self.max_diff_steps = max_diff_steps
        self.device = device
        self.action_noise_std = action_noise_std
        self.modes = ['policy', 'video', 'wm', 'id', 'forcing']
        if mode_weights is not None:
            weights = [float(mode_weights.get(m, 0.0)) for m in self.modes]
        else:
            weights = [1.0] * len(self.modes)
        total = sum(weights)
        assert total > 0, "At least one mode must have a positive weight"
        self.mode_probs = [w / total for w in weights]

    def sample_step_noise(self, batch_size, seq_len):
        B, T = batch_size, seq_len
        # Diffusion forcing noise level
        state_tau_d = torch.randint(0, self.max_diff_steps, (B,T)).to(self.device)
        state_tau = state_tau_d/self.max_diff_steps

        action_tau_d = torch.randint(0, self.max_diff_steps, (B,T)).to(self.device)
        action_tau = action_tau_d/self.max_diff_steps
        
        context_length=torch.randint(1, T, (1,)).item() # Choose a random context length
        mode = random.choices(self.modes, weights=self.mode_probs, k=1)[0]
        if mode == 'policy':
            # context: clean state, clean action ; chunk: noisy state, noisy action
            state_tau[:, :context_length] =  0.9999                               
            action_tau[:, :context_length] =  0.9999                                
            state_tau[:, context_length:] = state_tau[:, context_length].unsqueeze(-1)    
            action_tau[:, context_length:] = action_tau[:, context_length].unsqueeze(-1) 
        elif mode =='video':
            # context: clean state, noisy action ; chunk: noisy state, noisy action
            state_tau[:, :context_length] =  0.9999                               
            action_tau[:, :context_length] = 0.                                
            state_tau[:, context_length:] = state_tau[:, context_length].unsqueeze(-1)    
            action_tau[:, context_length:] = 0. 
        elif mode=='wm':
            # context: clean state, clean action ; chunk: noisy state, clean action
            state_tau[:, :context_length] =  0.9999                               
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
            # Chunked diffusion forcing mode
            # context: state and action with noise level 1 ; state and action with noise level 2
            state_tau[:, :context_length] =  state_tau[:, 0].unsqueeze(-1)                          
            action_tau[:, :context_length] = action_tau[:, 0].unsqueeze(-1)
            state_tau[:, context_length:] =  state_tau[:, context_length].unsqueeze(-1)   
            action_tau[:, context_length:] = action_tau[:, context_length].unsqueeze(-1)
        else:
            raise NotImplementedError

        state_tau_idx = (state_tau*self.max_diff_steps).to(torch.long)
        action_tau_idx = (action_tau*self.max_diff_steps).to(torch.long)
        obs_diff = dict(tau=state_tau, tau_idx = state_tau_idx.to(self.device))
        act_diff = dict(tau=action_tau, tau_idx = action_tau_idx.to(self.device))
        return obs_diff, act_diff, context_length, mode

    def forward(
        self,
        z_clean: torch.Tensor,     # (B, T, N_lat, D_lat)
        a_clean: torch.Tensor,     # (B, T, N_act_tokens, n_actions)   (raw actions from dataset)
    ):
        B, T, N_lat, D_lat = z_clean.shape
        device = z_clean.device
        obs_diff, act_diff, context_length, mode = self.sample_step_noise(B, T)
        
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
            "mode": mode
        }
    
def compute_uwm_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    device='cpu', 
):
    
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

    z_hat, a_hat = denoiser(
        noisy_act=a_tau.squeeze(-2),  # (B,T,A) — denoiser expects (B,T,n_actions)
        noisy_obs=x_tau,
        obs_sigma_idx=obs_tau_idx,
        obs_step_idx=step_idx,
        act_sigma_idx=act_tau_idx,
        act_step_idx=step_idx,
    )  # a_hat: (B,T,1,A)

    # X-prediction targets: directly regress clean signal
    obs_x_target = x                    # (B, T, N_lat, D_lat)
    act_x_target = a                    # (B, T, 1, n_actions)

    obs_flow_sq = (z_hat - obs_x_target).pow(2).mean(dim=(-1, -2))  # (B, T)
    act_flow_sq = (a_hat - act_x_target).pow(2).mean(dim=(-1, -2))  # (B, T)
    w_obs      = ramp_weight(info['obs_tau'].squeeze())           # (B, T)
    w_act      = ramp_weight(info['act_tau'].squeeze())           # (B, T)

    mode = info['mode']
    context_length = info['context_length']
    if mode=='policy':
        obs_flow_loss = (obs_flow_sq*w_obs)[:, context_length:].mean()
        act_flow_loss = (act_flow_sq*w_act)[:, context_length:].mean()
    elif mode=='video':
        obs_flow_loss = (obs_flow_sq*w_obs)[:, context_length:].mean()
        act_flow_loss = (act_flow_sq*w_act).mean()*0.

    elif mode=='wm':
        obs_flow_loss = (obs_flow_sq*w_obs)[:, context_length:].mean()
        act_flow_loss = (act_flow_sq*w_act).mean()*0.

    elif mode=='id':
        obs_flow_loss = (obs_flow_sq*w_obs).mean()*0.
        act_flow_loss = (act_flow_sq*w_act).mean()

    elif mode=='forcing':
        obs_flow_loss = (obs_flow_sq*w_obs).mean()
        act_flow_loss = (act_flow_sq*w_act).mean()
    else:
        raise NotImplementedError
        
    return obs_flow_loss, act_flow_loss

## To Finish later --->
class JointForwardDiffusionWithShortcut(nn.Module):
    """
    Dyadic shortcut schedule for BOTH obs latents and actions.

    Shortcut length (step/d) is shared between modalities so the denoiser always
    jumps the same distance in time.  Noise levels (τ_obs, τ_act) are sampled
    independently, enabling the denoiser to be used in single-modality or masked
    settings at inference time.

    num_noise_levels must be a power of 2, e.g. 32 or 64.

    Index convention:
      max_pow2 = log2(num_noise_levels)
      step_index = 0       ↔ d = d_min   (finest)
      step_index = max_pow2 ↔ d = 1      (coarsest)
    """

    def __init__(self, num_noise_levels: int = 32, action_noise_std: float = 1.0):
        super().__init__()
        assert (num_noise_levels & (num_noise_levels - 1)) == 0, \
            "num_noise_levels must be a power of 2"
        self.num_noise_levels = int(num_noise_levels)
        self.max_pow2 = int(math.log2(self.num_noise_levels))
        self.d_min = 1.0 / float(self.num_noise_levels)
        self.action_noise_std = action_noise_std

    def _sample_tau(self, step_index_raw, num_levels, step, step_index, device):
        """Sample an independent τ on the grid defined by step_index_raw."""
        B, T = step_index_raw.shape
        m = torch.floor(
            torch.rand(B, T, device=device) * 0.9999 * num_levels
        ).long()
        tau       = m.float() * step                                        # (B, T)
        tau_index = m * (2 ** step_index)                                   # (B, T)
        delta     = (2 ** step_index) // 2
        tau_plus_half_index = torch.clamp(
            tau_index + delta, min=0, max=self.num_noise_levels - 1
        )
        return tau, tau_index, tau_plus_half_index

    def sample_step_noise(self, batch_size: int, seq_len: int, device):
        """
        Returns per-frame diffusion parameters.

        Shared across modalities:
          step            : (B, T) float   d_t ∈ {1, 1/2, ..., 1/num_noise_levels}
          step_index      : (B, T) long    0 ↔ d_min, max_pow2 ↔ 1
          half_step_index : (B, T) long    step_index - 1 (clamped at 0)

        Independent per modality:
          tau_obs / tau_act               : (B, T) float  τ ∈ {0, d, 2d, ..., 1-d}
          tau_obs_index / tau_act_index   : (B, T) long   τ index on finest grid
          tau_obs_plus_half / tau_act_plus_half : (B, T) long  index for τ + d/2
        """
        B, T = batch_size, seq_len

        # --- shared step ---
        step_index_raw = torch.randint(
            low=0, high=self.max_pow2 + 1, size=(B, T), device=device, dtype=torch.long
        )
        step       = 1.0 / (2.0 ** step_index_raw.float())    # (B, T)
        step_index = self.max_pow2 - step_index_raw            # (B, T), 0 ↔ d_min
        num_levels = (2 ** step_index_raw).float()             # (B, T)
        half_step_index = torch.clamp(step_index - 1, min=0)  # (B, T)

        # --- independent τ for each modality ---
        tau_obs, tau_obs_index, tau_obs_plus_half = self._sample_tau(
            step_index_raw, num_levels, step, step_index, device
        )
        tau_act, tau_act_index, tau_act_plus_half = self._sample_tau(
            step_index_raw, num_levels, step, step_index, device
        )

        return dict(
            # shared
            step=step,
            step_index=step_index,
            half_step_index=half_step_index,
            # obs-specific
            tau_obs=tau_obs,
            tau_obs_index=tau_obs_index,
            tau_obs_plus_half=tau_obs_plus_half,
            # act-specific
            tau_act=tau_act,
            tau_act_index=tau_act_index,
            tau_act_plus_half=tau_act_plus_half,
        )

    def forward(
        self,
        z_clean: torch.Tensor,  # (B, T, N_lat, D_lat)
        a_clean: torch.Tensor,  # (B, T, N_act_tokens, n_actions)
    ):
        B, T, N_lat, D_lat = z_clean.shape
        device = z_clean.device

        diff = self.sample_step_noise(B, T, device)

        tau_obs_b = diff["tau_obs"].unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)
        tau_act_b = diff["tau_act"].unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)

        # obs forward diffusion: z_τ = (1 - τ_obs) z_0 + τ_obs z_1
        z0    = torch.randn_like(z_clean)
        z_tau = (1.0 - tau_obs_b) * z0 + tau_obs_b * z_clean

        # action forward diffusion: a_τ = (1 - τ_act) a_0 + τ_act a_1
        a0    = self.action_noise_std * torch.randn_like(a_clean)
        a_tau = (1.0 - tau_act_b) * a0 + tau_act_b * a_clean

        return {
            # clean / noisy obs
            "x":     z_clean,
            "x0":    z0,
            "x_tau": z_tau,
            # clean / noisy action
            "a":     a_clean,
            "a0":    a0,
            "a_tau": a_tau,
            # shared step schedule
            "step":             diff["step"],
            "step_index":       diff["step_index"],
            "half_step_index":  diff["half_step_index"],
            # obs noise schedule
            "tau_obs":              diff["tau_obs"],
            "tau_obs_index":        diff["tau_obs_index"],
            "tau_obs_plus_half":    diff["tau_obs_plus_half"],
            # act noise schedule
            "tau_act":              diff["tau_act"],
            "tau_act_index":        diff["tau_act_index"],
            "tau_act_plus_half":    diff["tau_act_plus_half"],
            "d_min":            self.d_min,
            "num_noise_levels": self.num_noise_levels,
        }

def compute_joint_bootstrap_diffusion_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
):
    """
    Bootstrap shortcut loss for JOINT obs + action denoising.

    Shortcut length (step/d) is shared; noise levels (τ_obs, τ_act) are independent.

    Returns
    -------
    obs_flow_loss, act_flow_loss, obs_bootstrap_loss, act_bootstrap_loss
      Each is a scalar tensor.
    """
    x     = info["x"]        # (B, T, N_lat, D_lat)
    x_tau = info["x_tau"]    # (B, T, N_lat, D_lat)
    a     = info["a"]        # (B, T, 1, n_actions)
    a_tau = info["a_tau"]    # (B, T, 1, n_actions)

    step             = info["step"]               # (B, T)
    step_index       = info["step_index"]         # (B, T) long
    half_step_index  = info["half_step_index"]    # (B, T) long
    tau_obs          = info["tau_obs"]            # (B, T)
    tau_obs_index    = info["tau_obs_index"]      # (B, T) long
    tau_obs_plus_half= info["tau_obs_plus_half"]  # (B, T) long
    tau_act          = info["tau_act"]            # (B, T)
    tau_act_index    = info["tau_act_index"]      # (B, T) long
    tau_act_plus_half= info["tau_act_plus_half"]  # (B, T) long

    B, T, N_lat, D_lat = x.shape
    tau_obs_b = tau_obs.view(B, T, 1, 1)
    tau_act_b = tau_act.view(B, T, 1, 1)
    step_b    = step.view(B, T, 1, 1)

    x_tau_det = x_tau.detach()
    a_tau_det = a_tau.detach()

    # ---------------------------------------------------------------
    # 1) Bootstrap target via two half-steps  (no gradient)
    # ---------------------------------------------------------------
    with torch.no_grad():
        denoiser.eval()

        # --- first half-step ---
        f_obs1, f_act1 = denoiser(
            noisy_act=a_tau_det.squeeze(-2),
            noisy_obs=x_tau_det,
            obs_sigma_idx=tau_obs_index,
            obs_step_idx=half_step_index,
            act_sigma_idx=tau_act_index,
            act_step_idx=half_step_index,
        )
        b_obs1 = (f_obs1 - x_tau) / (1.0 - tau_obs_b)
        b_act1 = (f_act1 - a_tau) / (1.0 - tau_act_b)

        z_prime = x_tau + b_obs1 * (step_b / 2.0)
        a_prime = a_tau + b_act1 * (step_b / 2.0)

        # --- second half-step ---
        f_obs2, f_act2 = denoiser(
            noisy_act=a_prime.squeeze(-2),
            noisy_obs=z_prime,
            obs_sigma_idx=tau_obs_plus_half,
            obs_step_idx=half_step_index,
            act_sigma_idx=tau_act_plus_half,
            act_step_idx=half_step_index,
        )
        b_obs2 = (f_obs2 - z_prime) / (1.0 - (tau_obs_b + step_b / 2.0))
        b_act2 = (f_act2 - a_prime) / (1.0 - (tau_act_b + step_b / 2.0))

        v_obs_target = 0.5 * (b_obs1 + b_obs2)   # (B, T, N_lat, D_lat)
        v_act_target = 0.5 * (b_act1 + b_act2)   # (B, T, 1, n_actions)

    denoiser.train()

    # ---------------------------------------------------------------
    # 2) Full-step prediction  (gradient tracked through denoiser)
    # ---------------------------------------------------------------
    z_hat, a_hat = denoiser(
        noisy_act=a_tau_det.clone().requires_grad_(True).squeeze(-2),
        noisy_obs=x_tau_det.clone().requires_grad_(True),
        obs_sigma_idx=tau_obs_index,
        obs_step_idx=step_index,
        act_sigma_idx=tau_act_index,
        act_step_idx=step_index,
    )  # z_hat: (B,T,N_lat,D_lat)  a_hat: (B,T,1,n_actions)

    # ---------------------------------------------------------------
    # 3) Losses
    # ---------------------------------------------------------------
    w_obs      = ramp_weight(tau_obs)           # (B, T)
    w_act      = ramp_weight(tau_act)           # (B, T)
    mask_small = (step_index == 0).float()      # finest-step frames
    mask_large = (step_index > 0).float()       # coarser-step frames

    # --- flow losses (finest step only, direct x-prediction) ---
    obs_flow_sq = (z_hat - x).pow(2).mean(dim=(-1, -2))     # (B, T)
    act_flow_sq = (a_hat - a).pow(2).mean(dim=(-1, -2))     # (B, T)

    denom_flow = mask_small.sum().clamp_min(1.0)
    obs_flow_loss = (w_obs * obs_flow_sq * mask_small).sum() / denom_flow
    act_flow_loss = (w_act * act_flow_sq * mask_small).sum() / denom_flow

    # --- bootstrap losses (coarser steps only) ---
    v_obs_hat = (z_hat - x_tau) / (1.0 - tau_obs_b)
    v_act_hat = (a_hat - a_tau) / (1.0 - tau_act_b)

    obs_boot_sq = ((1.0 - tau_obs_b) ** 2 * (v_obs_hat - v_obs_target).pow(2)).mean(dim=(-1, -2))
    act_boot_sq = ((1.0 - tau_act_b) ** 2 * (v_act_hat - v_act_target).pow(2)).mean(dim=(-1, -2))

    denom_boot = mask_large.sum().clamp_min(1.0)
    obs_bootstrap_loss = (w_obs * obs_boot_sq * mask_large).sum() / denom_boot
    act_bootstrap_loss = (w_act * act_boot_sq * mask_large).sum() / denom_boot

    return obs_flow_loss, act_flow_loss, obs_bootstrap_loss, act_bootstrap_loss

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