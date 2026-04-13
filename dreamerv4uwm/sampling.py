import math 
import numpy as np
import torch

def get_noise_index(noise_level, num_noise_levels):
    return int(np.clip(noise_level, 0., (num_noise_levels-1)/num_noise_levels)*num_noise_levels)

def get_step_index(step_length, num_noise_levels):
    num_steps = int(1./step_length)
    max_pow2 = int(math.log2(num_noise_levels))
    step_index = max_pow2 - int(math.log2(num_steps)) # Convention adopted in my loss
    return step_index

@torch.no_grad
def forward_dynamics_no_cache(
    denoiser,
    ctx_latents,
    actions=None,
    num_pred_steps=1,
    num_diffusion_steps=4,
    context_cond_tau=0.9,
    ):
    
    if actions is not None:
        assert actions.shape[1] == num_pred_steps + ctx_latents.shape[1], 'You should have one action per each context and prediction frames'
        actions = actions.to(device=ctx_latents.device, dtype=ctx_latents.dtype)
    
    B, _, N_lat, D_lat = ctx_latents.shape
    num_context_frames = ctx_latents.shape[1]

    # 1. Initialize pure noise at τ=0
    z = torch.randn(
        B,
        num_pred_steps+num_context_frames,
        N_lat,
        D_lat,
        device=ctx_latents.device,
        dtype=ctx_latents.dtype,
    )
    # Add a slight noise to the context tokens for robustness reasons according to the paper
    latents_cond = ctx_latents.clone()
    latents_cond = (1.0 - context_cond_tau) * torch.randn_like(latents_cond).to(latents_cond.device) + context_cond_tau * latents_cond
    z[:, :num_context_frames, ...] = latents_cond
    
    # Compute the discrete step index  
    step_size = 1.0 / num_diffusion_steps
    denoising_step_index = get_step_index(step_size, denoiser.cfg.denoiser.num_noise_levels)
    step_index_tensor = torch.full(
        (B, num_context_frames+num_pred_steps),
        denoising_step_index,
        dtype=torch.long,
        device=ctx_latents.device,
    )
    
    # Compute the discrete noise level for the context frames with slight noise added on them
    tau_cond_idx = get_noise_index(context_cond_tau, denoiser.cfg.denoiser.num_noise_levels) 

    # Start the shortcut denoising process
    for k in range(num_diffusion_steps):
        
        tau_current = k/num_diffusion_steps # Compute the noise level of the current step (parameter tau in the paper)
        tau_current_idx = get_noise_index(tau_current, denoiser.cfg.denoiser.num_noise_levels)
        tau_index_tensor = torch.full(
            (B, num_context_frames+num_pred_steps),
            tau_current_idx,
            dtype=torch.long,
            device=ctx_latents.device,
        )
        # ste the proper noise level for the context frames 
        tau_index_tensor[:,:num_context_frames] = tau_cond_idx 

        # Denoising
        z_hat = denoiser(
            noisy_z=z,
            action=actions,
            sigma_idx=tau_index_tensor,
            step_idx=step_index_tensor,
        )
        # v = (z_1 - z_τ) / (1 - τ)
        velocity = (z_hat - z) / (1.0 - tau_current)
        # Note: We only apply the denoising process on the future frames
        z[:,num_context_frames:] = z[:,num_context_frames:] + (velocity * step_size)[:,num_context_frames:]
    
    # return torch.cat([latents[:, :num_context_frames, ...], z_hat[:, num_context_frames:]], dim=-3) # z_hat is the output of the last denoiser step which is the predicted clean latents
    # z_hat is the output of the last denoiser step which is the predicted clean latents
    return z_hat[:, num_context_frames:] 

@torch.no_grad()
def forward_dynamics_flowmatching_no_cache(
    denoiser,
    ctx_latents,              # (B, T_ctx, N_lat, D_lat)
    ctx_actions,              # (B, T_ctx, n_act)
    num_pred_steps=1,
    num_diffusion_steps=4,    # power of two
    context_cond_tau=0.9,
):
    """
    Euler sampler for the plain flowmatching denoiser (no shortcut).

    step_idx is always 0 (finest step = d_min), matching compute_flowmatching_loss.
    Both obs latents and actions are jointly denoised.

    Returns
    -------
    pred_obs  : (B, num_pred_steps, N_lat, D_lat)
    pred_act  : (B, num_pred_steps, n_act)
    """
    device = ctx_latents.device
    dtype  = ctx_latents.dtype

    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    n_act   = ctx_actions.shape[-1]
    T_total = T_ctx + num_pred_steps

    assert (num_diffusion_steps & (num_diffusion_steps - 1)) == 0, \
        "num_diffusion_steps must be a power of two"

    num_noise_levels = denoiser.cfg.denoiser.num_noise_levels

    # flowmatching uses only the finest step (step_idx == 0 == d_min)
    step_index_tensor = torch.zeros((B, T_total), dtype=torch.long, device=device)

    # 1) Initialize pure noise for the full sequence
    z     = torch.randn(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
    z_act = torch.randn(B, T_total, n_act,         device=device, dtype=dtype)

    # 2) Replace context frames with slightly noised clean tokens
    z_ctx = (1.0 - context_cond_tau) * torch.randn_like(ctx_latents) + context_cond_tau * ctx_latents
    a_ctx = (1.0 - context_cond_tau) * torch.randn_like(ctx_actions) + context_cond_tau * ctx_actions
    z    [:, :T_ctx] = z_ctx
    z_act[:, :T_ctx] = a_ctx

    # discrete tau index for context frames
    tau_cond_idx = get_noise_index(context_cond_tau, num_noise_levels)

    # stride is exact because both num_noise_levels and num_diffusion_steps are powers of 2
    stride    = num_noise_levels // num_diffusion_steps
    step_size = 1.0 / num_diffusion_steps

    for k in range(num_diffusion_steps):
        tau_current_idx = k * stride
        tau_current     = tau_current_idx / float(num_noise_levels)

        tau_index_tensor = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
        tau_index_tensor[:, :T_ctx] = tau_cond_idx   # context frames stay at tau_c

        z_hat, act_hat = denoiser(
            noisy_act    = z_act,
            noisy_obs    = z,
            obs_sigma_idx= tau_index_tensor,
            obs_step_idx = step_index_tensor,
            act_sigma_idx= tau_index_tensor,
            act_step_idx = step_index_tensor,
        )
        act_hat = act_hat.squeeze(-2)   # (B, T, n_act)

        denom = max(1.0 - tau_current, 1e-5)
        v_obs = (z_hat  - z    ) / denom
        v_act = (act_hat - z_act) / denom

        # integrate only future frames
        z    [:, T_ctx:] = z    [:, T_ctx:] + (v_obs * step_size)[:, T_ctx:]
        z_act[:, T_ctx:] = z_act[:, T_ctx:] + (v_act * step_size)[:, T_ctx:]

    return z[:, T_ctx:], z_act[:, T_ctx:]


@torch.no_grad()
def unified_flowmatching_sampler(
    denoiser,
    ctx_latents,              # (B, T_ctx, N_lat, D_lat)
    mode='wm',                # 'policy', 'video', 'wm', 'id', 'forcing'
    ctx_actions=None,         # (B, T_ctx, n_act) — required for policy, wm, id, forcing
    all_actions=None,         # (B, T_ctx + num_pred_steps, n_act) — alternative: pass all actions at once
    all_latents=None,         # (B, T_ctx + num_pred_steps, N_lat, D_lat) — required for 'id' mode
    num_pred_steps=1,
    num_diffusion_steps=4,    # power of two
    context_cond_tau=0.9,
    action_noise_std=1.0,
):
    """
    Unified Euler sampler for all UWM training modes (no shortcut, no cache).

    Modes match the training loss in UWMForwardProcess / compute_uwm_loss:

      policy  — context: clean obs + clean act; denoise future obs + future act
      video   — context: clean obs, no actions; denoise future obs only
      wm      — context: clean obs + clean act; denoise future obs with all actions clean
      id      — all obs clean; denoise actions across all frames
      forcing — context obs/act conditioned; denoise future obs + future act

    Returns
    -------
    pred_obs : (B, num_pred_steps, N_lat, D_lat) or None
    pred_act : (B, num_pred_steps, n_act)         or None
        None is returned for the modality that is not denoised in a given mode.
    """
    assert mode in ('policy', 'video', 'wm', 'id', 'forcing'), \
        f"Unknown mode '{mode}'"
    assert (num_diffusion_steps & (num_diffusion_steps - 1)) == 0, \
        "num_diffusion_steps must be a power of two"

    device = ctx_latents.device
    dtype  = ctx_latents.dtype
    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_total = T_ctx + num_pred_steps

    num_noise_levels = denoiser.cfg.denoiser.num_noise_levels
    step_index_tensor = torch.zeros((B, T_total), dtype=torch.long, device=device)

    tau_cond_idx = get_noise_index(context_cond_tau, num_noise_levels)
    stride    = num_noise_levels // num_diffusion_steps
    step_size = 1.0 / num_diffusion_steps

    # --- resolve actions tensor ---
    if mode == 'video':
        # video mode: no action signal at all
        n_act = ctx_actions.shape[-1] if ctx_actions is not None else (
            all_actions.shape[-1] if all_actions is not None else 1
        )
    elif mode == 'wm':
        # wm needs actions for ALL frames (context + future)
        assert all_actions is not None, "wm mode requires all_actions (B, T_total, n_act)"
        n_act = all_actions.shape[-1]
    elif mode == 'id':
        # id needs all obs clean
        assert all_latents is not None, "id mode requires all_latents (B, T_total, N_lat, D_lat)"
        if all_actions is not None:
            n_act = all_actions.shape[-1]
        elif ctx_actions is not None:
            n_act = ctx_actions.shape[-1]
        else:
            raise ValueError("id mode requires ctx_actions or all_actions to determine n_act")
    else:
        # policy, forcing
        assert ctx_actions is not None, f"{mode} mode requires ctx_actions"
        n_act = ctx_actions.shape[-1]

    # ====================================================================
    # Initialize obs latents
    # ====================================================================
    denoise_obs = mode != 'id'

    if mode == 'id':
        # obs are fully clean everywhere
        assert all_latents.shape == (B, T_total, N_lat, D_lat)
        z = all_latents.clone().to(device=device, dtype=dtype)
        # set obs sigma to "clean" for all frames
        obs_clean_idx = num_noise_levels - 1
    else:
        z = torch.randn(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
        z_ctx = (1.0 - context_cond_tau) * torch.randn_like(ctx_latents) + context_cond_tau * ctx_latents
        z[:, :T_ctx] = z_ctx

    # ====================================================================
    # Initialize actions
    # ====================================================================
    denoise_act = mode in ('policy', 'id', 'forcing')

    if mode == 'wm':
        # actions are fully clean for all frames
        z_act = all_actions.to(device=device, dtype=dtype)
        act_clean_idx = num_noise_levels - 1
    elif mode == 'video':
        # actions are pure noise everywhere (never denoised, never conditioned)
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)
    elif mode in ('policy', 'forcing'):
        # context actions are slightly noised clean; future actions start as noise
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)
        a_ctx = (1.0 - context_cond_tau) * action_noise_std * torch.randn_like(ctx_actions) + context_cond_tau * ctx_actions.to(device=device, dtype=dtype)
        z_act[:, :T_ctx] = a_ctx
    elif mode == 'id':
        # actions start as noise everywhere
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)

    # ====================================================================
    # Euler integration loop
    # ====================================================================
    for k in range(num_diffusion_steps):
        tau_current_idx = k * stride
        tau_current     = tau_current_idx / float(num_noise_levels)

        # --- obs sigma indices ---
        if mode == 'id':
            obs_sigma_idx = torch.full((B, T_total), obs_clean_idx, dtype=torch.long, device=device)
        else:
            obs_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
            obs_sigma_idx[:, :T_ctx] = tau_cond_idx

        # --- act sigma indices ---
        if mode == 'wm':
            act_sigma_idx = torch.full((B, T_total), act_clean_idx, dtype=torch.long, device=device)
        elif mode == 'video':
            # actions are pure noise (tau=0)
            act_sigma_idx = torch.zeros((B, T_total), dtype=torch.long, device=device)
        elif mode == 'id':
            # denoising actions across all frames at same noise level
            act_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
        else:
            # policy, forcing: context actions conditioned, future denoised
            act_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
            act_sigma_idx[:, :T_ctx] = tau_cond_idx

        z_hat, act_hat = denoiser(
            noisy_act    = z_act,
            noisy_obs    = z,
            obs_sigma_idx= obs_sigma_idx,
            obs_step_idx = step_index_tensor,
            act_sigma_idx= act_sigma_idx,
            act_step_idx = step_index_tensor,
        )
        act_hat = act_hat.squeeze(-2)  # (B, T, n_act)

        denom = max(1.0 - tau_current, 1e-5)

        # --- integrate obs ---
        if denoise_obs:
            v_obs = (z_hat - z) / denom
            z[:, T_ctx:] = z[:, T_ctx:] + (v_obs * step_size)[:, T_ctx:]

        # --- integrate act ---
        if denoise_act:
            v_act = (act_hat - z_act) / denom
            if mode == 'id':
                # denoise actions for ALL frames
                z_act = z_act + v_act * step_size
            else:
                # denoise only future actions
                z_act[:, T_ctx:] = z_act[:, T_ctx:] + (v_act * step_size)[:, T_ctx:]

    # ====================================================================
    # Return results
    # ====================================================================
    out_obs = z[:, T_ctx:] if denoise_obs else None
    out_act = z_act[:, T_ctx:] if denoise_act else None

    # For modes that don't denoise a modality but still have meaningful output
    if mode == 'id':
        out_obs = None   # obs were given as input, not predicted
    if mode in ('video', 'wm'):
        out_act = None   # actions were not predicted

    return out_obs, out_act


@torch.no_grad()
def worldmodel_dynamics_flowmatching_no_cache(
    denoiser,
    ctx_latents,              # (B, T_ctx, N_lat, D_lat)  — context obs frames
    all_actions,              # (B, T_ctx + num_pred_steps, n_act) — ALL actions, clean
    num_pred_steps=1,
    num_diffusion_steps=4,    # power of two
    context_cond_tau=0.9,
):
    """
    World-model Euler sampler for the plain flowmatching denoiser (no shortcut).

    Actions are treated as clean conditions for all frames (context + future).
    Only obs latents are denoised: context frames are conditioned with slight noise,
    future frames start as pure noise and are integrated toward the clean signal.

    Returns
    -------
    pred_obs : (B, num_pred_steps, N_lat, D_lat)
    """
    device = ctx_latents.device
    dtype  = ctx_latents.dtype

    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_total = T_ctx + num_pred_steps

    assert all_actions.shape == (B, T_total, all_actions.shape[-1]), \
        "all_actions must be (B, T_ctx + num_pred_steps, n_act)"
    assert (num_diffusion_steps & (num_diffusion_steps - 1)) == 0, \
        "num_diffusion_steps must be a power of two"

    num_noise_levels = denoiser.cfg.denoiser.num_noise_levels

    # flowmatching uses only the finest step for both modalities
    step_index_tensor = torch.zeros((B, T_total), dtype=torch.long, device=device)

    # actions are fully clean — highest tau index (training convention: tau = (N-1)/N)
    act_clean_idx = num_noise_levels - 1
    act_sigma_idx = torch.full((B, T_total), act_clean_idx, dtype=torch.long, device=device)

    # 1) Initialize obs: context = slightly noised clean, future = pure noise
    z = torch.randn(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
    z_ctx = (1.0 - context_cond_tau) * torch.randn_like(ctx_latents) + context_cond_tau * ctx_latents
    z[:, :T_ctx] = z_ctx

    # discrete tau index for context obs frames
    tau_cond_idx = get_noise_index(context_cond_tau, num_noise_levels)

    # stride is exact because both num_noise_levels and num_diffusion_steps are powers of 2
    stride    = num_noise_levels // num_diffusion_steps
    step_size = 1.0 / num_diffusion_steps

    # clean actions passed as-is for all frames (no integration needed)
    clean_act = all_actions.to(device=device, dtype=dtype)

    for k in range(num_diffusion_steps):
        tau_current_idx = k * stride
        tau_current     = tau_current_idx / float(num_noise_levels)

        obs_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
        obs_sigma_idx[:, :T_ctx] = tau_cond_idx   # context obs stays at tau_c

        z_hat, _ = denoiser(
            noisy_act    = clean_act,
            noisy_obs    = z,
            obs_sigma_idx= obs_sigma_idx,
            obs_step_idx = step_index_tensor,
            act_sigma_idx= act_sigma_idx,
            act_step_idx = step_index_tensor,
        )

        denom = max(1.0 - tau_current, 1e-5)
        v_obs = (z_hat - z) / denom

        # integrate only future obs frames; context and actions are fixed
        z[:, T_ctx:] = z[:, T_ctx:] + (v_obs * step_size)[:, T_ctx:]

    return z[:, T_ctx:]


@torch.no_grad()
def unified_shortcut_sampler(
    denoiser,
    ctx_latents,              # (B, T_ctx, N_lat, D_lat)
    mode='wm',                # 'policy', 'video', 'wm', 'id', 'forcing'
    ctx_actions=None,         # (B, T_ctx, n_act) — required for policy, wm, id, forcing
    all_actions=None,         # (B, T_ctx + num_pred_steps, n_act)
    all_latents=None,         # (B, T_ctx + num_pred_steps, N_lat, D_lat) — required for 'id' mode
    num_pred_steps=1,
    num_diffusion_steps=4,    # power of two
    context_cond_tau=127/128,
    action_noise_std=1.0,
):
    """
    Unified Euler sampler for all UWM training modes with the shortcut
    denoiser. Behaviourally identical to `unified_flowmatching_sampler`,
    except that `step_idx` is set to
    `get_step_index(1/num_diffusion_steps, num_noise_levels)` instead of 0,
    matching compute_bootstrap_uwm_loss.
    """
    assert mode in ('policy', 'video', 'wm', 'id', 'forcing'), \
        f"Unknown mode '{mode}'"
    assert (num_diffusion_steps & (num_diffusion_steps - 1)) == 0, \
        "num_diffusion_steps must be a power of two"

    device = ctx_latents.device
    dtype  = ctx_latents.dtype
    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_total = T_ctx + num_pred_steps

    num_noise_levels = denoiser.cfg.denoiser.num_noise_levels

    step_size = 1.0 / num_diffusion_steps
    denoising_step_index = get_step_index(step_size, num_noise_levels)
    step_index_tensor = torch.full(
        (B, T_total), denoising_step_index, dtype=torch.long, device=device
    )

    tau_cond_idx = get_noise_index(context_cond_tau, num_noise_levels)
    stride = num_noise_levels // num_diffusion_steps

    # --- resolve actions tensor ---
    if mode == 'video':
        n_act = ctx_actions.shape[-1] if ctx_actions is not None else (
            all_actions.shape[-1] if all_actions is not None else 1
        )
    elif mode == 'wm':
        assert all_actions is not None, "wm mode requires all_actions (B, T_total, n_act)"
        n_act = all_actions.shape[-1]
    elif mode == 'id':
        assert all_latents is not None, "id mode requires all_latents (B, T_total, N_lat, D_lat)"
        if all_actions is not None:
            n_act = all_actions.shape[-1]
        elif ctx_actions is not None:
            n_act = ctx_actions.shape[-1]
        else:
            raise ValueError("id mode requires ctx_actions or all_actions to determine n_act")
    else:
        assert ctx_actions is not None, f"{mode} mode requires ctx_actions"
        n_act = ctx_actions.shape[-1]

    # ---- obs init ----
    denoise_obs = mode != 'id'

    if mode == 'id':
        assert all_latents.shape == (B, T_total, N_lat, D_lat)
        z = all_latents.clone().to(device=device, dtype=dtype)
        obs_clean_idx = num_noise_levels - 1
    else:
        z = torch.randn(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
        z_ctx = (1.0 - context_cond_tau) * torch.randn_like(ctx_latents) + context_cond_tau * ctx_latents
        z[:, :T_ctx] = z_ctx

    # ---- act init ----
    denoise_act = mode in ('policy', 'id', 'forcing')

    if mode == 'wm':
        z_act = all_actions.to(device=device, dtype=dtype)
        act_clean_idx = num_noise_levels - 1
    elif mode == 'video':
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)
    elif mode in ('policy', 'forcing'):
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)
        a_ctx = (1.0 - context_cond_tau) * action_noise_std * torch.randn_like(ctx_actions) + context_cond_tau * ctx_actions.to(device=device, dtype=dtype)
        z_act[:, :T_ctx] = a_ctx
    elif mode == 'id':
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)

    # ---- Euler loop ----
    for k in range(num_diffusion_steps):
        tau_current_idx = k * stride
        tau_current     = tau_current_idx / float(num_noise_levels)

        if mode == 'id':
            obs_sigma_idx = torch.full((B, T_total), obs_clean_idx, dtype=torch.long, device=device)
        else:
            obs_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
            obs_sigma_idx[:, :T_ctx] = tau_cond_idx

        if mode == 'wm':
            act_sigma_idx = torch.full((B, T_total), act_clean_idx, dtype=torch.long, device=device)
        elif mode == 'video':
            act_sigma_idx = torch.zeros((B, T_total), dtype=torch.long, device=device)
        elif mode == 'id':
            act_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
        else:
            act_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
            act_sigma_idx[:, :T_ctx] = tau_cond_idx

        z_hat, act_hat = denoiser(
            noisy_act    = z_act,
            noisy_obs    = z,
            obs_sigma_idx= obs_sigma_idx,
            obs_step_idx = step_index_tensor,
            act_sigma_idx= act_sigma_idx,
            act_step_idx = step_index_tensor,
        )
        act_hat = act_hat.squeeze(-2)

        denom = max(1.0 - tau_current, 1e-5)

        if denoise_obs:
            v_obs = (z_hat - z) / denom
            z[:, T_ctx:] = z[:, T_ctx:] + (v_obs * step_size)[:, T_ctx:]

        if denoise_act:
            v_act = (act_hat - z_act) / denom
            if mode == 'id':
                z_act = z_act + v_act * step_size
            else:
                z_act[:, T_ctx:] = z_act[:, T_ctx:] + (v_act * step_size)[:, T_ctx:]

    out_obs = z[:, T_ctx:] if denoise_obs else None
    out_act = z_act[:, T_ctx:] if denoise_act else None

    if mode == 'id':
        out_obs = None
    if mode in ('video', 'wm'):
        out_act = None

    return out_obs, out_act



import time
class AutoRegressiveForwardDynamics:
    def __init__(self, 
                 denoiser, 
                 tokenizer, 
                 context_length=32, 
                 max_forward_steps = 5000,
                 context_cond_tau=0.9, 
                 denoising_step_count=4,
                 device="cuda", 
                 dtype=torch.float32):
        
        self.denoiser = denoiser
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.context_length = context_length
        self.context_cond_tau = context_cond_tau
        self.denoising_step_count = denoising_step_count
        self.max_forward_steps = max_forward_steps
        self.current_frame_index = 0        
    
    @torch.no_grad
    def reset(self, imgs_init, actions_init=None):
        
        self.current_frame_index=0
        self.actions_ctx = actions_init.to(device=self.device, dtype=self.dtype) if actions_init is not None else None
        batch_size = imgs_init.shape[0]

        # Encode the context to compute the context tokens
        latents = self.tokenizer.encode(imgs_init)

        self.current_z = latents[:, -1].unsqueeze(1)
        latents_cond = latents.clone()
        self.latents_cond = latents_cond
        #Initialize the tokenizer decoder KV cache
        self.tokenizer.init_cache(batch_size, context_length=self.context_length, device=self.device, dtype=self.dtype)
        self.tokenizer.decode_step(latents_cond,
                                            start_step_idx = 0,
                                            update_cache = True)
        
        #Initialize the dynamics KV cache
        self.denoiser.init_cache(batch_size, context_length=self.context_length, device=self.device, dtype=self.dtype)    
        latents_cond = (1.0 - self.context_cond_tau) * torch.randn_like(latents_cond).to(latents_cond.device) + self.context_cond_tau * latents_cond
        self.cond_tau_idx = get_noise_index(self.context_cond_tau, self.denoiser.cfg.denoiser.num_noise_levels)
        tau_index_tensor = torch.full(
            (batch_size, latents_cond.shape[1]),
            self.cond_tau_idx,
            dtype=torch.long,
            device=self.device,
        )
        step_size = 1./self.denoising_step_count
        denoising_step_index = get_step_index(step_size, self.denoiser.cfg.denoiser.num_noise_levels)
        step_index_tensor = torch.full(
            (batch_size, latents_cond.shape[1]),
            denoising_step_index,
            dtype=torch.long,
            device=self.device,
        )
        self.denoiser.forward_step(
                            noisy_z = latents_cond,
                            sigma_idx=tau_index_tensor,
                            step_idx=step_index_tensor,
                            action=self.actions_ctx,
                            start_step_idx = 0,
                            update_cache = True)
            
        self.current_frame_index += latents_cond.shape[1]

    @torch.no_grad
    def step(self, actions_t=None):
        if actions_t is not None:
            actions_t = actions_t.to(self.device).to(self.dtype)
        B, _, N, D = self.current_z.shape
        z_t = torch.randn(B, 1, N, D, device=self.device, dtype=self.dtype)
        step_length = 1 / self.denoising_step_count
        step_length_idx = get_step_index(step_length, self.denoiser.cfg.denoiser.num_noise_levels)
        
        for i in range(self.denoising_step_count):
            tau_curr = i / self.denoising_step_count
            curr_tau_idx = get_noise_index(tau_curr, self.denoiser.cfg.denoiser.num_noise_levels)
            tau_idxs = torch.full((B, 1), curr_tau_idx, dtype=torch.long, device=self.device)
            step_idxs = torch.full((B, 1), step_length_idx, dtype=torch.long, device=self.device)
            
            pred = self.denoiser.forward_step(
                action=actions_t, noisy_z=z_t, sigma_idx=tau_idxs,
                step_idx=step_idxs, start_step_idx=self.current_frame_index, update_cache=False
            )
            z_t = z_t + (pred - z_t) / max(1.0 - tau_curr, 1e-5) * step_length


        tau_idxs = torch.full((B, 1), self.cond_tau_idx, dtype=torch.long, device=self.device)
        d_min_idx = get_step_index(1./self.denoiser.cfg.denoiser.num_noise_levels, self.denoiser.cfg.denoiser.num_noise_levels)
        step_idxs = torch.full((B, 1), d_min_idx, dtype=torch.long, device=self.device)
        
        seq_cor_tau = torch.full((B, 1, 1, 1), self.context_cond_tau, dtype=torch.bfloat16, device=self.device)
        eps = torch.randn_like(z_t)
        cor_z_t = (1. - seq_cor_tau) * eps + seq_cor_tau * z_t
            
        self.denoiser.forward_step(
            action=actions_t, noisy_z=cor_z_t, sigma_idx=tau_idxs,
            step_idx=step_idxs, start_step_idx=self.current_frame_index, update_cache=True
        )

        imgs_recon = self.tokenizer.decode_step(z_t,
                                                           start_step_idx = self.current_frame_index,
                                                           update_cache = True)
        
        self.current_z = z_t.clone()
        self.current_frame_index += 1
        return imgs_recon[:,0, ...]