import torch

def create_temporal_mask(T, device = "cpu"):
    S = T
    mask = torch.ones(S, S, dtype=torch.bool, device=device)
    for t_q in range(T):
        for t_k in range(T):
            if t_k < t_q:
                mask[t_q, t_k] = False
    return mask

def create_encoder_spatial_mask(N_patch, N_latent, device="cpu"):
    S = N_patch + N_latent
    mask = torch.zeros(S, S, dtype=torch.bool, device=device)
    mask[0:N_patch, 0:N_patch] = True
    mask[N_patch:S, 0:] = True
    return mask

def create_decoder_spatial_mask(N_patch, N_latent, device="cpu"):
    S = N_patch + N_latent
    mask = torch.zeros(S, S, dtype=torch.bool, device=device)
    mask[0:N_patch, 0:] = True
    mask[N_patch:, N_patch:] = True
    return mask