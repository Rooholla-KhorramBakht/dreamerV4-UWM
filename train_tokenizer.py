
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from model.tokenizer import (CausalTokenizerDecoder, 
                             CausalTokenizerEncoder, 
                             CausalTokenizerConfig, 
                             TokensToImageHead, 
                             ImagePatchifier)
from model.utils import TokenMasker
import torch.optim as optim
import lpips

@hydra.main(config_path="config", config_name="tokenizer.yaml")
def main(cfg: DictConfig):
    tokenizer_cfg = CausalTokenizerConfig(**OmegaConf.to_object(cfg.tokenizer)) 
    encoder = CausalTokenizerEncoder(tokenizer_cfg)
    decoder = CausalTokenizerDecoder(tokenizer_cfg)
    patchifier = ImagePatchifier(cfg.tokenizer.patch_size, cfg.tokenizer.model_dim)
    image_head = TokensToImageHead(cfg.tokenizer.model_dim, cfg.dataset.resolution, cfg.tokenizer.patch_size)
    masker = TokenMasker(cfg.tokenizer.model_dim, cfg.tokenizer.num_modality_tokens)

    num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Number of encoder parameters (M): {num_params/1e6:.2f}M")
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Number of decoder parameters (M): {num_params/1e6:.2f}M")

    mse_loss_fn = nn.MSELoss()
    lpips_loss_fn = lpips.LPIPS(net='vgg').eval()   # perceptual loss

    # Optimizer
    params = list(encoder.parameters()) + \
             list(decoder.parameters()) + \
             list(patchifier.parameters()) + \
             list(masker.parameters()) + \
             list(image_head.parameters())
    optimizer = optim.Adam(params, lr=cfg.train.lr)

    # ------------------------------
    # 2. Dummy batch creation
    # ------------------------------
    B = cfg.train.batch_size
    C = 3
    H = cfg.dataset.resolution[0]
    W = cfg.dataset.resolution[1]
    T = 16

    images = torch.randn(B, T, C, H, W)
    
    for step in range(int(cfg.train.num_training_steps)):
        optimizer.zero_grad()
        for _ in range(cfg.train.accum_grad_steps):

            tokens = patchifier(images)
            masked_tokens = masker(tokens)
            z, _ = encoder(masked_tokens)
            z_decoded = decoder(z)
            recon_images = image_head(z_decoded)
            mse = mse_loss_fn(recon_images, images)
            lp = lpips_loss_fn(recon_images, images).mean()
            total_loss = mse + cfg.train.lpips_weight * lp
            total_loss.backward()

        optimizer.step()


if __name__ == '__main__':
    main()