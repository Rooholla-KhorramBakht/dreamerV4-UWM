
import time
from PIL import Image
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import ConcatDataset
import kornia.augmentation as K
from dataset import SingleViewSequenceDataset
from model.tokenizer import (CausalTokenizerDecoder, 
                             CausalTokenizerEncoder, 
                             CausalTokenizerConfig, 
                             TokensToImageHead, 
                             ImagePatchifier)
from model.utils import TokenMasker
import torch.optim as optim
import lpips
from pathlib import Path
import wandb

def save_checkpoint(model, optimizer, step, epoch, cfg, scaler=None, scheduler=None):
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_step_{step:07d}.pt"

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "cfg": OmegaConf.to_container(cfg, resolve=True),  # snapshot config
    }

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    torch.save(checkpoint, ckpt_path)
    print(f"🔒 Checkpoint saved at: {str(ckpt_path)}")


def load_checkpoint(ckpt_path, model, optimizer=None, scaler=None, scheduler=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    step = ckpt.get("step", 0)
    epoch = ckpt.get("epoch", 0)

    print(f"🔄 Loaded checkpoint from {ckpt_path} at step={step}, epoch={epoch}")
    return step, epoch

class ModelWrapper(nn.Module):
    def __init__(self, cfg:DictConfig):
        super().__init__()
        self.cfg = cfg
        tokenizer_cfg = CausalTokenizerConfig(**OmegaConf.to_object(cfg.tokenizer)) 
        self.encoder = CausalTokenizerEncoder(tokenizer_cfg)
        self.decoder = CausalTokenizerDecoder(tokenizer_cfg)
        self.patchifier = ImagePatchifier(cfg.tokenizer.patch_size, cfg.tokenizer.model_dim)
        self.image_head = TokensToImageHead(cfg.tokenizer.model_dim, cfg.dataset.resolution, cfg.tokenizer.patch_size)
        self.masker = TokenMasker(cfg.tokenizer.model_dim, cfg.tokenizer.num_modality_tokens)

    def forward(self, images):
        tokens = self.patchifier(images)
        masked_tokens = self.masker(tokens)
        z, _ = self.encoder(masked_tokens)
        z_decoded = self.decoder(z)
        recon_images = self.image_head(z_decoded)
        return recon_images

@hydra.main(config_path="config", config_name="tokenizer.yaml")
def main(cfg: DictConfig):
    # --------------------------------------------------------------------
    #  W&B SETUP
    # --------------------------------------------------------------------
    if cfg.wandb.enable:
        if cfg.wandb.api_key:
            wandb.login(key=cfg.wandb.api_key)
        else:
            wandb.login()   # rely on env var or already logged-in session
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            project=cfg.wandb.project,
            entity="rk4342",
            config=wandb_config
        )
    # Torch configs
    if cfg.train.enable_fast_matmul:
        torch.set_float32_matmul_precision('high')
    if cfg.train.enable_flash_attention:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    device = cfg.train.device

    # Dataset and data loaders
    dataset_base_path = Path(cfg.dataset.base_path)
    train_datasets = [SingleViewSequenceDataset(dataset_base_path/seq_file, cfg.tokenizer.max_context_length, load_to_ram=cfg.dataset.load_to_ram)
            for seq_file in cfg.dataset.train_sequences]
    train_dataset = ConcatDataset(train_datasets)

    test_datasets = [SingleViewSequenceDataset(dataset_base_path/seq_file, cfg.tokenizer.max_context_length, load_to_ram=cfg.dataset.load_to_ram)
            for seq_file in cfg.dataset.test_sequences]
    test_dataset = ConcatDataset(test_datasets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    if cfg.augmentation.enable:
        augmentor = torch.nn.Sequential(
            K.RandomGaussianBlur(tuple(cfg.augmentation.gaussian_blur.kernel_size),
                                 tuple(cfg.augmentation.gaussian_blur.sigma_range), 
                                 p=cfg.augmentation.gaussian_blur.application_probability),
            
            K.ColorJitter(brightness=cfg.augmentation.color_jitter.brightness, 
                          contrast=  cfg.augmentation.color_jitter.contrast, 
                          saturation=cfg.augmentation.color_jitter.saturation,
                          hue=cfg.augmentation.color_jitter.hue,
                          p=cfg.augmentation.color_jitter.application_probability),
            
            K.RandomGaussianNoise(mean=cfg.augmentation.random_noise.mean, 
                                  std=cfg.augmentation.random_noise.std)
        )

    # Instantiate the model
    model = ModelWrapper(cfg)
    model.to(device)
    print('Compiling the model...')
    model = torch.compile(model, mode=cfg.train.torch_compile_mode)

    num_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Number of encoder parameters (M): {num_params/1e6:.2f}M")
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"Number of decoder parameters (M): {num_params/1e6:.2f}M")

    mse_loss_fn = nn.MSELoss()
    # lpips_loss_fn = lpips.LPIPS(net='vgg').eval().to(device)   # perceptual loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    step = 0
    for epoch in range(cfg.train.num_epochs):
        for batch in train_loader:
            step += 1
            ctx_len = torch.randint(1, cfg.tokenizer.max_context_length,(1,)).item()
            imgs = batch['observation.image'][:,:ctx_len, ... ].to(torch.float32).to(device)
            if cfg.augmentation.enable:
                B, T, C, H, W = imgs.shape
                imgs = imgs.view(B*T, C, H, W).contiguous().to(device)      # flatten time
                imgs = augmentor(imgs) 
                imgs = imgs.view(B, T, C, H, W).to(torch.float32)     # restore shape

            optimizer.zero_grad()
            for _ in range(cfg.train.accum_grad_steps):
                if cfg.train.mixed_precision:           
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        recon_images = model(imgs)
                        mse = mse_loss_fn(recon_images, imgs)
                else:
                    recon_images = model(imgs)
                    mse = mse_loss_fn(recon_images, imgs)
                
                #lp=lpips_loss_fn(recon_images.flatten(0,1), images.flatten(0,1)).mean()
                #total_loss = mse + cfg.train.lpips_weight * lp
                total_loss = mse
                total_loss.backward()
            optimizer.step()
            if step % cfg.print_every==0:
                print(f"MSE so far: {total_loss.item()}")

            # ------------------------------------------------------------------
            #  LOG LOSS TO WANDB
            # ------------------------------------------------------------------
            if cfg.wandb.enable and (step % cfg.train.log_every == 0):
                wandb.log({
                    "train/mse_loss": mse.item(),
                }, step=step)


            if step % cfg.plot_every == 0:
                # ------------------------------------------------------------------
                #  EVALUATION STEP ON RANDOM TEST BATCH
                # ------------------------------------------------------------------
                model.eval()
                with torch.no_grad():
                    # sample a random batch
                    try:
                        test_batch = next(test_iter)
                    except:
                        test_iter = iter(test_loader)
                        test_batch = next(test_iter)

                    test_imgs = test_batch['observation.image'].to(torch.float32).to(device)
                    B, T, C, H, W = test_imgs.shape

                    recon = model(test_imgs)               # (B, T, C, H, W)
                    recon = recon.clamp(0.0, 1.0)

                    # --------------------------------------------------------------
                    #  BUILD TILE: odd rows = recon, even rows = GT
                    #  Tile size: (2B rows, T columns)
                    # --------------------------------------------------------------
                    rows = []
                    for b in range(B):
                        row_recon = torch.cat([recon[b, t] for t in range(T)], dim=2)  # concat along width
                        row_gt    = torch.cat([test_imgs[b, t] for t in range(T)], dim=2)
                        rows.append(row_recon)
                        rows.append(row_gt)
                    tiled = torch.cat(rows, dim=1)   # stack vertically

                    # --------------------------------------------------------------
                    #  Convert to PIL Image
                    # --------------------------------------------------------------
                    tiled_np = (tiled.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')

                    tiled_img = Image.fromarray(tiled_np)

                    # --------------------------------------------------------------
                    #  Resize to a W&B-friendly size (max 512px)
                    # --------------------------------------------------------------
                    max_dim = 4096
                    w, h = tiled_img.size
                    scale = min(max_dim / max(w, h), 1.0)
                    new_w, new_h = int(w * scale), int(h * scale)
                    tiled_img_resized = tiled_img.resize((new_w, new_h), Image.BILINEAR)

                    # --------------------------------------------------------------
                    #  Save image locally
                    # --------------------------------------------------------------
                    output_dir = Path(cfg.output_dir) / "plots"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    file_path = output_dir / f"step_{step:06d}.png"
                    tiled_img_resized.save(file_path)

                    # --------------------------------------------------------------
                    #  Optional W&B logging
                    # --------------------------------------------------------------
                    if cfg.wandb.enable:
                        wandb.log({"reconstruction_preview": wandb.Image(tiled_img_resized)}, step=step)
        save_checkpoint(model, optimizer, step, epoch, cfg)

    model.train()
if __name__ == '__main__':
    main()
