import time
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
        images = (images*2.)-1. # Translate the images in +-1 range
        tokens = self.patchifier(images)
        masked_tokens = self.masker(tokens)
        z, _ = self.encoder(masked_tokens)
        z_decoded = self.decoder(z)
        recon_images = self.image_head(z_decoded)
        return  torch.clamp((recon_images + 1)/2., 0., 1.)

@hydra.main(config_path="config", config_name="tokenizer.yaml")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    device = cfg.train.device

    # Instantiate the model
    model = ModelWrapper(cfg)
    model.to(device)
    # model = torch.compile(model, mode="max-autotune", fullgraph=False)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    num_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Number of encoder parameters (M): {num_params/1e6:.2f}M")
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"Number of decoder parameters (M): {num_params/1e6:.2f}M")
    BATCH_PER_GPU = 1
    CONTEXT_T = 96
    C = 3
    H = 256
    W = 256
    imgs = torch.rand(BATCH_PER_GPU, CONTEXT_T, C, H, W).to(torch.float32).to(device)
    model.train()
    epoch_times = []
    epoch_fps = []
    world_size = 1

    # Create an optimizer (you're missing this!)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    NUM_EPOCHS=10
    STEPS_PER_EPOCH = 30
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.perf_counter()
        for _ in range(STEPS_PER_EPOCH):
            torch.compiler.cudagraph_mark_step_begin()
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                recon_images = model(imgs)
                loss = recon_images.sum().to(torch.float32)
            loss.backward()
            # Update parameters
            optimizer.step()
        torch.cuda.synchronize(device=None)
        if epoch >= 1: # Don't include the first wamup epoch
            epoch_end = time.perf_counter()
            epoch_time = epoch_end - epoch_start
            total_frames = BATCH_PER_GPU * CONTEXT_T * world_size * STEPS_PER_EPOCH
            epoch_fps_val = total_frames / epoch_time
            epoch_times.append(epoch_time)
            epoch_fps.append(epoch_fps_val)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Overall Statistics (across {NUM_EPOCHS} epochs):")
    print(f"  Average epoch time: {sum(epoch_times)/len(epoch_times):.2f}s")
    print(f"  Average FPS: {sum(epoch_fps)/len(epoch_fps):.2f}")
    print(f"  Min FPS: {min(epoch_fps):.2f}")
    print(f"  Max FPS: {max(epoch_fps):.2f}")
    
    # Memory report
    cur_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"\nGPU Memory (Rank 0):")
    print(f"  Current: {cur_alloc:.2f} GB")
    print(f"  Peak: {peak_alloc:.2f} GB")
    
    # Per-epoch breakdown
    print(f"\nPer-Epoch Breakdown:")
    for i in range(NUM_EPOCHS-1):
        print(f"  Epoch {i+1}:"
                f"Time={epoch_times[i]:.2f}s, FPS={epoch_fps[i]:.2f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()