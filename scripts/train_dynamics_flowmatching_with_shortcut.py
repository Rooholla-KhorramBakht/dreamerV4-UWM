import math
import os
import time

import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dreamerv4.datasets import create_distributed_dataloader
from dreamerv4.loss import JointForwardDiffusionWithShortcut, compute_joint_bootstrap_diffusion_loss
from dreamerv4.models.dynamics import DenoiserWrapper
from dreamerv4.models.utils import load_denoiser, load_tokenizer
from dreamerv4.utils.distributed import (
    cleanup_distributed,
    load_ddp_checkpoint,
    save_ddp_checkpoint,
    setup_distributed,
)


# Probability of using a random context length instead of full sequence.
# When triggered, context length is sampled uniformly from [1, T//2].
CONTEXT_PROB = 0.2


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-8):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / optimizer.defaults["lr"], cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def build_dataloader(cfg, rank, world_size):
    loader, sampler, _ = create_distributed_dataloader(
        data_dir=cfg.dataset.data_dir,
        window_size=cfg.denoiser.max_sequence_length,
        batch_size=cfg.train.batch_per_gpu,
        rank=rank,
        world_size=world_size,
        num_workers=cfg.train.num_workers,
        stride=1,
        seed=cfg.seed,
        split="train",
        train_fraction=cfg.dataset.train_episodes_fraction,
        split_seed=cfg.dataset.split_seed,
        shuffle=True,
        drop_last=True,
    )
    return loader, sampler


def build_models(cfg, device, local_rank):
    tokenizer = load_tokenizer(
        cfg, device=device, max_num_forward_steps=cfg.denoiser.max_sequence_length
    )

    if cfg.dynamics_ckpt:
        print(f"Loading dynamics from: {cfg.dynamics_ckpt}")
        denoiser = load_denoiser(
            cfg, device=device, model_key="model",
            max_num_forward_steps=cfg.denoiser.max_sequence_length,
        )
    else:
        denoiser = DenoiserWrapper(cfg, max_num_forward_steps=cfg.denoiser.max_sequence_length)

    diffuser = JointForwardDiffusionWithShortcut(
        num_noise_levels=cfg.denoiser.num_noise_levels,
    )

    tokenizer = tokenizer.to(device)
    denoiser = denoiser.to(device)

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    if cfg.train.use_compile:
        denoiser = torch.compile(denoiser, mode="max-autotune-no-cudagraphs", fullgraph=True)
        tokenizer = torch.compile(tokenizer, mode="max-autotune-no-cudagraphs", fullgraph=False)

    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=False)

    return tokenizer, denoiser, diffuser


def setup_logging(cfg, rank, world_size, log_dir, wandb_run_id):
    """Initialize TensorBoard and (optionally) W&B on rank 0. Returns (tb_writer, wandb_run_id)."""
    if rank != 0:
        return None, None

    if log_dir is None:
        log_dir = cfg.output_dir
        os.makedirs(log_dir, exist_ok=True)
    else:
        print(f"Reusing log directory: {log_dir}")

    tb_log_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)

    if cfg.wandb.enable:
        if wandb_run_id is not None:
            wandb.init(
                project=cfg.wandb.project,
                id=wandb_run_id,
                resume="must",
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,
                dir=log_dir,
            )
        else:
            wandb.init(
                project=cfg.wandb.project,
                name=f"run_num_gpus{world_size}",
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,
                dir=log_dir,
            )
        wandb_run_id = wandb.run.id

    return SummaryWriter(log_dir=tb_log_dir), wandb_run_id


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    epoch,
    train_loader,
    train_sampler,
    tokenizer,
    denoiser,
    diffuser,
    optim,
    scheduler,
    tb_writer,
    cfg,
    rank,
    device,
    global_update,
    log_dir,
    wandb_run_id,
):
    denoiser.train()
    train_sampler.set_epoch(epoch)

    epoch_start = time.perf_counter()
    epoch_loss_sum = 0.0
    num_updates = 0
    step_times = []
    data_times = []

    accum_obs_flow = 0.0
    accum_act_flow = 0.0
    accum_obs_boot = 0.0
    accum_act_boot = 0.0
    accum_total = 0.0

    data_start = time.perf_counter()

    for step_idx, batch in enumerate(train_loader):
        micro_idx = step_idx % cfg.train.accum_grad_steps
        is_last_micro = micro_idx == cfg.train.accum_grad_steps - 1

        data_times.append(time.perf_counter() - data_start)

        # --- Prepare batch ---
        images = batch["image"].to(device, non_blocking=True)  # (B, T, C, H, W)
        if not cfg.train.video_pretraining:
            actions = batch["action"].to(device, non_blocking=True)  # (B, T, action_dim)
        else:
            actions = torch.zeros(
                images.shape[0], images.shape[1], cfg.denoiser.n_actions,
                device=images.device, dtype=images.dtype,
            )

        images = images.to(torch.bfloat16)
        # Slice to configured action dims and add token dimension: (B, T, A) -> (B, T, 1, A)
        actions = actions.to(torch.bfloat16)[:, :, :cfg.denoiser.n_actions].unsqueeze(-2)

        torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        if micro_idx == 0:
            optim.zero_grad(set_to_none=True)

        # --- Forward pass ---
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_clean = tokenizer.encode(images).detach().clone()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            diffused_info = diffuser(z_clean, actions)
            obs_flow_loss, act_flow_loss, obs_boot_loss, act_boot_loss = \
                compute_joint_bootstrap_diffusion_loss(diffused_info, denoiser)
            loss_micro = (
                obs_flow_loss + act_flow_loss + obs_boot_loss + act_boot_loss
            ) / cfg.train.accum_grad_steps

        loss_micro.backward()

        accum_obs_flow += obs_flow_loss.item()
        accum_act_flow += act_flow_loss.item()
        accum_obs_boot += obs_boot_loss.item()
        accum_act_boot += act_boot_loss.item()
        accum_total += loss_micro.item()

        # --- Optimizer step at end of accumulation window ---
        if is_last_micro:
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optim.step()
            scheduler.step()
            global_update += 1

            total_tensor = torch.tensor([accum_total], device=device)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.AVG)
            sync_loss = total_tensor.item()
            epoch_loss_sum += sync_loss
            num_updates += 1

            if rank == 0:
                lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar("train/total_loss", sync_loss, global_update)
                tb_writer.add_scalar("train/obs_flow_loss", accum_obs_flow, global_update)
                tb_writer.add_scalar("train/act_flow_loss", accum_act_flow, global_update)
                tb_writer.add_scalar("train/obs_boot_loss", accum_obs_boot, global_update)
                tb_writer.add_scalar("train/act_boot_loss", accum_act_boot, global_update)
                tb_writer.add_scalar("train/lr", lr, global_update)

                if global_update % cfg.print_every == 0:
                    print(
                        f"  [step {global_update}]"
                        f"  loss: {sync_loss:.4f}"
                        f"  obs_flow: {accum_obs_flow:.4f}"
                        f"  act_flow: {accum_act_flow:.4f}"
                        f"  obs_boot: {accum_obs_boot:.4f}"
                        f"  act_boot: {accum_act_boot:.4f}"
                        f"  lr: {lr:.2e}"
                    )

                if global_update % cfg.save_every == 0:
                    print(f"[Checkpoint] Saving at global_update={global_update}")
                    save_ddp_checkpoint(
                        ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                        epoch=epoch,
                        global_update=global_update,
                        model=denoiser,
                        optim=optim,
                        scheduler=scheduler,
                        rank=rank,
                        wandb_run_id=wandb_run_id,
                        log_dir=log_dir,
                    )

            accum_obs_flow = 0.0
            accum_act_flow = 0.0
            accum_obs_boot = 0.0
            accum_act_boot = 0.0
            accum_total = 0.0

        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_start)
        data_start = time.perf_counter()

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = epoch_loss_sum / num_updates if num_updates > 0 else 0.0
    total_frames = (
        cfg.train.batch_per_gpu * cfg.denoiser.max_sequence_length * len(train_loader)
    )
    epoch_fps = total_frames / epoch_time

    if rank == 0:
        avg_step = sum(step_times) / len(step_times)
        avg_data = sum(data_times) / len(data_times)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss:        {avg_loss:.6f}")
        print(f"  Epoch Time:        {epoch_time:.2f}s")
        print(f"  Throughput:        {epoch_fps:.2f} FPS")
        print(f"  Avg Step Time:     {avg_step:.3f}s")
        print(f"  Avg Data Time:     {avg_data:.3f}s")
        print(f"{'='*60}\n")

    return global_update, avg_loss, epoch_time, epoch_fps


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@record
@hydra.main(config_path="config", config_name="dynamics/pushT", version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.enable_fast_matmul

    rank, local_rank, world_size, device = setup_distributed()
    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(f"Distributed training: {world_size} GPU(s)")
        print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
        effective_batch = cfg.train.batch_per_gpu * world_size * cfg.train.accum_grad_steps
        print(f"Effective global batch size: {effective_batch}")

    # --- Data ---
    train_loader, train_sampler = build_dataloader(cfg, rank, world_size)

    # --- Models ---
    if rank == 0:
        print("Building models...")
    tokenizer, denoiser, diffuser = build_models(cfg, device, local_rank)
    if rank == 0:
        n_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        print(f"Denoiser learnable parameters: {n_params:,}")

    # --- Optimizer & scheduler ---
    optim = torch.optim.AdamW(
        denoiser.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    steps_per_epoch = len(train_loader)
    total_steps = cfg.train.num_epochs * steps_per_epoch // cfg.train.accum_grad_steps
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    # --- Checkpoint resume ---
    wandb_run_id = cfg.wandb.run_name
    log_dir = None
    start_epoch = 0
    global_update = 0

    if cfg.reload_checkpoint is not None:
        if rank == 0:
            print(f"Resuming from checkpoint: {cfg.reload_checkpoint}")
        start_epoch, global_update, wandb_run_id, log_dir = load_ddp_checkpoint(
            ckpt_path=cfg.reload_checkpoint,
            model=denoiser,
            optim=optim,
            scheduler=scheduler,
            rank=rank,
        )
    elif rank == 0:
        print("Starting from scratch.")

    # --- Logging (rank 0 only, then broadcast) ---
    tb_writer, wandb_run_id_new = setup_logging(cfg, rank, world_size, log_dir, wandb_run_id)
    if rank == 0:
        log_dir = cfg.output_dir if log_dir is None else log_dir
        wandb_run_id = wandb_run_id_new

    obj_list = [log_dir, wandb_run_id]
    dist.broadcast_object_list(obj_list, src=0)
    log_dir, wandb_run_id = obj_list

    dist.barrier()
    if rank == 0:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    dist.barrier()

    # --- Training loop ---
    epoch_losses, epoch_times, epoch_fps_vals = [], [], []

    if rank == 0:
        print("Starting training...")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        global_update, avg_loss, epoch_time, epoch_fps = train_epoch(
            epoch=epoch,
            train_loader=train_loader,
            train_sampler=train_sampler,
            tokenizer=tokenizer,
            denoiser=denoiser,
            diffuser=diffuser,
            optim=optim,
            scheduler=scheduler,
            tb_writer=tb_writer,
            cfg=cfg,
            rank=rank,
            device=device,
            global_update=global_update,
            log_dir=log_dir,
            wandb_run_id=wandb_run_id,
        )
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        epoch_fps_vals.append(epoch_fps)

        if rank == 0:
            save_ddp_checkpoint(
                ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                epoch=epoch,
                global_update=global_update,
                model=denoiser,
                optim=optim,
                scheduler=scheduler,
                rank=rank,
                wandb_run_id=wandb_run_id,
                log_dir=log_dir,
            )
        dist.barrier()

    # --- Final summary ---
    if rank == 0:
        cur_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"  Avg loss:      {sum(epoch_losses) / len(epoch_losses):.6f}")
        print(f"  Avg epoch time:{sum(epoch_times) / len(epoch_times):.2f}s")
        print(f"  Avg FPS:       {sum(epoch_fps_vals) / len(epoch_fps_vals):.2f}")
        print(f"  GPU memory:    {cur_alloc:.2f} GB current / {peak_alloc:.2f} GB peak")
        print(f"{'='*60}")

        tb_writer.close()
        if cfg.wandb.enable:
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
