#!/bin/bash
#SBATCH --job-name=dreamer_v4_multinode
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1           
#SBATCH --cpus-per-task=16
#SBATCH --mem=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=torch_pr_155_tandon_advanced
#SBATCH --constraint='h200'

mkdir -p logs

# MASTER_ADDR and MASTER_PORT (only on first node)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((15000 + RANDOM % 10000))

# NCCL configuration for InfiniBand
# Use the ibs prefix to allow NCCL to choose from ibs1, ibs2, ibs5, ibs7
export NCCL_SOCKET_IFNAME=ibs                # Use any ibs* interface
export NCCL_IB_DISABLE=0                     # Enable InfiniBand
export NCCL_DEBUG=INFO                       # Verbose logging for debugging
export NCCL_IB_HCA=mlx5                      # Mellanox InfiniBand adapter
export NCCL_NET_GDR_LEVEL=5                  # Enable GPUDirect RDMA

# Optional: Further optimize NCCL for InfiniBand
export NCCL_IB_GID_INDEX=3                   # RoCE v2 if available
export NCCL_IB_TC=106                        # Traffic class for IB
export NCCL_IB_TIMEOUT=22                    # Timeout value
export OMP_NUM_THREADS=4

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Total GPUs: $(($SLURM_JOB_NUM_NODES * 8))"
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Network interface: $NCCL_SOCKET_IFNAME"
echo "=================================================="

module purge
# Run once per node
singularity exec --nv \
    --overlay /scratch/rk4342/dreamer-v4/hpc/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    bash -c "
        source /ext3/env.sh
        conda activate dreamerv4
        python train_tokenizer.py
    "