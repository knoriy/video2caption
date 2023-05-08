#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=video2caption
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --account laion
#SBATCH --output=%x_%j.out
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --signal=SIGTERM@90

module load openmpi
module load cuda/11.7

export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=1
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

srun /fsx/home-knoriy/miniconda3/envs/hf/bin/python /fsx/knoriy/code/video2caption/src/combine.py --urls "s3://s-laion/documentaries-videos/00000"