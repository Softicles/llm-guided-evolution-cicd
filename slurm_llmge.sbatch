#!/bin/bash
#SBATCH --job-name=llmge_ci
#SBATCH -t 8:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem-per-gpu 16G
#SBATCH --gres=gpu:1
#SBATCH -C "A100-40GB|A100-80GB|H100|V100-16GB|V100-32GB|RTX6000|A40|L40S"
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

echo "==== SLURM JOB START ===="
hostname
date
echo "PWD: $(pwd)"

module load cuda
module load anaconda3

export CUDA_VISIBLE_DEVICES=0
conda activate llm_guided_env

export LD_LIBRARY_PATH=~/.conda/envs/llm_guided_env/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# (Optional but recommended) put caches on scratch to avoid quota
export HF_HOME="$SLURM_SUBMIT_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$SLURM_SUBMIT_DIR/.cache/huggingface/transformers"
export TORCH_HOME="$SLURM_SUBMIT_DIR/.cache/torch"

# Your actual command (edit args as needed)
python run_improved.py first_test

echo "==== SLURM JOB END ===="
date
