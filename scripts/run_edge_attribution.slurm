#!/bin/bash
#SBATCH --job-name=edge_attribution
#SBATCH --output=logs/edge_attribution_%j.out
#SBATCH --error=logs/edge_attribution_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=h100-reserved
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo "========================================="
echo "PLM Circuits Edge Attribution Analysis"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "========================================="

# Set up environment
source ~/.bashrc
cd /mnt/polished-lake/home/connor/plm_circuits

# Activate virtual environment
source .venv/bin/activate

# Ensure logs directory exists
mkdir -p logs

# Set cache directories to avoid permission issues
export TORCH_HOME=/mnt/polished-lake/home/connor/plm_circuits/.cache/torch
export TRANSFORMERS_CACHE=/mnt/polished-lake/home/connor/plm_circuits/.cache/transformers
export HF_HOME=/mnt/polished-lake/home/connor/plm_circuits/.cache/huggingface
mkdir -p $TORCH_HOME $TRANSFORMERS_CACHE $HF_HOME /mnt/polished-lake/home/connor/plm_circuits/.cache/models

# Run the analysis
python scripts/edge_attribution_all_layers_jvp.py

echo "========================================="
echo "Analysis completed at $(date)"
echo "========================================="