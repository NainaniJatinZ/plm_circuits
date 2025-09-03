#!/bin/bash
#SBATCH --job-name=activation_dataset
#SBATCH --output=summarize_top_q_%j.out
#SBATCH --error=summarize_top_q_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=h100-reserved
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Activate the virtual environment
source /mnt/polished-lake/home/connor/plm_circuits/.venv/bin/activate

# -u ensures output is unbuffered
python3 -u notebooks/domain_corr/summarize_activations.py /mnt/polished-lake/home/connor/plm_circuits/acts 4096 --summarize top_q &> out.log
