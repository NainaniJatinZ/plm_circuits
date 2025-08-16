#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=100GB
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -t 8:00:00
#SBATCH --constraint=vram40
#SBATCH -o logs/slurm_g-%A_%a.out  # %A is the master job ID, %a is the array task ID
#SBATCH -e logs/slurm_g-%A_%a.err
#SBATCH -A pi_jensen_umass_edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load any necessary modules (adjust as needed)
module load conda/latest
conda activate finetuning

# Navigate to the plm_circuits directory and then to notebooks
cd /work/pi_jensen_umass_edu/jnainani_umass_edu/plm_circuits/notebooks
python3 domain_corr.py

echo "Job completed at: $(date)" 
