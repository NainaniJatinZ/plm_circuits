#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=16G  # Requested Memory
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load conda/latest
conda activate phylo_based

# -u ensures output is unbuffered
python3 -u ./summarize_activations.py ../../jatin/plm_circuits/acts 4096 --summarize length_normalized &> out.log
