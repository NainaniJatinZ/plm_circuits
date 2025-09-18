#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=100GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -t 12:00:00
#SBATCH --constraint=vram40
#SBATCH --output=property_assoc_%j.out
#SBATCH --error=property_assoc_%j.err
#SBATCH -A pi_jensen_umass_edu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Activate the virtual environment
module load conda/latest
conda activate finetuning

# -u ensures output is unbuffered  
python3 -u property_association_quantification.py data/summarized_acts_top_q.pt metadata/ptn_fam_tensor_nonzero.pt --subsetListFile metadata/list_of_desired_latents.pkl --outfile latents_property_og_two_top_q

