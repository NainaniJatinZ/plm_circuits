#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH --mem=16G  # Requested Memory
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load conda/latest
conda activate phylo_based

# -u ensures output is unbuffered
python3 -u property_association_quantification.py data/summarized_acts_max.pt metadata/ptn_fam_tensor_nonzero.pt --subsetListFile metadata/list_of_desired_latents.pkl --outfile latents_property_max &> out2.log  
