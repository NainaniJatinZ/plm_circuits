#!/bin/bash
#SBATCH --job-name=activation_dataset
#SBATCH --output=property_assoc_%j.out
#SBATCH --error=property_assoc_%j.err
#SBATCH --time=16:00:00
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

# Change to the notebook directory
cd notebooks/domain_corr

# Ensure the directory exists for output
mkdir -p .

# -u ensures output is unbuffered  
python3 -u property_association_quantification.py ../../data/summarized_acts_top_q.pt metadata/ptn_fam_tensor_nonzero.pt --subsetListFile metadata/list_of_desired_latents.pkl --outfile latents_property_top_q &> out2.log

# TEST VERSION - processes only 10 latents and 5 properties
# python3 -u test_property_association.py ../../data/summarized_acts_top_q.pt metadata/ptn_fam_tensor_nonzero.pt --subsetListFile metadata/list_of_desired_latents.pkl --outfile test_latents_property_top_q --test_latents 10 --test_properties 5 &> out2.log  
