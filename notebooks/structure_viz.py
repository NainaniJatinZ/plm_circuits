# %%

import sys

sys.path.append('../plm_circuits')

# Import utility functions
from helpers.utils import (
    clear_memory,
    load_esm,
    load_sae_prot,
    mask_flanks_segment,
    patching_metric,
    cleanup_cuda
)

import helpers.protein_viz_utils as viz

import torch
import numpy as np
import json
from functools import partial

# %%

# Setup device and load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ESM-2 model
esm_transformer, batch_converter, esm2_alphabet = load_esm(33, device=device)

# Load SAEs for multiple layers
main_layers = [4, 8, 12, 16, 20, 24, 28]
saes = []
for layer in main_layers:
    sae_model = load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=layer, device=device)
    saes.append(sae_model)

layer_2_saelayer = {layer: layer_idx for layer_idx, layer in enumerate(main_layers)}

# %%

# Load sequence data and define protein parameters
with open('../data/full_seq_dict.json', "r") as json_file:
    seq_dict = json.load(json_file)

# Define protein-specific parameters
sse_dict = {"2B61A": [[182, 316]], "1PVGA": [[101, 202]]}
fl_dict = {"2B61A": [44, 43], "1PVGA": [65, 63]}

# Choose protein for analysis
protein = "1PVGA"
seq = seq_dict[protein]
position = sse_dict[protein][0]

# Define segment boundaries
ss1_start = position[0] - 5 
ss1_end = position[0] + 5 + 1 
ss2_start = position[1] - 5 
ss2_end = position[1] + 5 + 1 

print(f"Analyzing protein: {protein}")
print(f"Sequence length: {len(seq)}")
print(f"Segment 1: {ss1_start}-{ss1_end}")
print(f"Segment 2: {ss2_start}-{ss2_end}")

# %%
# Prepare full sequence and get baseline contact predictions
full_seq_L = [(1, seq)]
_, _, batch_tokens_BL = batch_converter(full_seq_L)
batch_tokens_BL = batch_tokens_BL.to(device)
batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

with torch.no_grad():
    full_seq_contact_LL = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]

# Prepare clean sequence (with optimal flanks)
clean_fl = fl_dict[protein][0]
L = len(seq)
left_start = max(0, ss1_start - clean_fl)
left_end = ss1_start
right_start = ss2_end
right_end = min(L, ss2_end + clean_fl)
unmask_left_idxs = list(range(left_start, left_end))
unmask_right_idxs = list(range(right_start, right_end))

clean_seq_L = mask_flanks_segment(seq, ss1_start, ss1_end, ss2_start, ss2_end, unmask_left_idxs, unmask_right_idxs)
_, _, clean_batch_tokens_BL = batch_converter([(1, clean_seq_L)])
clean_batch_tokens_BL = clean_batch_tokens_BL.to(device)
clean_batch_mask_BL = (clean_batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

with torch.no_grad():
    clean_seq_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]

print(f"Clean flank size: {clean_fl}")
print(f"Clean sequence contact recovery: {patching_metric(clean_seq_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end):.4f}")

# Prepare corrupted sequence (with suboptimal flanks)
corr_fl = fl_dict[protein][1]
left_start = max(0, ss1_start - corr_fl)
left_end = ss1_start
right_start = ss2_end
right_end = min(L, ss2_end + corr_fl)
unmask_left_idxs = list(range(left_start, left_end))
unmask_right_idxs = list(range(right_start, right_end))

corr_seq_L = mask_flanks_segment(seq, ss1_start, ss1_end, ss2_start, ss2_end, unmask_left_idxs, unmask_right_idxs)
_, _, corr_batch_tokens_BL = batch_converter([(1, corr_seq_L)])
corr_batch_tokens_BL = corr_batch_tokens_BL.to(device)
corr_batch_mask_BL = (corr_batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

with torch.no_grad():
    corr_seq_contact_LL = esm_transformer.predict_contacts(corr_batch_tokens_BL, corr_batch_mask_BL)[0]

print(f"Corrupted flank size: {corr_fl}")
print(f"Corrupted sequence contact recovery: {patching_metric(corr_seq_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end):.4f}")

# Create patching metric function
_patching_metric = partial(
    patching_metric,
    orig_contact=full_seq_contact_LL,
    ss1_start=ss1_start,
    ss1_end=ss1_end,
    ss2_start=ss2_start,
    ss2_end=ss2_end,
)



# %%
L = len(seq)
seq_list = ["<mask>"] * L

clean_fl = fl_dict[protein][0]
left_start = max(0, ss1_start - clean_fl)
left_end = ss1_start
right_start = ss2_end
right_end = min(L, ss2_end + clean_fl)

# Always unmask patch
seq_list[ss1_start: ss1_end] = list(seq[ss1_start: ss1_end] )
seq_list[ss2_start: ss2_end] = list(seq[ss2_start: ss2_end] )

# unmask flanks
seq_list[left_start: left_end] = list(seq[left_start: left_end])
seq_list[right_start: right_end] = list(seq[right_start: right_end])

# # Unmask the chosen flank positions
# for pos in backward_result[0][0]:
#     seq_list[pos] = seq[pos]
print(seq_list[ss1_start: ss1_end])
print(seq_list[ss2_start: ss2_end])
print(seq_list[left_start: left_end])
print(seq_list[right_start: right_end])

# %%

# Custom colormap function for protein visualization
def custom_colormap_fn(value: float, vmin: float, vmax: float) -> str:
    """
    Custom colormap: maps value in [vmin, vmax] to a color.
    More vibrant, direct color mapping with transparency control.
    Negative values -> #4773b7 (blue)
    Zero -> light gray (will be made transparent)
    Positive values -> #e0912f (orange/amber)
    """
    if vmin == vmax:
        return "#e0912f"  # Default to orange if no range
    
    # More direct mapping for vibrant colors
    if value < 0:
        # Direct blue for negative values
        return "#4773b7"
    elif value > 0:
        # Direct orange for positive values  
        return "#e0912f"
    else:
        # Light gray for zero (will be transparent)
        return "#f0f0f0"

def get_opacity_for_value(value: float) -> float:
    """
    Return opacity based on activation value.
    Zero values get reduced transparency, others stay opaque.
    """
    if value == 0:
        return 0.6  # More transparent for background residues
    else:
        return 1.0  # Fully opaque for important residues (-1 and +1)

# %%
act_list_L = [0]*L 
# Set all flanking positions to -1
for pos in range(left_start, left_end):
    if 0 <= pos < L:
        act_list_L[pos] = -1
for pos in range(right_start, right_end):
    if 0 <= pos < L:
        act_list_L[pos] = -1
for pos in range(ss1_start, ss1_end):
    act_list_L[pos] = 1
for pos in range(ss2_start, ss2_end):
    act_list_L[pos] = 1
act_np_L = np.array(act_list_L)
act_np_L.shape

focus_prot = protein[:-1]
activation_tensor = act_np_L
padd_dict = {"2B61A": 20, "1PVGA": 11}
adapted_tensor = activation_tensor[1+padd_dict[protein]:]
alt_struc = viz.get_single_chain_pdb_structure(focus_prot, 'A')

residue3_1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

residue1_3 = {v: k for k, v in residue3_1.items()}

focus_prot = protein[:-1]
# activation_tensor = act_np_L
adapted_tensor = act_np_L[padd_dict[protein]:]
alt_struc = viz.get_single_chain_pdb_structure(focus_prot, 'A')

target_length = len(list(alt_struc.get_residues()))
padded_activation_tensor = np.pad(adapted_tensor, (0, target_length - len(adapted_tensor)), mode='constant')

view_obj = viz.view_single_protein(
    # uniprot_id=seq_id.split("|")[1],       # load from AlphaFold
    pdb_id=  protein[:-1] , #focus_prot,  # load from PDB
    chain_id="A",              # typically "A"
    values_to_color=padded_activation_tensor,
    colormap_fn=custom_colormap_fn,  # Using custom colormap with specified colors
    opacity_fn=get_opacity_for_value,  # Using custom opacity function
    default_color="white",
    residues_to_highlight=[], #[87, 88, 89, 90, 187, 188, 189, 190],
    highlight_color="lime",
    pymol_params={"width":800, "height":600}
)

# 2) If in a Jupyter/VSCode notebook, show colorbar + 3D view:
from IPython.display import display

display(viz.show_colorbar(custom_colormap_fn,
                        min(padded_activation_tensor),
                        max(padded_activation_tensor),
                        steps=8))
display(view_obj.show())
# %%