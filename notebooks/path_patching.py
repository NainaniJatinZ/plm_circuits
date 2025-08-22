# %%
"""
for neuron1 in l4:
    # get acts of l4 neuron1 when zero ablating or ablating with corrupted seq (prefer corrupted)
    corr_hook

    for neuron2 in l8:
        # record change in acts of l8 neuron2 when zero ablating or ablating with corrupted seq (prefer corrupted)
        cache_hook 

        # put changed acts of l8 neuron2 in corr_hook 
        corr_hook2 

        # record change in recovery between clean l8 and corrupted hook 2 run 
        IE (neuron1, neuron2) = delta m 
"""


#%%
# Import necessary libraries and functions from helper modules
import sys
sys.path.append('../')
sys.path.append('../plm_circuits')

# Import utility functions
from helpers.utils import (
    clear_memory,
    load_esm,
    load_sae_prot,
    mask_flanks_segment,
    patching_metric,
    cleanup_cuda,
    set_seed
)

# Import attribution functions
from attribution import (
    integrated_gradients_sae,
    topk_sae_err_pt
)

# Import hook classes
from hook_manager import SAEHookProt

from data.protein_params import sse_dict, fl_dict, protein_name, protein2pdb
from data.feature_clusters_hypotheses import feature_clusters_MetXA, feature_clusters_Top2

# Additional imports
import json
from functools import partial
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import collections
from typing import Dict, List, Tuple, Optional

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

print(f"Loaded SAEs for layers: {main_layers}")

# Load sequence data and define protein parameters
with open('../data/full_seq_dict.json', "r") as json_file:
    seq_dict = json.load(json_file)

# Choose protein for analysis
set_seed(0)

# %% SET PROTEIN TO INVESTIGATE
protein = "MetXA" # "MetXA" or "Top2"

# pdb reference 
pdb_id = protein2pdb[protein]
seq = seq_dict[pdb_id]
position = sse_dict[pdb_id][0]

print(f"Analyzing protein: {protein} {pdb_id}")
print(f"Sequence length: {len(seq)}")

# Define segment boundaries
ss1_start = position[0] - 5 
ss1_end = position[0] + 5 + 1 
ss2_start = position[1] - 5 
ss2_end = position[1] + 5 + 1 

print(f"Secondary structure element 1: {ss1_start}-{ss1_end}")
print(f"Secondary structure element 1: {ss2_start}-{ss2_end}")
print(f"Clean/optimal Flanking distance: {fl_dict[pdb_id][0]}")
print(f"Corrupted Flanking distance: {fl_dict[pdb_id][1]}")

RESULTS_DIR = '../results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# %%
# Prepare sequences and get baseline contact predictions
full_seq_L = [(1, seq)]
_, _, batch_tokens_BL = batch_converter(full_seq_L)
batch_tokens_BL = batch_tokens_BL.to(device)
batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

with torch.no_grad():
    full_seq_contact_LL = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]

# Prepare clean sequence (with optimal flanks)
clean_fl = fl_dict[pdb_id][0]
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

# Prepare corrupted sequence (with suboptimal flanks)
corr_fl = fl_dict[pdb_id][1]
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

# Create patching metric function
_patching_metric = partial(
    patching_metric,
    orig_contact=full_seq_contact_LL,
    ss1_start=ss1_start,
    ss1_end=ss1_end,
    ss2_start=ss2_start,
    ss2_end=ss2_end,
)

baseline_recovery = _patching_metric(clean_seq_contact_LL)
corrupted_recovery = _patching_metric(corr_seq_contact_LL)

print(f"Baseline contact recovery: {baseline_recovery:.4f}")
print(f"Corrupted contact recovery: {corrupted_recovery:.4f}")

# %%
# Compute causal effects for all layers
print("Computing causal effects using integrated gradients...")

all_effects_sae_ALS = []
all_effects_err_ABLF = []
clean_layer_caches = {}
corr_layer_caches = {}
clean_layer_errors = {}
corr_layer_errors = {}

for layer_idx in main_layers:
    print(f"\nProcessing layer {layer_idx}...")
    
    sae_model = saes[layer_2_saelayer[layer_idx]]

    # Get clean cache and error
    hook = SAEHookProt(sae=sae_model, mask_BL=clean_batch_mask_BL, cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        clean_seq_sae_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
    cleanup_cuda()
    handle.remove()
    clean_cache_LS = sae_model.feature_acts
    clean_err_cache_BLF = sae_model.error_term
    clean_contact_recovery = _patching_metric(clean_seq_sae_contact_LL)

    # Get corrupted cache and error
    hook = SAEHookProt(sae=sae_model, mask_BL=corr_batch_mask_BL, cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        corr_seq_sae_contact_LL = esm_transformer.predict_contacts(corr_batch_tokens_BL, corr_batch_mask_BL)[0]
    cleanup_cuda()
    handle.remove()
    corr_cache_LS = sae_model.feature_acts
    corr_err_cache_BLF = sae_model.error_term
    
    print(f"Layer {layer_idx}: Clean contact recovery: {clean_contact_recovery:.4f}")

    # Run integrated gradients
    effect_sae_LS, effect_err_BLF = integrated_gradients_sae(
        esm_transformer,
        sae_model,
        _patching_metric,
        clean_cache_LS.to(device),
        corr_cache_LS.to(device),
        clean_err_cache_BLF.to(device),
        corr_err_cache_BLF.to(device),
        batch_tokens=clean_batch_tokens_BL,
        batch_mask=clean_batch_mask_BL,
        hook_layer=layer_idx,
    )

    all_effects_sae_ALS.append(effect_sae_LS)
    all_effects_err_ABLF.append(effect_err_BLF)
    clean_layer_caches[layer_idx] = clean_cache_LS
    corr_layer_caches[layer_idx] = corr_cache_LS
    clean_layer_errors[layer_idx] = clean_err_cache_BLF
    corr_layer_errors[layer_idx] = corr_err_cache_BLF

# Stack all effects
all_effects_sae_ALS = torch.stack(all_effects_sae_ALS)
all_effects_err_ABLF = torch.stack(all_effects_err_ABLF)
print(f"\nCausal ranking complete!")
print(f"SAE effects shape: {all_effects_sae_ALS.shape}")
print(f"Error effects shape: {all_effects_err_ABLF.shape}")



# %%
target_recovery_percent = 0.7
latent_dict_path = os.path.join(RESULTS_DIR, 'layer_latent_dicts', f'layer_latent_dict_{protein}_{target_recovery_percent:.2f}.json')

# load the json file 
with open(latent_dict_path, 'r') as f:
    layer_latent_dict = json.load(f)

for layer in layer_latent_dict:
    print(f"Layer {layer}: #latents in {protein}: {len(layer_latent_dict[layer])}")

# %%
def compute_path_patching_effect(up_layer, up_latent, down_layer, down_latent, 
                                saes, layer_2_saelayer, clean_layer_caches, corr_layer_caches, 
                                clean_layer_errors, esm_transformer, clean_batch_tokens_BL, 
                                clean_batch_mask_BL, _patching_metric, device, baseline_score):
    """
    Compute the path patching effect for a single upstream-downstream latent pair.
    
    This implements steps C and D:
    - Step C: Ablate upstream latent and record change in downstream latent
    - Step D: Set changed downstream activation and measure final metric change
    
    Args:
        up_layer, down_layer: Layer indices
        up_latent, down_latent: Latent indices  
        ... (other arguments are the shared data structures)
        baseline_score: The clean baseline score for comparison
        
    Returns:
        metric_change: Change in metric score (step D result - baseline)
    """
    
    try:
        # Get SAE models and set mean errors
        up_sae_model = saes[layer_2_saelayer[up_layer]]
        down_sae_model = saes[layer_2_saelayer[down_layer]]
        up_sae_model.mean_error = clean_layer_errors[up_layer]
        down_sae_model.mean_error = clean_layer_errors[down_layer]
        
        # Step C: Ablate upstream and record downstream change
        
        # Setup corruption hook for upstream layer
        corrupt_hook = SAEHookProt(
            sae=up_sae_model,
            mask_BL=clean_batch_mask_BL,
            patch_latent_S=up_latent,
            patch_value=corr_layer_caches[up_layer][:, up_latent].to(device),
            use_mean_error=True,
        )
        
        # Setup caching hook for downstream layer
        cache_hook = SAEHookProt(
            sae=down_sae_model,
            mask_BL=clean_batch_mask_BL,
            cache_latents=True,
            layer_is_lm=False,
            calc_error=False,
            use_error=False,
            use_mean_error=True
        )
        
        # Register both hooks
        corrupt_handle = esm_transformer.esm.encoder.layer[up_layer].register_forward_hook(corrupt_hook)
        cache_handle = esm_transformer.esm.encoder.layer[down_layer].register_forward_hook(cache_hook)
        
        # Forward pass with corruption
        with torch.no_grad():
            _ = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
        
        # Get baseline and corrupted activations for downstream latent
        # baseline_acts_down = clean_layer_caches[down_layer][:, down_latent].clone()
        corrupted_acts_down = down_sae_model.feature_acts[:, down_latent].clone()
        
        # Clean up hooks from step C
        corrupt_handle.remove()
        cache_handle.remove()
        cleanup_cuda()
        
        # Step D: Set changed downstream activation and measure metric
        
        corrupt_hook2 = SAEHookProt(
            sae=down_sae_model,
            mask_BL=clean_batch_mask_BL,
            patch_latent_S=down_latent,
            patch_value=corrupted_acts_down.to(device),
            use_mean_error=True,
        )
        
        # Register hook
        corrupt_handle2 = esm_transformer.esm.encoder.layer[down_layer].register_forward_hook(corrupt_hook2)
        
        # Forward pass with downstream corruption
        with torch.no_grad():
            final_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
        
        # Clean up hook from step D
        corrupt_handle2.remove()
        cleanup_cuda()
        
        # Calculate final metric change
        final_score = _patching_metric(final_contact_LL)
        metric_change = final_score - baseline_score
        
        # Convert to CPU float if tensor
        if isinstance(metric_change, torch.Tensor):
            metric_change = metric_change.cpu().item()
            
        return metric_change
        
    except Exception as e:
        print(f"Error processing up_layer={up_layer}, up_latent={up_latent}, down_layer={down_layer}, down_latent={down_latent}: {e}")
        return None


# %%

def compute_all_path_patching_effects(layer_latent_dict, saes, layer_2_saelayer, clean_layer_caches, 
                                     corr_layer_caches, clean_layer_errors, esm_transformer, 
                                     clean_batch_tokens_BL, clean_batch_mask_BL, _patching_metric, 
                                     device, baseline_score):
    """
    Compute path patching effects for all valid layer-latent combinations.
    
    Args:
        layer_latent_dict: Dictionary with layer numbers as keys and lists of latent indices as values
        ... (other arguments are the shared data structures)
        
    Returns:
        results: List of tuples (up_layer, up_latent, down_layer, down_latent, metric_change)
                sorted by metric_change (most negative = strongest effect)
    """
    
    results = []
    total_combinations = 0
    processed_combinations = 0
    
    # Convert layer keys to integers and sort
    layers = sorted([int(layer) for layer in layer_latent_dict.keys()])
    
    # Count total combinations for progress tracking
    for i, up_layer in enumerate(layers):
        for j, down_layer in enumerate(layers):
            if up_layer < down_layer:  # Only test causal direction
                up_latents = layer_latent_dict[str(up_layer)]
                down_latents = layer_latent_dict[str(down_layer)]
                total_combinations += len(up_latents) * len(down_latents)
    
    print(f"Total combinations to test: {total_combinations}")
    
    # Test all layer pairs in causal order
    for i, up_layer in enumerate(layers):
        for j, down_layer in enumerate(layers):
            if up_layer >= down_layer:  # Skip non-causal directions
                continue
                
            print(f"\nTesting layer {up_layer} → layer {down_layer}")
            
            up_latents = layer_latent_dict[str(up_layer)]
            down_latents = layer_latent_dict[str(down_layer)]
            
            # Test all latent combinations for this layer pair
            for up_latent in up_latents:
                for down_latent in down_latents:
                    
                    metric_change = compute_path_patching_effect(
                        up_layer, up_latent, down_layer, down_latent,
                        saes, layer_2_saelayer, clean_layer_caches, corr_layer_caches,
                        clean_layer_errors, esm_transformer, clean_batch_tokens_BL,
                        clean_batch_mask_BL, _patching_metric, device, baseline_score
                    )
                    
                    if metric_change is not None:
                        results.append((up_layer, up_latent, down_layer, down_latent, metric_change))
                    
                    processed_combinations += 1
                    if processed_combinations % 100 == 0:
                        print(f"Progress: {processed_combinations}/{total_combinations} ({100*processed_combinations/total_combinations:.1f}%)")
    
    # Sort by metric change (most negative = strongest effect)
    results.sort(key=lambda x: x[4])
    
    print(f"\nCompleted! Processed {len(results)} valid combinations out of {total_combinations} total.")
    
    return results


# %%

# Get baseline score for comparison
with torch.no_grad():
    baseline_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
baseline_score = _patching_metric(baseline_contact_LL)
print(f"\nBaseline score: {baseline_score:.4f}")

# %%

# Filter the dictionary to only keep the three layers manually analyzed and deduplicate latents
n_layers = {"2B61A": [0, 1, 2], "1PVGA": [0, 2, 3]}

filtered_layers = [sorted(layer_latent_dict.keys(), key=int)[i] for i in n_layers[protein2pdb[protein]]]
layer_latent_dict_filtered = {}

print("\nFiltered and deduplicated layer-latent combinations:")
for layer in filtered_layers:
    # Deduplicate latents while preserving order
    unique_latents = list(dict.fromkeys(layer_latent_dict[layer]))
    layer_latent_dict_filtered[layer] = unique_latents
    duplicates_removed = len(layer_latent_dict[layer]) - len(unique_latents)
    print(f"Layer {layer}: {len(unique_latents)} unique latents (removed {duplicates_removed} duplicates)")

# Also check the full dataset for duplicates
print("\nDuplicate analysis across all layers:")
for layer in sorted(layer_latent_dict.keys(), key=int):
    original_count = len(layer_latent_dict[layer])
    unique_count = len(set(layer_latent_dict[layer]))
    duplicates = original_count - unique_count
    if duplicates > 0:
        print(f"Layer {layer}: {unique_count} unique latents, {duplicates} duplicates")

# %%
recompute_path_patching = True
if recompute_path_patching:

    # Run the path patching analysis
    print("Starting path patching analysis...")

    path_patching_results = compute_all_path_patching_effects(
        layer_latent_dict_filtered, saes, layer_2_saelayer, clean_layer_caches, 
        corr_layer_caches, clean_layer_errors, esm_transformer, 
        clean_batch_tokens_BL, clean_batch_mask_BL, _patching_metric, 
        device, baseline_score
    )

# %%
os.makedirs(os.path.join(RESULTS_DIR, 'path_patching_results'), exist_ok=True)
if recompute_path_patching:
    # Save results for further analysis
    results_data = {
        'baseline_score': float(baseline_score),
        'path_patching_results': [
            {
                'up_layer': int(up_layer),
                'up_latent': int(up_latent), 
                'down_layer': int(down_layer),
                'down_latent': int(down_latent),
                'metric_change': float(metric_change)
            }
            for up_layer, up_latent, down_layer, down_latent, metric_change in path_patching_results
        ]
    }

    with open(os.path.join(RESULTS_DIR, 'path_patching_results', f'path_patching_results_{protein}_{target_recovery_percent:.2f}.json'), 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {os.path.join(RESULTS_DIR, 'path_patching_results', f'path_patching_results_{protein}_{target_recovery_percent:.2f}.json')}")
    print(f"Total edges in results: {len(path_patching_results)}")

# load back regardless 
with open(os.path.join(RESULTS_DIR, 'path_patching_results', f'path_patching_results_{protein}_{target_recovery_percent:.2f}.json'), 'r') as f:
    results_data = json.load(f)

# Convert back to tuple format (up_layer, up_latent, down_layer, down_latent, metric_change)
path_patching_results = [
    (result['up_layer'], result['up_latent'], result['down_layer'], result['down_latent'], result['metric_change'])
    for result in results_data['path_patching_results']
]

print(f"Loaded {len(path_patching_results)} path patching results")
baseline_score = results_data['baseline_score']
print(f"Baseline score: {baseline_score:.6f}")

# %%

# def plot_edge_strength_distribution(path_patching_results, top_n=None, figsize=(12, 8)):
#     """
#     Plot the distribution of absolute edge strengths to identify natural thresholds.
    
#     Args:
#         path_patching_results: List of path patching results
#         top_n: Number of top edges to plot (None = all edges)
#         figsize: Figure size tuple
#     """
    
#     # Get absolute strengths and sort
#     abs_strengths = [abs(result[4]) for result in path_patching_results]
#     abs_strengths.sort(reverse=True)
    
#     if top_n is not None:
#         abs_strengths = abs_strengths[:top_n]
    
#     ranks = list(range(1, len(abs_strengths) + 1))
    
#     # Create the plot
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
#     # Main plot: Rank vs Absolute Strength
#     ax1.plot(ranks, abs_strengths, 'b-', linewidth=1.5, alpha=0.8)
#     ax1.scatter(ranks[:20], abs_strengths[:20], color='red', s=30, alpha=0.7, zorder=5, label='Top 20')
#     ax1.set_xlabel('Edge Rank')
#     ax1.set_ylabel('Absolute Metric Change')
#     ax1.set_title('Edge Strength Distribution (Sorted by Absolute Effect)')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
    
#     # Add percentile markers
#     percentiles = [0.1, 0.5, 1, 2, 5, 10]
#     for p in percentiles:
#         idx = int(len(abs_strengths) * p / 100) - 1
#         if 0 <= idx < len(abs_strengths):
#             ax1.axvline(x=idx+1, color='gray', linestyle='--', alpha=0.5)
#             ax1.text(idx+1, ax1.get_ylim()[1]*0.9, f'{p}%', rotation=90, 
#                     verticalalignment='top', fontsize=8)
    
#     # Zoom in on top edges
#     zoom_n = min(100, len(abs_strengths))
#     ax2.plot(ranks[:zoom_n], abs_strengths[:zoom_n], 'b-', linewidth=2)
#     ax2.scatter(ranks[:20], abs_strengths[:20], color='red', s=40, alpha=0.8, zorder=5)
#     ax2.set_xlabel('Edge Rank')
#     ax2.set_ylabel('Absolute Metric Change')
#     ax2.set_title(f'Top {zoom_n} Edges (Zoomed View)')
#     ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print some statistics
#     print(f"\nEdge Strength Distribution Statistics:")
#     print(f"Total edges: {len(path_patching_results)}")
#     print(f"Strongest edge: {max(abs_strengths):.6f}")
#     print(f"Median edge strength: {np.median(abs_strengths):.6f}")
#     print(f"Mean edge strength: {np.mean(abs_strengths):.6f}")
    
#     print(f"\nPercentile thresholds:")
#     for p in [0.1, 0.5, 1, 2, 5, 10]:
#         idx = int(len(abs_strengths) * p / 100) - 1
#         if 0 <= idx < len(abs_strengths):
#             threshold = abs_strengths[idx]
#             print(f"Top {p:4.1f}%: {threshold:.6f} (rank {idx+1})")
    
#     return abs_strengths, ranks

# %%
def plot_edge_strength_distribution(
    path_patching_results,
    top_n=None,
    figsize=(12, 6),
    save_plot=False,
    results_dir=RESULTS_DIR,
    protein=None,
    target_recovery_percent=None,
    fname_prefix='edge_strength_distribution',
    show=True,
):
    """
    Plot ONLY the main rank vs absolute strength curve and optionally save it.

    Args:
        path_patching_results: List of (up_layer, up_latent, down_layer, down_latent, metric_change)
        top_n: Plot only the top-N strongest edges (by |metric_change|)
        figsize: Figure size
        save_plot: If True, save PNG and PDF like performance_recovery_curves
        results_dir: Base results directory (uses RESULTS_DIR if defined in notebook)
        protein: Optional protein name for filename/title
        target_recovery_percent: Optional float used in filename suffix
        fname_prefix: Filename prefix
        show: Whether to call plt.show() at the end
    Returns:
        abs_strengths, ranks, fig, ax
    """
    import os

    # Compute absolute strengths and ranks
    abs_strengths = [abs(result[4]) for result in path_patching_results]
    abs_strengths.sort(reverse=True)
    if top_n is not None:
        abs_strengths = abs_strengths[:top_n]
    ranks = list(range(1, len(abs_strengths) + 1))

    # Create single plot
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(ranks, abs_strengths, 'b-', linewidth=2, alpha=0.9)
    ax.scatter(ranks[:10], abs_strengths[:10], color='red', s=40, alpha=0.8, zorder=5, label='Top 10')

    title_suffix = f" | Protein: {protein}" if protein is not None else ""
    ax.set_title(f'Edge Strength Distribution (|effect| vs rank){title_suffix}', fontsize=16, pad=10)
    ax.set_xlabel('Edge Rank', fontsize=13)
    ax.set_ylabel('Absolute Metric Change', fontsize=13)

    # Percentile markers
    percentiles = [1, 5, 10, 20, 50]
    for p in percentiles:
        idx = int(len(abs_strengths) * p / 100) - 1
        if 0 <= idx < len(abs_strengths):
            ax.axvline(x=idx + 1, color='gray', linestyle='--', alpha=0.5)
            ax.text(idx + 1, ax.get_ylim()[1] * 0.92, f'{p}%', rotation=90,
                    va='top', ha='right', fontsize=10, color='gray')

    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='best', frameon=True, framealpha=0.9)

    plt.tight_layout()

    # Save like performance_recovery_curves
    if save_plot:
        save_dir = os.path.join(results_dir, 'path_patching_results')
        os.makedirs(save_dir, exist_ok=True)
        suffix = ""
        if protein is not None and target_recovery_percent is not None:
            suffix = f'_{protein}_{target_recovery_percent:.2f}'
        png_path = os.path.join(save_dir, f'{fname_prefix}{suffix}.png')
        pdf_path = os.path.join(save_dir, f'{fname_prefix}{suffix}.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')

    if show:
        plt.show()

    return abs_strengths, ranks, fig, ax
# %%

# Create edge strength distribution plot
print("\n" + "="*80)
print("EDGE STRENGTH DISTRIBUTION ANALYSIS")
print("="*80)

abs_strengths, ranks, fig, ax = plot_edge_strength_distribution(path_patching_results, top_n=None, save_plot=True, protein=protein, target_recovery_percent=target_recovery_percent)
# %%

# %%

def calculate_area_under_curve(abs_strengths, top_percent=10):
    """
    Calculate area under the curve for top X% of edges.
    
    Args:
        abs_strengths: List of absolute strengths (sorted descending)
        top_percent: Percentage of top edges to include
    
    Returns:
        area: Area under the curve for top X% edges
        n_edges: Number of edges included
    """
    
    n_edges = int(len(abs_strengths) * top_percent / 100)
    if n_edges == 0:
        return 0, 0
    
    # Use trapezoidal rule for area calculation
    top_strengths = abs_strengths[:n_edges]
    x_values = list(range(1, len(top_strengths) + 1))
    
    # Calculate area using trapezoidal rule
    area = np.trapz(top_strengths, x_values)
    
    return area, n_edges

def analyze_interpretable_edges(path_patching_results, feature_clusters, top_percent=10):
    """
    Analyze edges between interpretable latents in the top X%.
    
    Args:
        path_patching_results: List of path patching results  
        feature_clusters: Dictionary of interpretable latents
        top_percent: Percentage of top edges to analyze
        
    Returns:
        interpretable_edges: List of edges with interpretable latents
    """
    
    # Sort by absolute strength
    abs_sorted_results = sorted(path_patching_results, key=lambda x: abs(x[4]), reverse=True)
    
    # Get top X%
    n_top = int(len(abs_sorted_results) * top_percent / 100)
    top_edges = abs_sorted_results[:n_top]
    
    print(f"\nAnalyzing top {top_percent}% edges ({n_top} edges)")
    
    # Calculate area under curve
    abs_strengths = [abs(result[4]) for result in abs_sorted_results]
    auc, _ = calculate_area_under_curve(abs_strengths, top_percent)
    total_auc, _ = calculate_area_under_curve(abs_strengths, 100)
    auc_fraction = auc / total_auc if total_auc > 0 else 0
    
    print(f"Area under curve for top {top_percent}%: {auc:.4f}")
    print(f"Fraction of total AUC captured: {auc_fraction:.3f} ({auc_fraction*100:.1f}%)")
    
    # Find interpretable edges
    interpretable_edges = []
    
    for rank, (up_layer, up_latent, down_layer, down_latent, metric_change) in enumerate(top_edges, 1):
        # Check if both upstream and downstream latents are in feature clusters
        up_interpretable = (up_layer in feature_clusters and 
                           up_latent in feature_clusters[up_layer])
        down_interpretable = (down_layer in feature_clusters and 
                             down_latent in feature_clusters[down_layer])
        
        if up_interpretable and down_interpretable:
            up_feature = feature_clusters[up_layer][up_latent]
            down_feature = feature_clusters[down_layer][down_latent]
            
            interpretable_edges.append({
                'rank': rank,
                'up_layer': up_layer,
                'up_latent': up_latent,
                'up_feature': up_feature,
                'down_layer': down_layer,  
                'down_latent': down_latent,
                'down_feature': down_feature,
                'metric_change': metric_change,
                'abs_change': abs(metric_change)
            })
    
    return interpretable_edges

def print_interpretable_edges(interpretable_edges):
    """Print formatted table of interpretable edges"""
    
    if not interpretable_edges:
        print("\nNo edges found with both upstream and downstream latents in feature clusters.")
        return
    
    print(f"\n{'='*100}")
    print(f"INTERPRETABLE EDGES (Both upstream and downstream latents are interpretable)")
    print(f"{'='*100}")
    
    print(f"{'Rank':<6} {'Up Layer':<8} {'Up Latent':<10} {'Up Feature':<15} {'Down Layer':<10} {'Down Latent':<12} {'Down Feature':<15} {'Metric Change':<13}")
    print(f"{'-'*100}")
    
    for edge in interpretable_edges:
        print(f"{edge['rank']:<6} {edge['up_layer']:<8} {edge['up_latent']:<10} {edge['up_feature']:<15} "
              f"{edge['down_layer']:<10} {edge['down_latent']:<12} {edge['down_feature']:<15} {edge['metric_change']:<13.6f}")
    
    print(f"\nFound {len(interpretable_edges)} interpretable edges in the top 10%")
    
    # Group by feature pairs
    feature_pairs = {}
    for edge in interpretable_edges:
        pair = (edge['up_feature'], edge['down_feature'])
        if pair not in feature_pairs:
            feature_pairs[pair] = []
        feature_pairs[pair].append(edge)
    
    print(f"\nFeature pair summary:")
    for (up_feat, down_feat), edges in feature_pairs.items():
        print(f"{up_feat} → {down_feat}: {len(edges)} edges")
        
def write_interpretable_edges_markdown(
    interpretable_edges,
    save_report=True,
    results_dir=RESULTS_DIR,
    subfolder='path_patching_results',
    protein=None,
    target_recovery_percent=None,
    fname_prefix='interpretable_edges',
):
    """
    Write interpretable edges to a Markdown file (instead of printing).

    Args:
        interpretable_edges: List of dicts with keys:
            'rank', 'up_layer', 'up_latent', 'up_feature',
            'down_layer', 'down_latent', 'down_feature', 'metric_change'
        save_report: If True, save the Markdown to disk
        results_dir: Base results directory
        subfolder: Subdirectory under results_dir to store the report
        protein: Optional protein name for filename/title
        target_recovery_percent: Optional float used in filename suffix
        fname_prefix: Filename prefix (without extension)

    Returns:
        markdown_text, saved_path (saved_path is None if save_report=False)
    """
    import os

    if not interpretable_edges:
        markdown_text = "# Interpretable Edges Report\n\n_No interpretable edges found._\n"
        saved_path = None
        if save_report:
            save_dir = os.path.join(results_dir, subfolder)
            os.makedirs(save_dir, exist_ok=True)
            suffix = ""
            if protein is not None and target_recovery_percent is not None:
                suffix = f"_{protein}_{target_recovery_percent:.2f}"
            saved_path = os.path.join(save_dir, f"{fname_prefix}{suffix}.md")
            with open(saved_path, 'w') as f:
                f.write(markdown_text)
        return markdown_text, saved_path

    # Header
    title_suffix = f" — Protein: {protein}" if protein is not None else ""
    markdown_lines = [
        f"# Interpretable Edges Report{title_suffix}",
        "",
        f"_Total interpretable edges: {len(interpretable_edges)}_",
        "",
        "## Edges",
        "",
        "| Rank | Up Layer | Up Latent | Up Feature | Down Layer | Down Latent | Down Feature | Metric Change | Abs Change |",
        "|:----:|:--------:|:---------:|:-----------|:----------:|:-----------:|:------------|--------------:|---------:|",
    ]

    for edge in interpretable_edges:
        rank = edge.get('rank', '')
        up_layer = edge.get('up_layer', '')
        up_latent = edge.get('up_latent', '')
        up_feature = edge.get('up_feature', '')
        down_layer = edge.get('down_layer', '')
        down_latent = edge.get('down_latent', '')
        down_feature = edge.get('down_feature', '')
        metric_change = edge.get('metric_change', 0.0)
        abs_change = abs(metric_change) if metric_change is not None else ''
        markdown_lines.append(
            f"| {rank} | {up_layer} | {up_latent} | {up_feature} | {down_layer} | {down_latent} | {down_feature} | {metric_change:.6f} | {abs_change:.6f} |"
        )

    # Feature pair summary
    feature_pairs = {}
    for edge in interpretable_edges:
        pair = (edge.get('up_feature', ''), edge.get('down_feature', ''))
        feature_pairs.setdefault(pair, 0)
        feature_pairs[pair] += 1

    markdown_lines += [
        "",
        "## Feature Pair Summary",
        "",
        "| Up Feature | Down Feature | Count |",
        "|:-----------|:-------------|------:|",
    ]
    for (up_feat, down_feat), count in feature_pairs.items():
        markdown_lines.append(f"| {up_feat} | {down_feat} | {count} |")

    markdown_text = "\n".join(markdown_lines)

    saved_path = None
    if save_report:
        save_dir = os.path.join(results_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)
        suffix = ""
        if protein is not None and target_recovery_percent is not None:
            suffix = f"_{protein}_{target_recovery_percent:.2f}"
        saved_path = os.path.join(save_dir, f"{fname_prefix}{suffix}.md")
        with open(saved_path, 'w') as f:
            f.write(markdown_text)

    return markdown_text, saved_path

# %%
feature_cluster = feature_clusters_MetXA if protein == "MetXA" else feature_clusters_Top2
interpretable_nodes = {}
for layer, clusters in feature_cluster.items():
    interpretable_nodes[layer] = {}
    for cluster, nodes in clusters.items():
        interpretable_nodes[layer].update(nodes)


# %%

# Analyze the top % edges and find interpretable ones
print("\n" + "="*80)
print("TOP 10% EDGE ANALYSIS WITH INTERPRETABLE FEATURES")
print("="*80)

interpretable_edges = analyze_interpretable_edges(path_patching_results, interpretable_nodes, top_percent=20 if protein == "MetXA" else 22.9)
write_interpretable_edges_markdown(interpretable_edges, save_report=True, protein=protein, target_recovery_percent=target_recovery_percent)

# %%
