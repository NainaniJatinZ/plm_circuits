# %%

"""
Steps: 
1. load the model and saes 
2. load the sequence data
3. define protein-specific parameters
4. prepare sequences and get baseline contact predictions
5. compute causal effects for all layers
6. find the number of latents needed for each layer to reach 60% of baseline performance
7. iterate on nice looking plots 
"""

# %%
# Import necessary libraries and functions from helper modules
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

# Import attribution functions
from attribution import (
    integrated_gradients_sae,
    topk_sae_err_pt
)

# Import hook classes
from hook_manager import SAEHookProt

# Additional imports
import json
from functools import partial
import torch
import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

# %%
# Load sequence data and define protein parameters
with open('../data/full_seq_dict.json', "r") as json_file:
    seq_dict = json.load(json_file)

# Define protein-specific parameters
sse_dict = {"2B61A": [[182, 316]], "1PVGA": [[101, 202]]}
fl_dict = {"2B61A": [44, 43], "1PVGA": [65, 63]}

# Choose protein for analysis
protein = "2B61A" #"1PVGA" #"2B61A"
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
# Prepare sequences and get baseline contact predictions
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
def find_k_for_recovery_threshold(target_layer: int, target_recovery_percent: float, 
                                all_effects_sae_ALS: torch.Tensor,
                                clean_layer_caches: Dict, corr_layer_caches: Dict,
                                clean_layer_errors: Dict, 
                                max_k: int = 1000, step_size: int = 10, baseline_recovery: float = 0.0, 
                                r0_percent: float = 0.0) -> Tuple[int, float]:
    """
    Find the number of top-k features needed to reach a target recovery threshold.
    
    Args:
        target_layer: Layer to analyze
        target_recovery: Target recovery score (e.g., 0.6 * baseline_recovery)
        max_k: Maximum number of features to test
        step_size: Step size for binary search
    
    Returns:
        Tuple of (optimal_k, actual_recovery)
    """
    
    def patch_top_k_features_local(k_value: int) -> float:
        """Local function to patch top-k features and return recovery score"""
        sae_model = saes[layer_2_saelayer[target_layer]]
        sae_model.mean_error = clean_layer_errors[target_layer]
        
        target_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[target_layer]]
        target_effect_sae_flat_LxS = target_effect_sae_LS.reshape(-1)

        if k_value == 0:
            # Special case: no features active (all patched)
            sae_mask_LS = torch.ones(target_effect_sae_LS.shape, dtype=torch.bool, device=device)
        else:
            # Get top-k indices (largest=False for most negative effects)
            top_rank_vals, top_idx = torch.topk(target_effect_sae_flat_LxS, k=k_value, largest=False, sorted=True)
            
            # Convert flattened indices back to 2D coordinates
            L, S = target_effect_sae_LS.shape
            row_indices = top_idx // S
            col_indices = top_idx % S
            
            # Create mask - start with all True (patch), set False for positions to not patch
            sae_mask_LS = torch.ones((L, S), dtype=torch.bool, device=device)
            
            for i in range(len(top_idx)):
                row = row_indices[i]
                col = col_indices[i]
                sae_mask_LS[row, col] = False
        
        # Set up hook for patching
        hook = SAEHookProt(
            sae=sae_model,
            mask_BL=clean_batch_mask_BL,
            patch_mask_BLS=sae_mask_LS.to(device),
            patch_value=corr_layer_caches[target_layer].to(device),
            use_mean_error=True,
        )
        handle = esm_transformer.esm.encoder.layer[target_layer].register_forward_hook(hook)
        
        # Forward pass & metric
        with torch.no_grad():
            preds_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
        recovery = _patching_metric(preds_LL)
        
        # Clean up
        handle.remove()
        cleanup_cuda()
        
        if isinstance(recovery, torch.Tensor):
            recovery = recovery.cpu().item()
        
        return recovery
    # First, calculate r0 (performance with k=0)
    print(f"Calculating r0 (k=0 performance) for layer {target_layer}...")
    r0 = patch_top_k_features_local(0)
    print(f"  r0: {r0:.4f}")
    
    # Calculate target recovery threshold
    target_recovery = target_recovery_percent * (baseline_recovery - r0_percent * r0) + r0_percent * r0
    print(f"  Target recovery threshold ({target_recovery_percent*100:.0f}% of improvement): {target_recovery:.4f}")
    
    # Binary search approach
    low_k = 1
    high_k = max_k
    best_k = max_k
    best_recovery = 0.0
    
    print(f"Finding k for layer {target_layer} to reach recovery {target_recovery:.4f}")
    
    # First, check if we can reach the target at all
    max_recovery = patch_top_k_features_local(max_k)
    if max_recovery < target_recovery:
        print(f"  Warning: Cannot reach target recovery {target_recovery:.4f} with {max_k} features. Max recovery: {max_recovery:.4f}")
        return max_k, max_recovery
    
    # Binary search
    while low_k <= high_k:
        mid_k = (low_k + high_k) // 2
        recovery = patch_top_k_features_local(mid_k)
        
        print(f"  k={mid_k}: recovery={recovery:.4f}")
        
        if recovery >= target_recovery:
            best_k = mid_k
            best_recovery = recovery
            high_k = mid_k - 1
        else:
            low_k = mid_k + 1
    
    print(f"  Found optimal k={best_k} with recovery={best_recovery:.4f}")
    return best_k, best_recovery

# %%
# Find the number of latents needed for each layer to reach 70% of baseline performance
target_recovery_percent = 0.7
layer_circuit_sizes = {}
layer_circuit_recoveries = {}

for layer in main_layers:
    k, recovery = find_k_for_recovery_threshold(
        layer, target_recovery_percent, all_effects_sae_ALS,
        clean_layer_caches, corr_layer_caches, clean_layer_errors, baseline_recovery=baseline_recovery, 
        r0_percent=0.0 
    )
    layer_circuit_sizes[layer] = k
    layer_circuit_recoveries[layer] = recovery

print(f"\nCircuit sizes for {target_recovery_percent*100:.0f}% baseline recovery:")
for layer in main_layers:
    print(f"Layer {layer}: {layer_circuit_sizes[layer]} features (recovery: {layer_circuit_recoveries[layer]:.4f})")

# %%
# Create performance recovery curves for all layers
def compute_recovery_curve(target_layer: int, k_values: List[int], 
                          all_effects_sae_ALS: torch.Tensor,
                          clean_layer_caches: Dict, corr_layer_caches: Dict,
                          clean_layer_errors: Dict) -> List[float]:
    """
    Compute recovery scores for a range of k values for a specific layer.
    
    Args:
        target_layer: Layer to analyze
        k_values: List of k values to test
        all_effects_sae_ALS: Tensor of causal effects
        clean_layer_caches: Cache of clean activations
        corr_layer_caches: Cache of corrupted activations  
        clean_layer_errors: Cache of clean error terms
    
    Returns:
        List of recovery scores corresponding to k_values
    """
    
    def patch_top_k_features_local(k_value: int) -> float:
        """Local function to patch top-k features and return recovery score"""
        sae_model = saes[layer_2_saelayer[target_layer]]
        sae_model.mean_error = clean_layer_errors[target_layer]
        
        target_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[target_layer]]
        target_effect_sae_flat_LxS = target_effect_sae_LS.reshape(-1)

        if k_value == 0:
            # Special case: no features active (all patched)
            sae_mask_LS = torch.ones(target_effect_sae_LS.shape, dtype=torch.bool, device=device)
        else:
            # Get top-k indices (largest=False for most negative effects)
            top_rank_vals, top_idx = torch.topk(target_effect_sae_flat_LxS, k=k_value, largest=False, sorted=True)
            
            # Convert flattened indices back to 2D coordinates
            L, S = target_effect_sae_LS.shape
            row_indices = top_idx // S
            col_indices = top_idx % S
            
            # Create mask - start with all True (patch), set False for positions to not patch
            sae_mask_LS = torch.ones((L, S), dtype=torch.bool, device=device)
            
            for i in range(len(top_idx)):
                row = row_indices[i]
                col = col_indices[i]
                sae_mask_LS[row, col] = False
        
        # Set up hook for patching
        hook = SAEHookProt(
            sae=sae_model,
            mask_BL=clean_batch_mask_BL,
            patch_mask_BLS=sae_mask_LS.to(device),
            patch_value=corr_layer_caches[target_layer].to(device),
            use_mean_error=True,
        )
        handle = esm_transformer.esm.encoder.layer[target_layer].register_forward_hook(hook)
        
        # Forward pass & metric
        with torch.no_grad():
            preds_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
        recovery = _patching_metric(preds_LL)
        
        # Clean up
        handle.remove()
        cleanup_cuda()
        
        if isinstance(recovery, torch.Tensor):
            recovery = recovery.cpu().item()
        
        return recovery
    
    recoveries = []
    for k in k_values:
        recovery = patch_top_k_features_local(k)
        recoveries.append(recovery)
        if k % 50 == 0 or k in [1, 2, 5, 10, 20]:  # Print progress for key values
            print(f"  Layer {target_layer}, k={k}: recovery={recovery:.4f}")
    
    return recoveries


# %%

"""
- change k list to be every 10 values
- remove dots from line chart
- use linear scale for x axis
- fix the problem of it starting with some padding ???
"""


# %%
# Configure k values for testing
max_k = 200  # Maximum number of features to test

# Create k values list with every 10 values
k_values = list(range(0, max_k + 1, 5))  # Every 10 values: [0, 10, 20, 30, ..., 200]

print(f"Testing k values: {k_values[:10]}...{k_values[-5:]} (total: {len(k_values)} points)")
print("Computing performance recovery curves for all layers...")
layer_recovery_curves = {}

for layer in main_layers:
    print(f"\nComputing curve for layer {layer}...")
    recoveries = compute_recovery_curve(
        layer, k_values, all_effects_sae_ALS,
        clean_layer_caches, corr_layer_caches, clean_layer_errors
    )
    layer_recovery_curves[layer] = recoveries

# Convert tensor values to CPU for matplotlib compatibility
baseline_recovery_cpu = baseline_recovery.cpu().item() if isinstance(baseline_recovery, torch.Tensor) else baseline_recovery
corrupted_recovery_cpu = corrupted_recovery.cpu().item() if isinstance(corrupted_recovery, torch.Tensor) else corrupted_recovery

# %%
# ---- load latent lists ---------------------------------------------------
latent_json = '/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_circuits/results/layer_latent_dict_metx.json'

with open(latent_json, 'r') as f:
    layer_latent_dict_metx = json.load(f)           # keys are strings

print("Loaded latent dictionary keys:", list(layer_latent_dict_metx.keys()))
print("Example - Layer 4 latent count:", len(layer_latent_dict_metx['4']) if '4' in layer_latent_dict_metx else 'Not found')


#%%

# Create publication-quality plot
plt.style.use('default')  # Clean matplotlib style
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Define better colors using plasma colormap (depth → lightness)
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(main_layers)))

# Calculate target threshold
target_threshold = target_recovery_percent * baseline_recovery_cpu

# Prepare data using loaded latent dictionary
k_exact_list = []

# Plot recovery curves for each layer with exact K from loaded dictionary
for i, layer in enumerate(main_layers):
    y_curve = layer_recovery_curves[layer]
    
    # Get exact K from loaded dictionary
    k_exact = len(layer_latent_dict_metx[str(layer)]) if str(layer) in layer_latent_dict_metx else None
    k_exact_list.append(k_exact)
    
    # Plot the curve
    ax.plot(k_values, y_curve, 
           color=colors[i], linewidth=2.5,
           label=f'Layer {layer} (k={k_exact})', alpha=0.8)
    
    # Add exact K marker if within our k range and we have the data
    if k_exact is not None and k_exact <= max(k_values):
        # Find closest k_value to k_exact
        # closest_k_idx = min(range(len(k_values)), key=lambda i: abs(k_values[i] - k_exact))
        # ax.scatter(k_exact, y_curve[closest_k_idx], 
        #           color=colors[i], s=70, zorder=3, 
        #           edgecolors='white', linewidth=2, marker='o')
        ax.scatter(k_exact, target_threshold, 
                  color=colors[i], s=100, zorder=3, 
                  edgecolors='white', linewidth=2, marker='o')

# Add horizontal lines with improved styling
ax.axhline(y=target_threshold, color='red', linestyle='--', linewidth=3, zorder=1,
          label=f'{target_recovery_percent*100:.0f}% threshold ({target_threshold:.3f})', alpha=0.9)

ax.axhline(y=baseline_recovery_cpu, color='black', linestyle='-.', linewidth=2, zorder=1,
          label=f'Baseline recovery ({baseline_recovery_cpu:.3f})', alpha=0.8)

# Styling for publication quality with protein length context
seq_len = len(seq)
ax.set_xlabel('Number of Top-K Features', fontsize=14, fontweight='bold')
ax.set_ylabel('Contact Recovery Score', fontsize=14, fontweight='bold')
ax.set_title(f'Performance Recovery Curves by Layer\nProtein: {protein} (L={seq_len})', 
            fontsize=16, fontweight='bold', pad=20)

# Set reasonable axis limits
ax.set_xlim(0, max(k_values))
y_min = min([min(curve) for curve in layer_recovery_curves.values()] + [corrupted_recovery_cpu]) - 0.05
y_max = max([max(curve) for curve in layer_recovery_curves.values()] + [baseline_recovery_cpu]) + 0.05
ax.set_ylim(y_min, y_max)

# Grid and legend positioning (back inside plot)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, 
         shadow=True, framealpha=0.9)

# Tick styling
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=10)

# Make it tight
plt.tight_layout()

# Save high-quality version for paper
plt.savefig(f'performance_recovery_curves_{protein}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'performance_recovery_curves_{protein}.pdf', bbox_inches='tight')

plt.show()

# Print summary statistics
print(f"\n{'='*60}")
print(f"PERFORMANCE RECOVERY CURVE ANALYSIS - {protein}")
print(f"{'='*60}")
print(f"Baseline recovery: {baseline_recovery_cpu:.4f}")
print(f"Target threshold ({target_recovery_percent*100:.0f}%): {target_threshold:.4f}")
print(f"Corrupted recovery: {corrupted_recovery_cpu:.4f}")

print(f"\nExact K values from loaded latent dictionary:")
for i, layer in enumerate(main_layers):
    k_exact = k_exact_list[i]
    print(f"  Layer {layer}: {k_exact} features")

print(f"\nK values needed to reach {target_recovery_percent*100:.0f}% threshold (from curves):")
for layer in main_layers:
    # Find first k where we exceed threshold
    curve = layer_recovery_curves[layer]
    k_needed = None
    for i, recovery in enumerate(curve):
        if recovery >= target_threshold:
            k_needed = k_values[i]
            break
    if k_needed is not None:
        print(f"  Layer {layer}: {k_needed} features")
    else:
        print(f"  Layer {layer}: >{max_k} features (threshold not reached)")

# %% more color blind friendly plot 

# Create colorblind-friendly version with distinct styles
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Colorblind-friendly palette with high contrast
colors_cb = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
markers = ['o', 's', '^', 'D', 'v', 'p', '*']

# Plot recovery curves with distinct visual elements
for i, layer in enumerate(main_layers):
    y_curve = layer_recovery_curves[layer]
    k_exact = k_exact_list[i]
    
    # Plot with distinct color, style, and marker
    ax.plot(k_values, y_curve, 
           color=colors_cb[i], linewidth=3, linestyle=line_styles[i],
           marker=markers[i], markersize=3, markevery=8,  # Show marker every 8th point
           label=f'Layer {layer} (k={k_exact})', alpha=0.9)
    
    # Add exact K marker with distinct styling
    if k_exact is not None and k_exact <= max(k_values):
        ax.scatter(k_exact, target_threshold, 
                  color=colors_cb[i], s=120, zorder=5, 
                  edgecolors='black', linewidth=3, marker=markers[i])

# Enhanced horizontal lines
ax.axhline(y=target_threshold, color='red', linestyle='--', linewidth=4, zorder=1,
          label=f'{target_recovery_percent*100:.0f}% threshold ({target_threshold:.3f})', alpha=0.9)

ax.axhline(y=baseline_recovery_cpu, color='black', linestyle='-.', linewidth=3, zorder=1,
          label=f'Baseline recovery ({baseline_recovery_cpu:.3f})', alpha=0.8)

# Enhanced styling
ax.set_xlabel('Number of Top-K Features', fontsize=14, fontweight='bold')
ax.set_ylabel('Contact Recovery Score', fontsize=14, fontweight='bold')
ax.set_title(f'Performance Recovery Curves by Layer (Colorblind-Friendly)\nProtein: {protein} (L={seq_len})', 
            fontsize=16, fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(0, max(k_values))
y_min = min([min(curve) for curve in layer_recovery_curves.values()] + [corrupted_recovery_cpu]) - 0.05
y_max = max([max(curve) for curve in layer_recovery_curves.values()] + [baseline_recovery_cpu]) + 0.05
ax.set_ylim(y_min, y_max)

# Enhanced grid and legend
ax.grid(True, alpha=0.4, linestyle=':', linewidth=1)
ax.legend(loc='lower right', fontsize=9, frameon=True, fancybox=True, 
         shadow=True, framealpha=0.95, ncol=1)

# Tick styling
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig(f'performance_recovery_curves_{protein}_colorblind.png', dpi=300, bbox_inches='tight')
plt.savefig(f'performance_recovery_curves_{protein}_colorblind.pdf', bbox_inches='tight')
plt.show()

# Alternative version with direct line labeling (no legend)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

for i, layer in enumerate(main_layers):
    y_curve = layer_recovery_curves[layer]
    k_exact = k_exact_list[i]
    
    # Plot line
    line = ax.plot(k_values, y_curve, 
                  color=colors_cb[i], linewidth=3, linestyle=line_styles[i],
                  alpha=0.9)[0]
    
    # Direct line labeling - place label at end of line
    ax.annotate(f'L{layer}', 
               xy=(k_values[-1], y_curve[-1]),
               xytext=(5, 0), textcoords='offset points',
               va='center', fontsize=11, fontweight='bold',
               color=colors_cb[i])
    
    # Add exact K marker
    if k_exact is not None and k_exact <= max(k_values):
        ax.scatter(k_exact, target_threshold, 
                  color=colors_cb[i], s=120, zorder=5, 
                  edgecolors='black', linewidth=3, marker=markers[i])
        # Label the exact k value
        ax.annotate(f'k={k_exact}', 
                   xy=(k_exact, target_threshold),
                   xytext=(0, 15), textcoords='offset points',
                   ha='center', fontsize=9, fontweight='bold',
                   color=colors_cb[i],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Threshold lines with labels
ax.axhline(y=target_threshold, color='red', linestyle='--', linewidth=4, zorder=1, alpha=0.9)
ax.annotate(f'70% threshold', xy=(max_k*0.8, target_threshold), 
           xytext=(0, 5), textcoords='offset points',
           ha='center', fontsize=12, fontweight='bold', color='red')

ax.axhline(y=baseline_recovery_cpu, color='black', linestyle='-.', linewidth=3, zorder=1, alpha=0.8)
ax.annotate(f'Baseline', xy=(max_k*0.8, baseline_recovery_cpu), 
           xytext=(0, -15), textcoords='offset points',
           ha='center', fontsize=12, fontweight='bold', color='black')

# Enhanced styling
ax.set_xlabel('Number of Top-K Features', fontsize=14, fontweight='bold')
ax.set_ylabel('Contact Recovery Score', fontsize=14, fontweight='bold')
ax.set_title(f'Performance Recovery Curves by Layer (Direct Labels)\nProtein: {protein} (L={seq_len})', 
            fontsize=16, fontweight='bold', pad=20)

ax.set_xlim(0, max(k_values))
ax.set_ylim(y_min, y_max)
ax.grid(True, alpha=0.4, linestyle=':', linewidth=1)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig(f'performance_recovery_curves_{protein}_labeled.png', dpi=300, bbox_inches='tight')
plt.savefig(f'performance_recovery_curves_{protein}_labeled.pdf', bbox_inches='tight')
plt.show()






# %%

# %% GLOBAL PERFORMANCE RECOVERY CURVE and per layer distribution

# Global analysis using real SAE and error effects from integrated gradients
print(f"Using real effect data - SAE: {all_effects_sae_ALS.shape}, Error: {all_effects_err_ABLF.shape}")

# Import functions needed for global analysis
def topk_sae_err_pt(
    effects_sae_ALS: torch.Tensor,   
    effects_err_ALF: torch.Tensor,   
    k: int = 10,
    mode: str = "abs",              
) -> List[Dict]:
    """Return the k most influential elements among SAE latents and FFN-error sum"""
    
    if mode not in {"abs", "pos", "neg"}:
        raise ValueError(f"mode must be 'abs', 'pos' or 'neg' – got {mode!r}")

    A, L, S = effects_sae_ALS.shape
    
    # Flatten tensors so we can rank them together
    sae_flat  = effects_sae_ALS.reshape(-1)                  
    err_flat  = effects_err_ALF.sum(dim=-1).reshape(-1)      
    combined  = torch.cat([sae_flat, err_flat], dim=0)       

    # Choose ranking criterion
    if mode == "abs":
        ranking_tensor = combined.abs()
        largest_flag   = True
    elif mode == "pos":
        ranking_tensor = combined
        largest_flag   = True            
    else:   # mode == "neg"
        ranking_tensor = combined
        largest_flag   = False           

    # Top-k according to the selected criterion
    top_rank_vals, top_idx = torch.topk(ranking_tensor, k, largest=largest_flag, sorted=True)
    top_vals = combined[top_idx]

    # Decode indices back to coordinates
    sae_len = sae_flat.numel()
    out = []
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        if idx < sae_len:                                   
            layer  = idx // (L * S)
            token  = (idx % (L * S)) // S
            latent = idx % S
            out.append({
                "type": "SAE",
                "value": val,
                "layer_idx": layer,
                "token_idx": token,
                "latent_idx": latent,
            })
        else:                                               
            idx   -= sae_len
            layer  = idx // L
            token  = idx % L
            out.append({
                "type": "ERR",
                "value": val,
                "layer_idx": layer,
                "token_idx": token,
            })
    return out

def topk_performance_global(k, mode, target_recovery=None):
    """Global top-k performance using elements across all layers"""
    if target_recovery is None:
        target_recovery = target_recovery_percent * baseline_recovery_cpu
        
    topk_circuit = topk_sae_err_pt(all_effects_sae_ALS, all_effects_err_ABLF, k=k, mode=mode)
    
    # Create layer masks for all layers
    layer_masks = {}
    
    for layer_idx in range(len(main_layers)):
        L, S = clean_layer_caches[main_layers[layer_idx]].shape
        F = clean_layer_errors[main_layers[layer_idx]].shape[-1]
        sae_m = torch.ones((L, S), dtype=torch.bool, device=device)  # TRUE = patch with corrupted
        err_m = torch.ones((1, L, F), dtype=torch.bool, device=device)
        layer_masks[layer_idx] = {"sae": sae_m, "err": err_m}

    # Set selected positions to FALSE (keep clean)
    for entry in topk_circuit:
        l = entry["layer_idx"]
        t = entry["token_idx"]
        if entry["type"] == "SAE":
            u = entry["latent_idx"]
            layer_masks[l]["sae"][t, u] = False
        else:  # "ERR"
            layer_masks[l]["err"][0, t, :] = False

    # Apply hooks
    handles = []
    for i, layer_idx in enumerate(main_layers):
        sae_model = saes[layer_2_saelayer[layer_idx]]
        
        corr_lat_LS = corr_layer_caches[layer_idx]
        clean_err_LF = clean_layer_errors[layer_idx]
        corr_err_LF = corr_layer_errors[layer_idx]

        m_sae = layer_masks[i]["sae"]
        m_err = layer_masks[i]["err"]

        sae_model.mean_error = m_err * corr_err_LF + (~m_err) * clean_err_LF

        hook = SAEHookProt(
            sae=sae_model,
            mask_BL=clean_batch_mask_BL,
            patch_mask_BLS=m_sae.to(device),
            patch_value=corr_lat_LS.to(device),
            use_mean_error=True,
        )
        handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
        handles.append(handle)

    # Forward pass & metric
    with torch.no_grad():
        preds_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
    recovery = _patching_metric(preds_LL)

    # Clean up
    for handle in handles:
        handle.remove()
    cleanup_cuda()
    
    return recovery.item(), topk_circuit

# Plot 1: Global Performance Recovery Curves (Negative vs Absolute vs Positive)
print("Computing global performance recovery curves...")

modes = ["abs", "pos", "neg"]
mode2label = {"abs": "Absolute", "pos": "Positive", "neg": "Negative"}
k_values_global = list(range(1, 5001, 250))  # Start from 1 to 5000 by 250s

plt.figure(figsize=(10, 6))

for mode in modes:
    print(f"\nComputing {mode} mode...")
    recoveries = []
    for k in k_values_global:
        recovery, _ = topk_performance_global(k, mode)
        recoveries.append(recovery)
        print(f"  k={k}: recovery={recovery:.4f}")
    
    plt.plot(k_values_global, recoveries, marker="o", linewidth=2.5, markersize=4,
            label=mode2label[mode], alpha=0.8)

# Reference lines
plt.axhline(baseline_recovery_cpu, linestyle="--", color="black", linewidth=2, 
           label=f"Baseline ({baseline_recovery_cpu:.3f})", alpha=0.8)
plt.axhline(target_threshold, linestyle=":", color="red", linewidth=2,
           label=f"70% target ({target_threshold:.3f})", alpha=0.8)

plt.xlabel("k (Top-K Elements Preserved)", fontsize=14, fontweight='bold')
plt.ylabel("Contact Recovery Score", fontsize=14, fontweight='bold')
plt.title(f"Global Performance Recovery Curves\nProtein: {protein} (L={seq_len})", 
         fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(f'global_performance_recovery_{protein}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'global_performance_recovery_{protein}.pdf', bbox_inches='tight')
plt.show()

# Plot 2: Distribution of SAE vs Error nodes for 70% performance target
print("\nAnalyzing component distribution for 70% target...")

# Find k value that hits 70% for negative mode (most effective)
target_k = None
for k in k_values_global:
    recovery, circuit = topk_performance_global(k, "neg")
    if recovery >= target_threshold:
        target_k = k
        target_circuit = circuit
        target_recovery = recovery
        break

if target_k is not None:
    print(f"Found k={target_k} gives recovery={target_recovery:.4f} (target: {target_threshold:.4f})")
    
    # Analyze distribution
    sae_per_layer = collections.Counter()
    err_per_layer = collections.Counter()
    
    for entry in target_circuit:
        layer = entry["layer_idx"]
        if entry["type"] == "SAE":
            sae_per_layer[layer] += 1
        else:
            err_per_layer[layer] += 1
    
    layers_sorted = sorted(set(list(sae_per_layer) + list(err_per_layer)))
    actual_layers = [main_layers[l] for l in layers_sorted]  # Convert to actual layer numbers
    sae_counts = [sae_per_layer[l] for l in layers_sorted]
    err_counts = [err_per_layer[l] for l in layers_sorted]
    
    # Create stacked bar plot
    x = np.arange(len(actual_layers))
    width = 0.6
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    p1 = ax.bar(x, sae_counts, width, label="SAE Features", color="#1f77b4", alpha=0.8)
    p2 = ax.bar(x, err_counts, width, bottom=sae_counts, label="Error Terms", color="#ff7f0e", alpha=0.8)
    
    # Add value labels on bars
    for i, (sae_count, err_count) in enumerate(zip(sae_counts, err_counts)):
        total = sae_count + err_count
        if sae_count > 0:
            ax.text(i, sae_count/2, str(sae_count), ha='center', va='center', fontweight='bold')
        if err_count > 0:
            ax.text(i, sae_count + err_count/2, str(err_count), ha='center', va='center', fontweight='bold')
        # Total on top
        ax.text(i, total + 5, str(total), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in actual_layers])
    ax.set_xlabel('Transformer Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Component Count', fontsize=14, fontweight='bold')
    ax.set_title(f'Circuit Component Distribution for 70% Performance\n'
                f'k={target_k} components, Recovery={target_recovery:.3f}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'circuit_distribution_70pct_{protein}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'circuit_distribution_70pct_{protein}.pdf', bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    total_sae = sum(sae_counts)
    total_err = sum(err_counts)
    total_components = total_sae + total_err
    
    print(f"\nCircuit Summary for 70% Performance:")
    print(f"Total components: {total_components}")
    print(f"SAE features: {total_sae} ({total_sae/total_components*100:.1f}%)")
    print(f"Error terms: {total_err} ({total_err/total_components*100:.1f}%)")
    print(f"Recovery achieved: {target_recovery:.4f}")
    
else:
    print("Could not find k value that reaches 70% target within tested range")

# %%

# FLANK LENGTH vs CONTACT RECOVERY ANALYSIS
print("Analyzing contact recovery vs flank length...")

def compute_recovery_for_flank_length(flank_length: int) -> float:
    """
    Compute contact recovery for a given flank length.
    
    Args:
        flank_length: Number of residues to unmask on each side of segments
        
    Returns:
        Contact recovery score
    """
    # Calculate flank boundaries
    left_start = max(0, ss1_start - flank_length)
    left_end = ss1_start
    right_start = ss2_end
    right_end = min(L, ss2_end + flank_length)
    unmask_left_idxs = list(range(left_start, left_end))
    unmask_right_idxs = list(range(right_start, right_end))
    
    # Create sequence with this flank length
    test_seq_L = mask_flanks_segment(seq, ss1_start, ss1_end, ss2_start, ss2_end, 
                                   unmask_left_idxs, unmask_right_idxs)
    _, _, test_batch_tokens_BL = batch_converter([(1, test_seq_L)])
    test_batch_tokens_BL = test_batch_tokens_BL.to(device)
    test_batch_mask_BL = (test_batch_tokens_BL != esm2_alphabet.padding_idx).to(device)
    
    # Get contact predictions
    with torch.no_grad():
        test_contact_LL = esm_transformer.predict_contacts(test_batch_tokens_BL, test_batch_mask_BL)[0]
    
    # Calculate recovery using patching metric
    recovery = _patching_metric(test_contact_LL)
    
    # Convert to float if tensor
    if isinstance(recovery, torch.Tensor):
        recovery = recovery.cpu().item()
        
    return recovery

# Test flank lengths from 0 to 50
flank_lengths = list(range(0, 51, 1))  # Every single value from 0 to 50
recoveries = []

print(f"Testing flank lengths from 0 to 50 for protein {protein}...")
for fl in flank_lengths:
    recovery = compute_recovery_for_flank_length(fl)
    recoveries.append(recovery)
    if fl % 10 == 0 or fl in [1, 2, 5]:  # Print progress for key values
        print(f"  Flank length {fl}: recovery={recovery:.4f}")
# %%
# Create clean publication-quality plot
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(11, 7))

# Plot the recovery curve
ax.plot(flank_lengths, recoveries, 
        color='#2E86AB', linewidth=3, marker='o', markersize=4, 
        alpha=0.9, markerfacecolor='white', markeredgecolor='#2E86AB', markeredgewidth=2)

# Add reference lines
baseline_recovery_value = baseline_recovery.cpu().item() if isinstance(baseline_recovery, torch.Tensor) else baseline_recovery
ax.axhline(y=baseline_recovery_value, color='black', linestyle='--', linewidth=2, 
          label=f'Baseline recovery ({baseline_recovery_value:.3f})', alpha=0.7)

# Find where recovery "kicks in" (steepest increase)
recovery_diffs = np.diff(recoveries)
max_increase_idx = np.argmax(recovery_diffs)

# Get points before and after the kick-in
before_kickin_flank = flank_lengths[max_increase_idx]
before_kickin_recovery = recoveries[max_increase_idx]
after_kickin_flank = flank_lengths[max_increase_idx + 1] 
after_kickin_recovery = recoveries[max_increase_idx + 1]

# Add annotation for recovery before kick-in
ax.annotate(f'Before: {before_kickin_recovery:.3f}', 
           xy=(before_kickin_flank, before_kickin_recovery),
           xytext=(before_kickin_flank - 20, before_kickin_recovery + 0.15),
           arrowprops=dict(arrowstyle='->', color='#d62728', lw=2),
           fontsize=18, fontweight='bold', color='#d62728')

# Add annotation for "Recovery kicks in" 
ax.annotate(f'After: {after_kickin_recovery:.3f}', # 'Recovery kicks in'
           xy=(after_kickin_flank, after_kickin_recovery),
           xytext=(after_kickin_flank - 20, after_kickin_recovery - 0.15),
           arrowprops=dict(arrowstyle='->', color='#17becf', lw=2),
           fontsize=18, fontweight='bold', color='#17becf')

# Styling
ax.set_xlabel('Flank Length', fontsize=14, fontweight='bold')
ax.set_ylabel('Contact Recovery Score', fontsize=14, fontweight='bold')
ax.set_title(f'Contact Recovery vs Flank Length\nProtein: {protein} (L={len(seq)})', 
            fontsize=16, fontweight='bold', pad=20)

# Set axis limits
ax.set_xlim(0, 50)
y_min = min(recoveries) - 0.05
y_max = max(max(recoveries), baseline_recovery_value) + 0.05
ax.set_ylim(y_min, y_max)

# Grid and styling
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
# ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, framealpha=0.9)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()

# Save the plot
plt.savefig(f'flank_length_recovery_{protein}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'flank_length_recovery_{protein}.pdf', bbox_inches='tight')
plt.show()

# # Print summary statistics
# print(f"\n{'='*60}")
# print(f"FLANK LENGTH vs RECOVERY ANALYSIS - {protein}")
# print(f"{'='*60}")
# print(f"Sequence length: {len(seq)}")
# print(f"Baseline recovery (optimal flanks): {baseline_recovery_value:.4f}")
# print(f"Recovery at flank length 0: {recoveries[0]:.4f}")
# print(f"Recovery at flank length 50: {recoveries[-1]:.4f}")
# print(f"Maximum recovery achieved: {max(recoveries):.4f} at flank length {flank_lengths[np.argmax(recoveries)]}")
# print(f"Steepest increase at flank length: {before_kickin_flank} (increase: {recovery_diffs[max_increase_idx]:.4f})")
# print(f"Recovery before kick-in: {before_kickin_recovery:.4f}")
# print(f"Recovery after kick-in: {after_kickin_recovery:.4f}")

# # Find optimal flank length (where we first reach 90% of baseline)
# target_90pct = 0.9 * baseline_recovery_value
# optimal_flank = None
# for i, recovery in enumerate(recoveries):
#     if recovery >= target_90pct:
#         optimal_flank = flank_lengths[i]
#         break

# if optimal_flank is not None:
#     print(f"Flank length for 90% baseline: {optimal_flank}")
# else:
#     print("90% baseline not reached within tested range")

# %%
