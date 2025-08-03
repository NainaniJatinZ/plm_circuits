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
    clean_layer_caches[layer_idx] = clean_cache_LS
    corr_layer_caches[layer_idx] = corr_cache_LS
    clean_layer_errors[layer_idx] = clean_err_cache_BLF
    corr_layer_errors[layer_idx] = corr_err_cache_BLF

# Stack all effects
all_effects_sae_ALS = torch.stack(all_effects_sae_ALS)

print(f"\nCausal ranking complete!")
print(f"SAE effects shape: {all_effects_sae_ALS.shape}")

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
JATIN !!! 

change k list to be every 10 values

remove dots from line chart

use linear scale for x axis

fix the problem of it starting with some padding ???


"""


# %%
# Configure k values for testing
max_k = 200  # Maximum number of features to test
step_size = 50  # Regular step size for larger k values

# Create k values list with higher resolution at small k (where most change happens)
k_values = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]  # Fine-grained for small k
# k_values.extend(range(75, max_k + 1, step_size))  # Regular steps for larger k
k_values = sorted(list(set(k_values)))  # Remove duplicates and sort

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

# Create publication-quality plot
plt.style.use('default')  # Clean matplotlib style
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Define colors for each layer (using a colormap for consistency)
colors = plt.cm.viridis(np.linspace(0, 1, len(main_layers)))

# Plot recovery curves for each layer
for i, layer in enumerate(main_layers):
    ax.plot(k_values, layer_recovery_curves[layer], 
           color=colors[i], linewidth=2.5, marker='o', markersize=4,
           label=f'Layer {layer}', alpha=0.8)

# Add horizontal line for target recovery threshold
target_threshold = target_recovery_percent * baseline_recovery_cpu
ax.axhline(y=target_threshold, color='red', linestyle='--', linewidth=2, 
          label=f'{target_recovery_percent*100:.0f}% of baseline ({target_threshold:.3f})', alpha=0.8)

# Add baseline recovery line for reference
ax.axhline(y=baseline_recovery_cpu, color='black', linestyle='-', linewidth=1.5, 
          label=f'Baseline recovery ({baseline_recovery_cpu:.3f})', alpha=0.6)

# Styling for publication quality
ax.set_xlabel('Number of Top-K Features', fontsize=14, fontweight='bold')
ax.set_ylabel('Contact Recovery Score', fontsize=14, fontweight='bold')
ax.set_title(f'Performance Recovery Curves by Layer\nProtein: {protein}', 
            fontsize=16, fontweight='bold', pad=20)

# Set reasonable axis limits
ax.set_xlim(-25, max(k_values) + 50)
y_min = min([min(curve) for curve in layer_recovery_curves.values()] + [corrupted_recovery_cpu]) - 0.05
y_max = max([max(curve) for curve in layer_recovery_curves.values()] + [baseline_recovery_cpu]) + 0.05
ax.set_ylim(y_min, y_max)

# Grid and legend
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=11, 
         frameon=True, fancybox=True, shadow=True)

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
print(f"\nK values needed to reach {target_recovery_percent*100:.0f}% threshold:")
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

# %%
# Create the same plot with logarithmic x-axis scaling
plt.style.use('default')  # Clean matplotlib style
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

# Define colors for each layer (using a colormap for consistency)
colors = plt.cm.viridis(np.linspace(0, 1, len(main_layers)))

# Plot recovery curves for each layer
for i, layer in enumerate(main_layers):
    ax.plot(k_values, layer_recovery_curves[layer], 
           color=colors[i], linewidth=2.5, marker='o', markersize=4,
           label=f'Layer {layer}', alpha=0.8)

# Add horizontal line for target recovery threshold
target_threshold = target_recovery_percent * baseline_recovery_cpu
ax.axhline(y=target_threshold, color='red', linestyle='--', linewidth=2, 
          label=f'{target_recovery_percent*100:.0f}% of baseline ({target_threshold:.3f})', alpha=0.8)

# Add baseline recovery line for reference
ax.axhline(y=baseline_recovery_cpu, color='black', linestyle='-', linewidth=1.5, 
          label=f'Baseline recovery ({baseline_recovery_cpu:.3f})', alpha=0.6)

# Styling for publication quality
ax.set_xlabel('Number of Top-K Features (Log Scale)', fontsize=14, fontweight='bold')
ax.set_ylabel('Contact Recovery Score', fontsize=14, fontweight='bold')
ax.set_title(f'Performance Recovery Curves by Layer (Log Scale)\nProtein: {protein}', 
            fontsize=16, fontweight='bold', pad=20)

# Set logarithmic scaling on x-axis (symlog to handle k=0)
ax.set_xscale('symlog', linthresh=1)  # linthresh=1 means linear below k=1, log above

# Set reasonable axis limits
ax.set_xlim(-0.5, max_k * 1.1)
y_min = min([min(curve) for curve in layer_recovery_curves.values()] + [corrupted_recovery_cpu]) - 0.05
y_max = max([max(curve) for curve in layer_recovery_curves.values()] + [baseline_recovery_cpu]) + 0.05
ax.set_ylim(y_min, y_max)

# Grid and legend
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=11, 
         frameon=True, fancybox=True, shadow=True)

# Tick styling
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=10)

# Custom x-axis ticks for better readability on log scale
ax.set_xticks([0, 1, 2, 5, 10, 20, 50, 100, 200])
ax.set_xticklabels(['0', '1', '2', '5', '10', '20', '50', '100', '200'])

# Make it tight
plt.tight_layout()

# Save high-quality version for paper
plt.savefig(f'performance_recovery_curves_{protein}_logscale.png', dpi=300, bbox_inches='tight')
plt.savefig(f'performance_recovery_curves_{protein}_logscale.pdf', bbox_inches='tight')

plt.show()

print(f"Log-scale plot saved as: performance_recovery_curves_{protein}_logscale.png/pdf")

# %%
