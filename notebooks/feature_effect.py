# %%
# # Goal: For each type of feature, we need % of features of that type in the circuit layer and drop in recovery if they are ablated. 
"""
Steps: 
1. load the model and saes 
2. load the sequence data
3. define protein-specific parameters
4. prepare sequences and get baseline contact predictions
5. compute causal effects for all layers
6. find the number of latents needed for each layer to reach 70% of baseline performance
7. analyze all feature clusters
8. plot the results
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
# Find the number of latents needed for each layer to reach 60% of baseline performance
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

def get_top_k_feature_indices(layer: int, k: int, all_effects_sae_ALS: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Get the top-k most important (token, latent) pairs for a given layer.
    
    Args:
        layer: Target layer
        k: Number of top features to get
        all_effects_sae_ALS: Causal effects tensor
    
    Returns:
        List of (token_idx, latent_idx) tuples
    """
    target_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[layer]]
    target_effect_sae_flat_LxS = target_effect_sae_LS.reshape(-1)
    
    # Get top-k indices (largest=False for most negative effects)
    top_rank_vals, top_idx = torch.topk(target_effect_sae_flat_LxS, k=k, largest=False, sorted=True)
    
    # Convert flattened indices back to 2D coordinates
    L, S = target_effect_sae_LS.shape
    row_indices = top_idx // S
    col_indices = top_idx % S
    
    feature_indices = []
    for i in range(len(top_idx)):
        token_idx = row_indices[i].item()
        latent_idx = col_indices[i].item()
        feature_indices.append((token_idx, latent_idx))
    
    return feature_indices

# %%
# %% create layer latent dict or load it from json file
recompute_latent_dict = False # Set to True to force recomputation
latent_dict_path = os.path.join(RESULTS_DIR, 'layer_latent_dicts', f'layer_latent_dict_{protein}_{target_recovery_percent:.2f}.json')

if recompute_latent_dict or not os.path.exists(latent_dict_path):
    print("Computing layer latent dictionary")
    # Compute layer latent dictionary
    layer_latent_dict = {}
    for layer in main_layers:
        layer_latent_dict[layer] = [latent for _, latent in get_top_k_feature_indices(layer, layer_circuit_sizes[layer], all_effects_sae_ALS)]
    
    # Save to file
    os.makedirs(os.path.join(RESULTS_DIR, 'layer_latent_dicts'), exist_ok=True)
    with open(latent_dict_path, 'w') as f:
        json.dump(layer_latent_dict, f)
    
    # load again to get the keys to be strings
    with open(latent_dict_path, 'r') as f:
        layer_latent_dict = json.load(f)
else:
    print("Loading existing latent dict")
    # Load existing dictionary
    with open(latent_dict_path, 'r') as f:
        layer_latent_dict = json.load(f)

# %% LOAD IN Hypothesized Feature Clusters from manual case studies

feature_clusters = feature_clusters_MetXA if protein == "MetXA" else feature_clusters_Top2

print("\nExample cluster structure:")
for layer, clusters in feature_clusters.items():
    print(f"Layer {layer}:")
    for cluster_name, indices in clusters.items():
        print(f"  {cluster_name}: {len(indices)} features")

# %% Check how many features in each cluster are in the circuit

for layer, _ in feature_clusters.items(): 
    latents = get_top_k_feature_indices(layer, layer_circuit_sizes[layer], all_effects_sae_ALS)
    latents = [latent for token_idx, latent in latents]
    cluster_dict = feature_clusters[layer] 
    for cluster_name, indices in cluster_dict.items():
        cluster_len = len(indices)
        circuit_len = len([latent for latent in latents if latent in indices.keys()])
        print(f"Layer {layer}  {cluster_name}: {circuit_len}/{cluster_len} features in circuit")

# %%

def test_feature_cluster_exclusion(layer: int, cluster_dict: Dict[int, str], 
                                  circuit_size: int, all_effects_sae_ALS: torch.Tensor,
                                  clean_layer_caches: Dict, corr_layer_caches: Dict,
                                  clean_layer_errors: Dict) -> Tuple[float, float]:
    """
    Test performance when excluding a specific cluster of features from the circuit.
    
    Args:
        layer: Target layer
        cluster_dict: Dictionary where keys are latent indices and values are descriptions
        circuit_size: Total size of the circuit for this layer
        
    Returns:
        Tuple of (percentage_of_circuit, recovery_with_exclusion)
    """
    
    # Get the full circuit (top circuit_size features)
    circuit_indices = get_top_k_feature_indices(layer, circuit_size, all_effects_sae_ALS)
    
    # Get cluster latent indices
    cluster_latent_indices = set(cluster_dict.keys())
    
    # Count how many circuit features have latents in the cluster
    features_in_cluster = 0
    for token_idx, latent_idx in circuit_indices:
        if latent_idx in cluster_latent_indices:
            features_in_cluster += 1
    
    # Calculate percentage
    percentage = features_in_cluster / len(circuit_indices) if len(circuit_indices) > 0 else 0.0
    
    # Now test the circuit with cluster features excluded
    sae_model = saes[layer_2_saelayer[layer]]
    sae_model.mean_error = clean_layer_errors[layer]
    
    # Get the effect matrix for this layer
    target_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[layer]]
    L, S = target_effect_sae_LS.shape
    
    # Create mask - start with all True (patch), set False for features in circuit EXCEPT cluster features
    sae_mask_LS = torch.ones((L, S), dtype=torch.bool, device=device)
    
    for token_idx, latent_idx in circuit_indices:
        # Only turn off the mask if the latent is NOT in the cluster
        if latent_idx not in cluster_latent_indices:
            sae_mask_LS[token_idx, latent_idx] = False
    
    # Set up hook for patching
    hook = SAEHookProt(
        sae=sae_model,
        mask_BL=clean_batch_mask_BL,
        patch_mask_BLS=sae_mask_LS.to(device),
        patch_value=corr_layer_caches[layer].to(device),
        use_mean_error=True,
    )
    handle = esm_transformer.esm.encoder.layer[layer].register_forward_hook(hook)
    
    # Forward pass & metric
    with torch.no_grad():
        preds_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
    recovery = _patching_metric(preds_LL)
    
    # Clean up
    handle.remove()
    cleanup_cuda()
    
    if isinstance(recovery, torch.Tensor):
        recovery = recovery.cpu().item()
    
    return percentage, recovery

def get_zero_circuit_performance(layer: int, clean_layer_caches: Dict, 
                                corr_layer_caches: Dict, clean_layer_errors: Dict) -> float:
    """
    Get the performance when no circuit features are active (just error node/residual stream).
    
    Args:
        layer: Target layer
        clean_layer_caches, corr_layer_caches, clean_layer_errors: Cache dictionaries
    
    Returns:
        Recovery score with no circuit features active
    """
    sae_model = saes[layer_2_saelayer[layer]]
    sae_model.mean_error = clean_layer_errors[layer]
    
    # Get the effect matrix for this layer
    target_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[layer]]
    L, S = target_effect_sae_LS.shape
    
    # Create mask where ALL features are patched (none are active)
    sae_mask_LS = torch.ones((L, S), dtype=torch.bool, device=device)
    
    # Set up hook for patching
    hook = SAEHookProt(
        sae=sae_model,
        mask_BL=clean_batch_mask_BL,
        patch_mask_BLS=sae_mask_LS.to(device),
        patch_value=corr_layer_caches[layer].to(device),
        use_mean_error=True,
    )
    handle = esm_transformer.esm.encoder.layer[layer].register_forward_hook(hook)
    
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

# %%
def analyze_all_feature_clusters(feature_clusters: Dict, layer_circuit_sizes: Dict,
                               all_effects_sae_ALS: torch.Tensor, clean_layer_caches: Dict,
                               corr_layer_caches: Dict, clean_layer_errors: Dict) -> Dict:
    """
    Analyze all defined feature clusters across all layers.
    
    Returns:
        Dictionary with analysis results for each layer and cluster
    """
    
    def calculate_baseline_performance(layer: int, circuit_size: int) -> float:
        """Calculate baseline performance for a layer using top-k circuit features"""
        sae_model = saes[layer_2_saelayer[layer]]
        sae_model.mean_error = clean_layer_errors[layer]
        
        target_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[layer]]
        target_effect_sae_flat_LxS = target_effect_sae_LS.reshape(-1)

        if circuit_size == 0:
            # Special case: no features active (all patched)
            sae_mask_LS = torch.ones(target_effect_sae_LS.shape, dtype=torch.bool, device=device)
        else:
            # Get top-k indices (largest=False for most negative effects)
            top_rank_vals, top_idx = torch.topk(target_effect_sae_flat_LxS, k=circuit_size, largest=False, sorted=True)
            
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
            patch_value=corr_layer_caches[layer].to(device),
            use_mean_error=True,
        )
        handle = esm_transformer.esm.encoder.layer[layer].register_forward_hook(hook)
        
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
    
    results = {}
    
    for layer, clusters in feature_clusters.items():
        if layer not in layer_circuit_sizes:
            print(f"Warning: Layer {layer} not found in circuit sizes. Skipping.")
            continue
            
        print(f"\n{'='*50}")
        print(f"ANALYZING LAYER {layer}")
        print(f"Circuit size: {layer_circuit_sizes[layer]} features")
        print(f"{'='*50}")
        
        # Get zero-circuit performance for this layer
        print(f"Computing zero-circuit performance for layer {layer}...")
        zero_circuit_performance = get_zero_circuit_performance(layer, clean_layer_caches, corr_layer_caches, clean_layer_errors)
        print(f"Zero-circuit performance: {zero_circuit_performance:.4f}")
        
        layer_results = {}
        circuit_size = layer_circuit_sizes[layer]
        
        # Recalculate baseline performance fresh
        print(f"Recalculating baseline performance for layer {layer}...")
        baseline_performance = calculate_baseline_performance(layer, circuit_size)
        print(f"Baseline performance: {baseline_performance:.4f}")
        
        for cluster_name, cluster_dict in clusters.items():
            print(f"\nTesting {cluster_name} ({len(cluster_dict)} features)...")
            
            try:
                percentage, recovery = test_feature_cluster_exclusion(
                    layer, cluster_dict, circuit_size, all_effects_sae_ALS,
                    clean_layer_caches, corr_layer_caches, clean_layer_errors
                )
                
                # Calculate various performance metrics
                absolute_drop = baseline_performance - recovery
                relative_drop_circuit = (absolute_drop / baseline_performance) * 100 if baseline_performance > 0 else 0
                
                # Relative drop accounting for zero-circuit baseline
                circuit_range = baseline_performance - zero_circuit_performance
                relative_drop_zero_baseline = (absolute_drop / circuit_range) * 100 if circuit_range > 0 else 0
                
                layer_results[cluster_name] = {
                    'cluster_size': len(cluster_dict),
                    'percentage_in_circuit': percentage * 100,  # Convert to percentage
                    'features_in_circuit': int(percentage * circuit_size),
                    'recovery_with_exclusion': recovery,
                    'baseline_recovery': baseline_performance,
                    'zero_circuit_recovery': zero_circuit_performance,
                    'absolute_drop': absolute_drop,
                    'relative_drop_circuit': relative_drop_circuit,
                    'relative_drop_zero_baseline': relative_drop_zero_baseline,
                }
                
                print(f"  Cluster size: {len(cluster_dict)} features")
                print(f"  Percentage in circuit: {percentage*100:.2f}%")
                print(f"  Features in circuit: {int(percentage * circuit_size)}/{circuit_size}")
                print(f"  Recovery with exclusion: {recovery:.4f}")
                print(f"  Absolute drop: {absolute_drop:.4f}")
                print(f"  Relative drop (w.r.t. circuit): {relative_drop_circuit:.2f}%")
                print(f"  Relative drop (w.r.t. zero-circuit): {relative_drop_zero_baseline:.2f}%")
                
            except Exception as e:
                print(f"  Error analyzing {cluster_name}: {str(e)}")
                layer_results[cluster_name] = {"error": str(e)}
        
        results[layer] = layer_results
    
    return results


# %%
# Run the feature cluster analysis
print("Starting feature cluster analysis...")
cluster_analysis_results = analyze_all_feature_clusters(
    feature_clusters, layer_circuit_sizes, all_effects_sae_ALS,
    clean_layer_caches, corr_layer_caches, clean_layer_errors
)

# %%
# Print comprehensive results summary

save_analysis = True

# Create the output string
output_str = f"\n{'='*70}\n"
output_str += "COMPREHENSIVE FEATURE CLUSTER ANALYSIS RESULTS\n"
output_str += f"{'='*70}\n"

for layer, layer_results in cluster_analysis_results.items():
    output_str += f"\nLAYER {layer} (Circuit size: {layer_circuit_sizes[layer]} features)\n"
    output_str += f"Baseline recovery: {layer_circuit_recoveries[layer]:.4f}\n"
    output_str += "-" * 50 + "\n"
    
    if not layer_results:
        output_str += "  No clusters analyzed\n"
        continue
        
    # Sort clusters by absolute performance drop (descending)
    sorted_clusters = sorted(
        [(name, data) for name, data in layer_results.items() if 'error' not in data],
        key=lambda x: x[1]['absolute_drop'],
        reverse=True
    )
    
    for cluster_name, data in sorted_clusters:
        output_str += (f"  {cluster_name:12s}: {data['percentage_in_circuit']:5.1f}% of circuit | "
                      f"Abs Drop: {data['absolute_drop']:6.4f} | "
                      f"Rel Drop (circuit): {data['relative_drop_circuit']:4.1f}% | "
                      f"Rel Drop (zero): {data['relative_drop_zero_baseline']:4.1f}% | "
                      f"Features: {data['features_in_circuit']:3d}/{data['cluster_size']:3d}\n")

# Save to file
os.makedirs(os.path.join(RESULTS_DIR, 'feature_cluster_analysis'), exist_ok=True)
if save_analysis:
    output_path = os.path.join(RESULTS_DIR, 'feature_cluster_analysis', f'feature_cluster_analysis_{protein}_{target_recovery_percent:.2f}.md')
    with open(output_path, 'w') as f:
        f.write(output_str)


# %%

