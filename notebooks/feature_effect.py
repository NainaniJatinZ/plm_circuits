# %%
# # Goal: For each type of feature, we need % of features of that type in the circuit layer and drop in recovery if they are ablated. 
"""
Steps: 
1. load the model and saes 
2. load the sequence data
3. define protein-specific parameters
4. prepare sequences and get baseline contact predictions
5. compute causal effects for all layers
6. find the number of latents needed for each layer to reach 60% of baseline performance
7. analyze all feature clusters
8. plot the results

"""

#%%
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
protein = "2B61A"
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
def find_k_for_recovery_threshold(target_layer: int, target_recovery: float, 
                                all_effects_sae_ALS: torch.Tensor,
                                clean_layer_caches: Dict, corr_layer_caches: Dict,
                                clean_layer_errors: Dict, 
                                max_k: int = 1000, step_size: int = 10) -> Tuple[int, float]:
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
target_recovery_threshold = 0.60 * baseline_recovery
print(f"Target recovery threshold (60% of baseline): {target_recovery_threshold:.4f}")

layer_circuit_sizes = {}
layer_circuit_recoveries = {}

for layer in main_layers:
    k, recovery = find_k_for_recovery_threshold(
        layer, target_recovery_threshold, all_effects_sae_ALS,
        clean_layer_caches, corr_layer_caches, clean_layer_errors
    )
    layer_circuit_sizes[layer] = k
    layer_circuit_recoveries[layer] = recovery

print(f"\nCircuit sizes for 60% baseline recovery:")
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

# %%
# MANUAL FEATURE CLUSTER DEFINITIONS
# You can modify these dictionaries to define your feature clusters for each layer

# Example cluster definitions (these are placeholders - you should replace with your actual clusters), TODO: add IE scores
feature_clusters = {
    # Layer 4 clusters
    4: {
        "direct_motif_detectors": {340: "FX'", 237: "FX'", 3788: "X'XI", 798: "D'XXGN", 1690: "X'XXF"},  # Example: specific (token, latent) pairs
        "indirect_motif_detectors": {2277: "G", 2311: "X'XM", 3634: "X'XXXG", 1682: "PXXXXXX'", 3326: "H"},
        "motif_detectors": {340: "FX'", 237: "FX'", 3788: "X'XI", 798: "D'XXGN", 1690: "X'XXF", 2277: "G", 2311: "X'XM", 3634: "X'XXXG", 1682: "PXXXXXX'", 3326: "H"}, 
    },
    
    # # Layer 8 clusters  
    8: {
        "annotated_domain_detector": {488:"AB_Hydrolase_fold"}, #, 2693: "AB_Hydrolase_fold", 1244:"AB_Hydrolase_fold"},
        "misc_domain_detector": {2677:"FAD/NAD", 2775:"Transketolase", 2166:"DHFR"},
        "motif_detectors": {488:"AB_Hydrolase_fold", 2677:"FAD/NAD", 2775:"Transketolase", 2166:"DHFR"}, #, 2693: "AB_Hydrolase_fold", 1244:"AB_Hydrolase_fold"},
    },
    
    # # Layer 12 clusters
    12: {
        "annotated_domain_detector": {2112: "AB_Hydrolase_fold"},
        "misc_domain_detector": {3536:"SAM_mtases", 1256: "FAM", 2797: "Aldolase", 3794: "SAM_mtases", 3035: "WD40"},
        "motif_detectors": {2112: "AB_Hydrolase_fold", 3536:"SAM_mtases", 1256: "FAM", 2797: "Aldolase", 3794: "SAM_mtases", 3035: "WD40"},
    },
    # Add more layers and clusters as needed...
}

print("Feature clusters defined. You can modify the 'feature_clusters' dictionary above to specify your clusters.")
print("\nExample cluster structure:")
for layer, clusters in feature_clusters.items():
    print(f"Layer {layer}:")
    for cluster_name, indices in clusters.items():
        print(f"  {cluster_name}: {len(indices)} features")

# %%

circuit_indices = get_top_k_feature_indices(12, 39, all_effects_sae_ALS)
print(f"Circuit indices: {circuit_indices}")

# %%


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
        baseline_performance = layer_circuit_recoveries[layer]
        
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

0.0148 / 0.57

# %%
# Print comprehensive results summary
print(f"\n{'='*70}")
print("COMPREHENSIVE FEATURE CLUSTER ANALYSIS RESULTS")
print(f"{'='*70}")

for layer, layer_results in cluster_analysis_results.items():
    print(f"\nLAYER {layer} (Circuit size: {layer_circuit_sizes[layer]} features)")
    print(f"Baseline recovery: {layer_circuit_recoveries[layer]:.4f}")
    print("-" * 50)
    
    if not layer_results:
        print("  No clusters analyzed")
        continue
        
    # Sort clusters by absolute performance drop (descending)
    sorted_clusters = sorted(
        [(name, data) for name, data in layer_results.items() if 'error' not in data],
        key=lambda x: x[1]['absolute_drop'],
        reverse=True
    )
    
    for cluster_name, data in sorted_clusters:
        print(f"  {cluster_name:12s}: {data['percentage_in_circuit']:5.1f}% of circuit | "
              f"Abs Drop: {data['absolute_drop']:6.4f} | "
              f"Rel Drop (circuit): {data['relative_drop_circuit']:4.1f}% | "
              f"Rel Drop (zero): {data['relative_drop_zero_baseline']:4.1f}% | "
              f"Features: {data['features_in_circuit']:3d}/{data['cluster_size']:3d}")

# %%
# Create visualization of results
import matplotlib.pyplot as plt

def plot_cluster_analysis(cluster_analysis_results: Dict, layer_circuit_sizes: Dict):
    """Create visualizations of the cluster analysis results"""
    
    # Prepare data for plotting
    layers = []
    cluster_names = []
    percentages = []
    absolute_drops = []
    relative_drops_circuit = []
    relative_drops_zero = []
    
    for layer, layer_results in cluster_analysis_results.items():
        for cluster_name, data in layer_results.items():
            if 'error' not in data:
                layers.append(layer)
                cluster_names.append(f"L{layer}_{cluster_name}")
                percentages.append(data['percentage_in_circuit'])
                absolute_drops.append(data['absolute_drop'])
                relative_drops_circuit.append(data['relative_drop_circuit'])
                relative_drops_zero.append(data['relative_drop_zero_baseline'])
    
    if not layers:
        print("No data to plot")
        return
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Percentage in circuit
    x_pos = np.arange(len(cluster_names))
    ax1.bar(x_pos, percentages, alpha=0.7)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Percentage in Circuit (%)')
    ax1.set_title('Cluster Representation in Circuits')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(cluster_names, rotation=45, ha='right')
    
    # Plot 2: Absolute performance drop
    ax2.bar(x_pos, absolute_drops, alpha=0.7, color='orange')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Absolute Performance Drop')
    ax2.set_title('Absolute Performance Drop When Cluster Excluded')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cluster_names, rotation=45, ha='right')
    
    # Plot 3: Relative performance drop (w.r.t. circuit)
    ax3.bar(x_pos, relative_drops_circuit, alpha=0.7, color='red')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Relative Performance Drop (%)')
    ax3.set_title('Relative Performance Drop (w.r.t. Circuit Performance)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(cluster_names, rotation=45, ha='right')
    
    # Plot 4: Relative performance drop (w.r.t. zero-circuit)
    ax4.bar(x_pos, relative_drops_zero, alpha=0.7, color='purple')
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Relative Performance Drop (%)')
    ax4.set_title('Relative Performance Drop (w.r.t. Zero-Circuit Baseline)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(cluster_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Create scatter plot: percentage vs performance drop
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(layers))))
    layer_colors = {layer: colors[i] for i, layer in enumerate(set(layers))}
    
    for i, layer in enumerate(layers):
        plt.scatter(percentages[i], absolute_drops[i], 
                   color=layer_colors[layer], alpha=0.7, s=100,
                   label=f'Layer {layer}' if layer not in [layers[j] for j in range(i)] else "")
    
    plt.xlabel('Percentage in Circuit (%)')
    plt.ylabel('Absolute Performance Drop')
    plt.title('Cluster Importance: Percentage vs Performance Impact')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Create visualizations
plot_cluster_analysis(cluster_analysis_results, layer_circuit_sizes)

# %%
# Save results to file
import pickle

results_to_save = {
    'layer_circuit_sizes': layer_circuit_sizes,
    'layer_circuit_recoveries': layer_circuit_recoveries,
    'feature_clusters': feature_clusters,
    'cluster_analysis_results': cluster_analysis_results,
    'baseline_recovery': baseline_recovery,
    'corrupted_recovery': corrupted_recovery,
    'protein': protein,
    'main_layers': main_layers
}

with open('../results/feature_cluster_analysis.pkl', 'wb') as f:
    pickle.dump(results_to_save, f)

print("Results saved to '../results/feature_cluster_analysis.pkl'")
print("\nAnalysis complete!")


# %%
