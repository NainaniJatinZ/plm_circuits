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
protein = "1PVGA" #"1PVGA" #"2B61A"
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

0.7 * baseline_recovery

# %%
# Find the number of latents needed for each layer to reach 60% of baseline performance
# target_recovery_threshold = 0.65 * baseline_recovery
# print(f"Target recovery threshold (60% of baseline): {target_recovery_threshold:.4f}")
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
# layer = 4
# for target_recovery_percent in [0.6, 0.65, 0.7, 0.75, 0.8]:
#     k, recovery = find_k_for_recovery_threshold(
#             layer, target_recovery_percent, all_effects_sae_ALS,
#             clean_layer_caches, corr_layer_caches, clean_layer_errors, baseline_recovery=baseline_recovery, 
#             r0_percent=0.0 #if (layer != 8 and layer != 4) else 0.6
#         )
#     print(f"Layer {layer} target recovery {target_recovery_percent}: {k} features (recovery: {recovery:.4f})")



# %%
# layer = main_layers[1]
# target_recovery_percent = 0.6
# k, recovery = find_k_for_recovery_threshold(
#         layer, target_recovery_percent, all_effects_sae_ALS,
#         clean_layer_caches, corr_layer_caches, clean_layer_errors, baseline_recovery=baseline_recovery, 
#         r0_percent=0.6
#     )
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
layer_circuit_sizes
# %%

# print the top k features for each layer 
for layer in main_layers:
    print(f"Layer {layer}: {get_top_k_feature_indices(layer, layer_circuit_sizes[layer], all_effects_sae_ALS)}")

# %% record the layer latent dict, only storing the latents, not the tokens
layer_latent_dict = {}
for layer in main_layers:
    layer_latent_dict[layer] = [latent for _, latent in get_top_k_feature_indices(layer, layer_circuit_sizes[layer], all_effects_sae_ALS)]

# %%
layer_latent_dict

# %% save the layer latent dict as a json file
import json
with open('/project/pi_annagreen_umass_edu/jatin/plm_circuits/layer_latent_dict_metx.json', 'w') as f:
    json.dump(layer_latent_dict, f)

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

feature_clusters_1pvg = {
    4: {
        "direct_motif_detectors": {1509:"E", 2511:"X'XQ", 2112:"YXX'", 3069: "GX'", 3544: "C", 2929: "N"},
        "indirect_motif_detectors": {3170: "X'N", 3717:"V", 527: "DX'", 3229: "IXX'", 1297: "I", 1468: "X'XXN", 1196: "D"},
        "motif_detectors": {1509:"E", 2511:"X'XQ", 2112:"YXX'", 3069: "GX'", 3544: "C", 2929: "N", 3170: "X'N", 3717:"V", 527: "DX'", 3229: "IXX'", 1297: "I", 1468: "X'XXN", 1196: "D"}
    },
    8: {
        # "direct_motif_detectors": {},
        "indirect_motif_detectors": {1916: "NX'XXNA"},
        # "motif_detectors": {}, 
        "annotated_domain_detector": {2529:"Hatpase_C", 3159: "Hatpase_C", 3903: "Hatpase_C", 1055: "Hatpase_C", 2066: "Hatpase_C"},
    },
    12: {
        "annotated_domain_detector": {3943: "Hatpase_C", 1796: "Hatpase_C", 1204: "Hatpase_C", 1145:  "Hatpase_C"},
        "misc_domain_detector": {1082: "XPG-I", 2472: "Kinesin"},
        "domain_detectors": {3943: "Hatpase_C", 1796: "Hatpase_C", 1204: "Hatpase_C", 1145:  "Hatpase_C", 1082: "XPG-I", 2472: "Kinesin"},
    },
    16: {
        "annotated_domain_detector": {3077: "Hatpase_C", 1353: "Hatpase_C", 1597: "Hatpase_C", 1814: "Hatpase_C", 3994: "Ribosomal", 1166: "Hatpase_C"},
        # "misc_domain_detector": {},
        # "domain_detectors": {3077: "Hatpase_C", 1353: "Hatpase_C", 1597: "Hatpase_C", 1814: "Hatpase_C", 3994: "Ribosomal", 1166: "Hatpase_C"},
    }
}

print("Feature clusters defined. You can modify the 'feature_clusters' dictionary above to specify your clusters.")
print("\nExample cluster structure:")
for layer, clusters in feature_clusters_1pvg.items(): #feature_clusters.items(): #feature_clusters_1pvg.items():
    print(f"Layer {layer}:")
    for cluster_name, indices in clusters.items():
        print(f"  {cluster_name}: {len(indices)} features")

# %%
# temp_layer_circuit_sizes = {4:29, 8:17, 12:18, 16:25}
for layer in main_layers[:4]: #[:4]:
    latents = get_top_k_feature_indices(layer, layer_circuit_sizes[layer], all_effects_sae_ALS)
    latents = [latent for token_idx, latent in latents]
    # print(f"Layer {layer} latents: {latents}")
    cluster_dict = feature_clusters_1pvg[layer] #feature_clusters[layer] #feature_clusters_1pvg[layer]
    for cluster_name, indices in cluster_dict.items():
        # print(f"  {cluster_name}: {indices}")
        cluster_len = len(indices)
        circuit_len = len([latent for latent in latents if latent in indices.keys()])
        print(f"Layer {layer}  {cluster_name}: {circuit_len}/{cluster_len} features in circuit")
        print(layer_circuit_sizes[layer])

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
# added_layer_circuit_sizes = {layer: layer_circuit_sizes[layer] + 8 for layer in main_layers}
# added_layer_circuit_sizes = temp_layer_circuit_sizes#{4:29, 8:17, 12:18, 16:25}
cluster_analysis_results = analyze_all_feature_clusters(
    feature_clusters_1pvg, layer_circuit_sizes, all_effects_sae_ALS,
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

with open('../results/feature_cluster_analysis_top2_1pvg.pkl', 'wb') as f:
    pickle.dump(results_to_save, f)

print("Results saved to '../results/feature_cluster_analysis_top2_1pvg.pkl'")
print("\nAnalysis complete!")


# %% load the pickle back in 
import pickle

with open('../results/feature_cluster_analysis.pkl', 'rb') as f:
    results_to_load = pickle.load(f)

print(results_to_load)




# %%

import json 

with open('/project/pi_annagreen_umass_edu/jatin/plm_circuits/layer_latent_dict_2b61.json', 'r') as f:
    latent_circuit = json.load(f)
print(latent_circuit)

# %%

def find_layer_for_latent(latent_idx: int, latent_circuit: dict) -> str:
    """
    Find which layer a specific latent index belongs to.
    
    Args:
        latent_idx: The latent index to search for
        latent_circuit: Dictionary with layer keys and latent lists as values
    
    Returns:
        Layer string if found, None if not found
    """
    for layer, latents in latent_circuit.items():
        if latent_idx in latents:
            return layer
    return None

def get_all_latents_with_layers(latent_circuit: dict) -> list:
    """
    Get all latents with their corresponding layers as (layer, latent_idx) tuples.
    
    Args:
        latent_circuit: Dictionary with layer keys and latent lists as values
    
    Returns:
        List of (layer, latent_idx) tuples
    """
    all_latents = []
    for layer, latents in latent_circuit.items():
        for latent_idx in latents:
            all_latents.append((layer, latent_idx))
    return all_latents

def get_latent_by_global_index(global_idx: int, latent_circuit: dict) -> tuple:
    """
    Get layer and latent index by global position in the flattened circuit.
    
    Args:
        global_idx: Position in the flattened list of all latents across layers
        latent_circuit: Dictionary with layer keys and latent lists as values
    
    Returns:
        Tuple of (layer, latent_idx) if valid index, None if out of bounds
    """
    all_latents = get_all_latents_with_layers(latent_circuit)
    if 0 <= global_idx < len(all_latents):
        return all_latents[global_idx]
    return None

def get_layer_latent_counts(latent_circuit: dict) -> dict:
    """
    Get count of latents per layer.
    
    Args:
        latent_circuit: Dictionary with layer keys and latent lists as values
    
    Returns:
        Dictionary with layer as key and count as value
    """
    return {layer: len(latents) for layer, latents in latent_circuit.items()}

# Example usage:
print("Utility functions for latent_circuit analysis:")
print("\n1. Find layer for specific latent:")
example_latent = 2311
layer = find_layer_for_latent(example_latent, latent_circuit)
print(f"Latent {example_latent} belongs to layer: {layer}")

print("\n2. Get all latents with layers (first 10):")
all_latents = get_all_latents_with_layers(latent_circuit)
for i, (layer, latent) in enumerate(all_latents[:10]):
    print(f"Global index {i}: Layer {layer}, Latent {latent}")

print("\n3. Get latent by global index:")
global_idx = 5
result = get_latent_by_global_index(global_idx, latent_circuit)
if result:
    layer, latent = result
    print(f"Global index {global_idx}: Layer {layer}, Latent {latent}")

print("\n4. Latent counts per layer:")
counts = get_layer_latent_counts(latent_circuit)
for layer, count in counts.items():
    print(f"Layer {layer}: {count} latents")

print(f"\nTotal latents across all layers: {len(all_latents)}")
# %%
global_indices = [44, 57, 90, 109, 138, 152, 155]

for global_idx in global_indices:
    layer, latent = get_latent_by_global_index(global_idx, latent_circuit)
    print(f"Global index {global_idx}: Layer {layer}, Latent {latent}")

# %%