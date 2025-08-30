# %%
"""
Steps: 
1. load in the SSE dict 
2. load model and sae 
3. iterate over sses, with flank range (10 to 70), calc the recovery for each, print protein, flank and recovery if change in recovery is greater than 0.5
4. calcualte the indirect effects and layer wise caches 
5. calc layer wise and global recovery curves 
6. do motif conservation analysis for layer 4 
"""
# %%
# Import necessary libraries and functions from helper modules
import sys
sys.path.append('../')
sys.path.append('../plm_circuits')
# Import utility functions
from plm_circuits.helpers.utils import (
    clear_memory,
    load_esm,
    load_sae_prot,
    mask_flanks_segment,
    patching_metric,
    cleanup_cuda,
    set_seed
)

# Import attribution functions
from plm_circuits.attribution import (
    integrated_gradients_sae,
    topk_sae_err_pt
)

# Import hook classes
from plm_circuits.hook_manager import SAEHookProt

# data
from data.protein_params import sse_dict, fl_dict, protein_name, protein2pdb

# Additional imports
import json
from functools import partial
import os
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

# Load SAEs for multiple layers from Adams et al. 
main_layers = [4, 8, 12, 16, 20, 24, 28]
saes = []
for layer in main_layers:
    sae_model = load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=layer, device=device)
    saes.append(sae_model)

layer_2_saelayer = {layer: layer_idx for layer_idx, layer in enumerate(main_layers)}

print(f"Loaded SAEs for layers: {main_layers}")

# Load sequence data and define protein parameters from Zhang et al. 
with open('../data/full_seq_dict.json', "r") as json_file:
    seq_dict = json.load(json_file)

# Load SSE dict from Zhang et al. 
with open('../data/reproduce_pair_recovery_100_w_bos_eos.json', "r") as json_file:
    ss_dict = json.load(json_file)

set_seed(0)

# %%

# %%
def compute_recovery_for_flank_length(flank_length: int, ss1_start: int, ss1_end: int, ss2_start: int, ss2_end: int, seq: str) -> float:
    """
    Compute contact recovery for a given flank length.
    
    Args:
        flank_length: Number of residues to unmask on each side of segments
        
    Returns:
        Contact recovery score
    """
    L = len(seq)
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


# %%

jump_details_esm_650 = {}

for protein, sse in ss_dict.items():
    print(protein)
    print(sse)
    position = sse[0]
    seq = seq_dict[protein]
    ss1_start = position[0] - 5 
    ss1_end = position[0] + 5 + 1 
    ss2_start = position[1] - 5 
    ss2_end = position[1] + 5 + 1 

    full_seq_L = [(1, seq)]
    _, _, batch_tokens_BL = batch_converter(full_seq_L)
    batch_tokens_BL = batch_tokens_BL.to(device)
    batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

    with torch.no_grad():
        full_seq_contact_LL = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]

    _patching_metric = partial(
        patching_metric,
        orig_contact=full_seq_contact_LL,
        ss1_start=ss1_start,
        ss1_end=ss1_end,
        ss2_start=ss2_start,
        ss2_end=ss2_end,
    )
    recoveries = []
    for flank_length in range(10, 70):
        recovery = compute_recovery_for_flank_length(flank_length, ss1_start, ss1_end, ss2_start, ss2_end, seq)
        recoveries.append(recovery)
        if len(recoveries) > 1 and recoveries[-1] - recoveries[-2] > 0.5:
            print(f"Protein: {protein}\nClean Flank length: {flank_length}, Clean Recovery: {recoveries[-1]}\nCorrupted Flank length: {flank_length - 1}, Corrupted Recovery: {recoveries[-2]}")

            jump_details_esm_650[protein] = {
                "clean_flank_length": flank_length,
                "clean_recovery": recoveries[-1],
                "corrupted_flank_length": flank_length - 1,
                "corrupted_recovery": recoveries[-2]
            }
            break


# %%

with open('../data/jump_details_esm_650.json', 'w') as f:
    json.dump(jump_details_esm_650, f, indent=4)

# %%

protein = '2EK8A'
seq = seq_dict[protein]
ss1_start = ss_dict[protein][0][0] - 5 
ss1_end = ss_dict[protein][0][0] + 5 + 1 
ss2_start = ss_dict[protein][0][1] - 5 
ss2_end = ss_dict[protein][0][1] + 5 + 1 

details = jump_details_esm_650[protein]
clean_flank_length = details['clean_flank_length']
clean_recovery = details['clean_recovery']
corrupted_flank_length = details['corrupted_flank_length']
corrupted_recovery = details['corrupted_recovery']

print(f"Protein: {protein}\nClean Flank length: {clean_flank_length}, Clean Recovery: {clean_recovery}\nCorrupted Flank length: {corrupted_flank_length}, Corrupted Recovery: {corrupted_recovery}")

full_seq_L = [(1, seq)]
_, _, batch_tokens_BL = batch_converter(full_seq_L)
batch_tokens_BL = batch_tokens_BL.to(device)
batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

with torch.no_grad():
    full_seq_contact_LL = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]

_patching_metric = partial(
    patching_metric,
    orig_contact=full_seq_contact_LL,
    ss1_start=ss1_start,
    ss1_end=ss1_end,
    ss2_start=ss2_start,
    ss2_end=ss2_end,
)

# %%

# Prepare clean sequence (with optimal flanks)
clean_fl = clean_flank_length
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
corr_fl = corrupted_flank_length
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

baseline_recovery = _patching_metric(clean_seq_contact_LL)
corrupted_recovery = _patching_metric(corr_seq_contact_LL)

print(f"Baseline contact recovery: {baseline_recovery:.4f}")
print(f"Corrupted contact recovery: {corrupted_recovery:.4f}")

# %%

print(clean_seq_contact_LL[ss1_start+5, ss2_start+4])
print(corr_seq_contact_LL[ss1_start+5, ss2_start+4])

# %%

def patching_metric_single_contact(contact_preds, orig_contact, ss1_start, ss2_start, ss1_idx, ss2_idx):

    seg_cross_contact = contact_preds[ss1_start+ss1_idx, ss2_start+ss2_idx]
    orig_contact_seg = orig_contact[ss1_start+ss1_idx, ss2_start+ss2_idx]
    return seg_cross_contact / orig_contact_seg   #torch.sum(seg_cross_contact * orig_contact_seg) / torch.sum(orig_contact_seg * orig_contact_seg)

_patching_metric_single_contact = partial(
    patching_metric_single_contact,
    orig_contact=full_seq_contact_LL,
    ss1_start=ss1_start,
    ss2_start=ss2_start,
    ss1_idx=5,
    ss2_idx=4,
)

print(_patching_metric_single_contact(clean_seq_contact_LL))
print(_patching_metric_single_contact(corr_seq_contact_LL))

# %%

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
    # clean_contact_recovery_single_contact = _patching_metric_single_contact(clean_seq_sae_contact_LL)

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
from typing import Callable, Dict, Tuple, List
def find_k_for_recovery_threshold(target_layer: int, target_recovery_percent: float, 
                                all_effects_sae_ALS: torch.Tensor,
                                clean_layer_caches: Dict, corr_layer_caches: Dict,
                                clean_layer_errors: Dict, 
                                max_k: int = 1000, step_size: int = 10, baseline_recovery: float = 0.0, 
                                r0_percent: float = 0.0, patching_metric: Callable = _patching_metric) -> Tuple[int, float]:
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
        recovery = patching_metric(preds_LL)
        
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

for layer in main_layers[:1]:
    k, recovery = find_k_for_recovery_threshold(
        layer, target_recovery_percent, all_effects_sae_ALS,
        clean_layer_caches, corr_layer_caches, clean_layer_errors, baseline_recovery=baseline_recovery, 
        r0_percent=0.0, patching_metric=_patching_metric
    )
    layer_circuit_sizes[layer] = k
    layer_circuit_recoveries[layer] = recovery

print(f"\nCircuit sizes for {target_recovery_percent*100:.0f}% baseline recovery:")
for layer in main_layers[:1]:
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

# %% GLOBAL PERFORMANCE RECOVERY CURVE and per layer distribution

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

def topk_performance_global(k, mode, target_recovery=None, patching_metric: Callable = _patching_metric_single_contact):
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
    recovery = patching_metric(preds_LL)

    # Clean up
    for handle in handles:
        handle.remove()
    cleanup_cuda()
    
    return recovery.item(), topk_circuit

# %%
RESULTS_DIR = '../results'
save_global_recovery_curves = True
baseline_recovery = _patching_metric_single_contact(clean_seq_contact_LL)
corrupted_recovery = _patching_metric_single_contact(corr_seq_contact_LL)

baseline_recovery_cpu = baseline_recovery.cpu().item() if isinstance(baseline_recovery, torch.Tensor) else baseline_recovery
corrupted_recovery_cpu = corrupted_recovery.cpu().item() if isinstance(corrupted_recovery, torch.Tensor) else corrupted_recovery
target_threshold = target_recovery_percent * baseline_recovery_cpu
seq_len = len(seq)

# Plot 1: Global Performance Recovery Curves (Negative vs Absolute vs Positive)

modes = ["abs", "pos", "neg"]
mode2label = {"abs": "Absolute", "pos": "Positive", "neg": "Negative"}
k_values_global = list(range(1, 1001, 50))  # Start from 1 to 5000 by 250s

plt.figure(figsize=(10, 6))

for mode in modes:
    print(f"\nComputing {mode} mode...")
    recoveries = []
    for k in k_values_global:
        recovery, _ = topk_performance_global(k, mode, patching_metric=_patching_metric_single_contact)
        recoveries.append(recovery)
        # print(f"  k={k}: recovery={recovery:.4f}")
    
    plt.plot(k_values_global, recoveries, marker="o", linewidth=2.5, markersize=4,
            label=mode2label[mode], alpha=0.8)

# Reference lines
plt.axhline(baseline_recovery_cpu, linestyle="--", color="black", linewidth=2, 
           label=f"Baseline ({baseline_recovery_cpu:.3f})", alpha=0.8)
plt.axhline(target_threshold, linestyle=":", color="red", linewidth=2,
           label=f"{target_recovery_percent*100:.0f}% target ({target_threshold:.3f})", alpha=0.8)

plt.xlabel("k (Top-K Elements Preserved)", fontsize=14, fontweight='bold')
plt.ylabel("Contact Recovery Score", fontsize=14, fontweight='bold')
plt.title(f"Global Performance Recovery Curves\nProtein: {protein} (L={seq_len})", 
         fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

if save_global_recovery_curves:
    os.makedirs(os.path.join(RESULTS_DIR, 'performance_recovery_curves_single_contact'), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, 'performance_recovery_curves_single_contact', f'global_performance_recovery_{protein}_{target_recovery_percent:.2f}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(RESULTS_DIR, 'performance_recovery_curves_single_contact', f'global_performance_recovery_{protein}_{target_recovery_percent:.2f}.pdf'), bbox_inches='tight')
plt.show()

# %%

# Plot 2: Distribution of SAE vs Error nodes for 70% performance target
print(f"\nAnalyzing component distribution for {target_recovery_percent*100:.0f}% target...")

# Find k value that hits target_recovery_percent for negative mode (most effective)
target_k = None
for k in k_values_global:
    recovery, circuit = topk_performance_global(k, "neg", patching_metric=_patching_metric_single_contact)
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
    ax.set_title(f'Circuit Component Distribution for {target_recovery_percent*100:.0f}% Performance\n'
                f'k={target_k} components, Recovery={target_recovery:.3f}, Protein: {protein}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_global_recovery_curves:
        os.makedirs(os.path.join(RESULTS_DIR, 'performance_recovery_curves_single_contact'), exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, 'performance_recovery_curves_single_contact', f'circuit_distribution_{target_recovery_percent:.2f}_{protein}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(RESULTS_DIR, 'performance_recovery_curves_single_contact', f'circuit_distribution_{target_recovery_percent:.2f}_{protein}.pdf'), bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    total_sae = sum(sae_counts)
    total_err = sum(err_counts)
    total_components = total_sae + total_err
    
    print(f"\nCircuit Summary for {target_recovery_percent*100:.0f}% Performance:")
    print(f"Total components: {total_components}")
    print(f"SAE features: {total_sae} ({total_sae/total_components*100:.1f}%)")
    print(f"Error terms: {total_err} ({total_err/total_components*100:.1f}%)")
    print(f"Recovery achieved: {target_recovery:.4f}")
    
else:
    print(f"Could not find k value that reaches {target_recovery_percent*100:.0f}% target within tested range")

# %%
unique_latents = set()
for entry in target_circuit:
    layer = entry["layer_idx"]
    if entry["type"] == "SAE" and entry["layer_idx"] == 4:
        unique_latents.add(entry["latent_idx"])
print(sorted(unique_latents))



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
# %%


contact_map = full_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end]
print(contact_map.shape)

# Visualize the contact map
plt.figure(figsize=(8, 6))
plt.imshow(contact_map.cpu().numpy(), cmap='viridis', interpolation='nearest')
plt.colorbar(label='Contact Strength')
plt.title(f'Contact Map ({contact_map.shape[0]} x {contact_map.shape[1]})')
plt.xlabel('Residue Index (SS2)')
plt.ylabel('Residue Index (SS1)')
plt.tight_layout()
plt.show()

# %%
contact_map = clean_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end]
print(contact_map.shape)

# Visualize the contact map
plt.figure(figsize=(8, 6))
plt.imshow(contact_map.cpu().numpy(), cmap='viridis', interpolation='nearest')
plt.colorbar(label='Contact Strength')
plt.title(f'Contact Map ({contact_map.shape[0]} x {contact_map.shape[1]})')
plt.xlabel('Residue Index (SS2)')
plt.ylabel('Residue Index (SS1)')
plt.tight_layout()
plt.show()


# %%

contact_map = corr_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end]
print(contact_map.shape)

# Visualize the contact map
plt.figure(figsize=(8, 6))
plt.imshow(contact_map.cpu().numpy(), cmap='viridis', interpolation='nearest')
plt.colorbar(label='Contact Strength')
plt.title(f'Corrupted Contact Map ({contact_map.shape[0]} x {contact_map.shape[1]})')
plt.xlabel('Residue Index (SS2)')
plt.ylabel('Residue Index (SS1)')
plt.tight_layout()
plt.show()

# %%
# Calculate the difference between clean and corrupted contact maps
clean_contact_map = clean_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end]
corr_contact_map = corr_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end]
contact_diff = clean_contact_map - corr_contact_map

print(f"Contact difference shape: {contact_diff.shape}")
print(f"Max difference: {contact_diff.max().item():.4f}")
print(f"Min difference: {contact_diff.min().item():.4f}")

# Find indices with highest differences
contact_diff_flat = contact_diff.flatten()
top_k = 5  # Get top 5 highest differences
top_values, top_indices = torch.topk(contact_diff_flat, top_k)

print(f"\nTop {top_k} highest differences:")
for i, (val, idx) in enumerate(zip(top_values, top_indices)):
    # Convert flat index back to 2D coordinates
    row_idx = idx // contact_diff.shape[1]
    col_idx = idx % contact_diff.shape[1]
    print(f"{i+1}. Position ({row_idx.item()}, {col_idx.item()}): difference = {val.item():.4f}")

# Visualize the difference map
plt.figure(figsize=(8, 6))
plt.imshow(contact_diff.cpu().numpy(), cmap='RdBu_r', interpolation='nearest')
plt.colorbar(label='Clean - Corrupted Difference')
plt.title(f'Contact Difference Map (Clean - Corrupted)')
plt.xlabel('Residue Index (SS2)')
plt.ylabel('Residue Index (SS1)')
plt.tight_layout()
plt.show()

# %%


# %%

# %%