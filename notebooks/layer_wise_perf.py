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

400*4096

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

# Perform causal ranking for all latent-token pairs across layers
print("Starting causal ranking with integrated gradients...")

all_effects_sae_ALS = []
all_effects_err_ABLF = []

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
    corr_contact_recovery = _patching_metric(corr_seq_sae_contact_LL)
    
    print(f"Layer {layer_idx}: Clean contact recovery: {clean_contact_recovery:.4f}, Corr contact recovery: {corr_contact_recovery:.4f}")

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

# Stack all effects
all_effects_sae_ALS = torch.stack(all_effects_sae_ALS)
all_effects_err_ABLF = torch.stack(all_effects_err_ABLF)

print(f"\nCausal ranking complete!")
print(f"SAE effects shape: {all_effects_sae_ALS.shape}")
print(f"Error effects shape: {all_effects_err_ABLF.shape}")

# %%

print("Creating layer-wise caches for performance analysis...")

clean_layer_caches = {}
corr_layer_caches = {}
clean_layer_errors = {}
corr_layer_errors = {}

for layer_idx in main_layers:
    sae_model = saes[layer_2_saelayer[layer_idx]]
    
    # Clean caches
    hook = SAEHookProt(sae=sae_model, mask_BL=clean_batch_mask_BL, cache_latents=True, 
                       layer_is_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        clean_seq_sae_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
    cleanup_cuda()
    handle.remove()
    print(f"Layer {layer_idx}, clean score: {_patching_metric(clean_seq_sae_contact_LL):.4f}")
    clean_layer_caches[layer_idx] = sae_model.feature_acts
    clean_layer_errors[layer_idx] = sae_model.error_term
    # print shapes
    print(clean_layer_caches[layer_idx].shape, clean_layer_errors[layer_idx].shape)

    # Corrupted caches
    hook = SAEHookProt(sae=sae_model, mask_BL=corr_batch_mask_BL, cache_latents=True, 
                       layer_is_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        corr_seq_sae_contact_LL = esm_transformer.predict_contacts(corr_batch_tokens_BL, corr_batch_mask_BL)[0]
    cleanup_cuda()
    handle.remove()
    print(f"Layer {layer_idx}, corr score: {_patching_metric(corr_seq_sae_contact_LL):.4f}")
    corr_layer_caches[layer_idx] = sae_model.feature_acts
    corr_layer_errors[layer_idx] = sae_model.error_term

print("Layer-wise caches created successfully!")

# %% Function to patch top-k features for a given layer

def patch_top_k_features(target_layer, k_value, all_effects_sae_ALS, clean_layer_caches, corr_layer_caches, 
                        saes, layer_2_saelayer, esm_transformer, clean_batch_tokens_BL, clean_batch_mask_BL, 
                        clean_layer_errors, corr_layer_errors,
                        _patching_metric, device, verbose=False):
    """
    Patch the top-k most important features for a given layer and return recovery score.
    
    Args:
        target_layer: Layer to patch
        k_value: Number of top features to patch
        ... (other arguments are the data structures from above)
        verbose: Whether to print debug info
    
    Returns:
        recovery: Contact recovery score after patching
    """
    
    # Get the SAE model and effects for this layer
    sae_model = saes[layer_2_saelayer[target_layer]]
    
    # Set mean error to clean error for this layer
    sae_model.mean_error = clean_layer_errors[target_layer]
    
    target_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[target_layer]]
    target_effect_sae_flat_LxS = target_effect_sae_LS.reshape(-1)
    
    # Get top-k indices
    top_rank_vals, top_idx = torch.topk(target_effect_sae_flat_LxS, k=k_value, largest=False, sorted=True)
    
    # Convert flattened indices back to 2D coordinates
    L, S = target_effect_sae_LS.shape
    row_indices = top_idx // S
    col_indices = top_idx % S
    
    if verbose:
        print(f"Layer {target_layer}, K={k_value}: Top effect = {top_rank_vals[0]:.6f}")
    
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
    
    # Convert to CPU float for plotting
    if isinstance(recovery, torch.Tensor):
        recovery = recovery.cpu().item()
    
    return recovery

# %% Function to sweep k values across layers

def sweep_k_values(target_layers, k_values, all_effects_sae_ALS, clean_layer_caches, corr_layer_caches,
                  saes, layer_2_saelayer, esm_transformer, clean_batch_tokens_BL, clean_batch_mask_BL,
                  clean_layer_errors, corr_layer_errors,
                  _patching_metric, device):
    """
    Sweep through k values for multiple target layers and collect recovery scores.
    
    Returns:
        results: Dictionary with layer as key, and (k_values, recoveries) as values
    """
    
    results = {}
    
    for target_layer in target_layers:
        print(f"\nProcessing layer {target_layer}...")
        recoveries = []
        
        for k in k_values:
            recovery = patch_top_k_features(
                target_layer, k, all_effects_sae_ALS, clean_layer_caches, corr_layer_caches,
                saes, layer_2_saelayer, esm_transformer, clean_batch_tokens_BL, clean_batch_mask_BL,
                clean_layer_errors, corr_layer_errors,
                _patching_metric, device, verbose=True
            )
            recoveries.append(recovery)
            
        results[target_layer] = (k_values, recoveries)
        print(f"Layer {target_layer} complete!")
    
    return results

# %%

all_effects_sae_ALS.shape

# %% Run the sweep

# Define parameters for the sweep
target_layers = [4, 8, 12, 16, 20, 24, 28]  # You can modify this list
k_values = [1, 5, 10, 20, 50, 100, 200]  # You can modify this list

print("Starting k-value sweep across layers...")

# Clear memory before starting
sae_model_list, esm_transformer = clear_memory(saes, esm_transformer)
saes = sae_model_list

# Run the sweep
sweep_results = sweep_k_values(
    target_layers, k_values, all_effects_sae_ALS, clean_layer_caches, corr_layer_caches,
    saes, layer_2_saelayer, esm_transformer, clean_batch_tokens_BL, clean_batch_mask_BL,
    clean_layer_errors, corr_layer_errors,
    _patching_metric, device
)

print("Sweep complete!")

# %% Plot results

plt.figure(figsize=(10, 6))

for layer in target_layers:
    k_vals, recoveries = sweep_results[layer]
    plt.plot(k_vals, recoveries, marker='o', label=f'Layer {layer}', linewidth=2)

plt.xlabel('Number of Top Features Patched (k)', fontsize=12)
plt.ylabel('Contact Recovery Score', fontsize=12) 
plt.title('Contact Recovery vs Number of Patched Features by Layer', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')  # Log scale for k values since they span wide range
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Results:")
print("=" * 50)
for layer in target_layers:
    k_vals, recoveries = sweep_results[layer]
    min_recovery = min(recoveries)
    max_recovery = max(recoveries) 
    print(f"Layer {layer:2d}: Recovery range [{min_recovery:.4f}, {max_recovery:.4f}]")

# %% sanity check
target_layer = 8
sae_model = saes[layer_2_saelayer[target_layer]]
corr_err_cache_BLF = corr_layer_errors[target_layer]
clean_err_cache_BLF = clean_layer_errors[target_layer]
# put the mean error for that of clean 
sae_model.mean_error = clean_err_cache_BLF

sae_hook = SAEHookProt(sae=sae_model, mask_BL=clean_batch_mask_BL, patch_latent_S=torch.arange(4096), patch_value=corr_layer_caches[target_layer].to(device),cache_latents=False, layer_is_lm=False, calc_error=False, use_error=False, use_mean_error=True)
# sae_hook = SAEHookProt(sae=sae_model, mask_BL=clean_batch_mask_BL, layer_is_lm=False, calc_error=False, use_error=False, use_mean_error=True)
hook = esm_transformer.esm.encoder.layer[target_layer].register_forward_hook(sae_hook)
with torch.no_grad():
    sanity_check_clean_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
hook.remove()
cleanup_cuda()
print(f"Layer {target_layer}, clean score: {_patching_metric(sanity_check_clean_contact_LL):.4f}")
# patching_metric(sanity_check_clean_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end)

# %% Test Gating Hypothesis - Corrupt Layer 4, Cache Layer 8

print("Testing gating hypothesis: Corrupting layer 4, caching layer 8...")

# Define layers for the gating test
corrupt_layer = 4
cache_layer = 8

# Get top-k positions for layer 4 (to corrupt)
k_corrupt = 10
sae_model_corrupt = saes[layer_2_saelayer[corrupt_layer]]
sae_model_corrupt.mean_error = clean_layer_errors[corrupt_layer]

target_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[corrupt_layer]]
target_effect_sae_flat_LxS = target_effect_sae_LS.reshape(-1)
top_rank_vals, top_idx = torch.topk(target_effect_sae_flat_LxS, k=k_corrupt, largest=False, sorted=True)

# Convert to 2D coordinates
L, S = target_effect_sae_LS.shape
row_indices = top_idx // S
col_indices = top_idx % S

print(f"Corrupting top {k_corrupt} features in layer {corrupt_layer}")
print(f"Top effect value: {top_rank_vals[0]:.6f}")

# Create corruption mask for layer 4 - True means corrupt these positions
corrupt_mask_LS = torch.zeros((L, S), dtype=torch.bool, device=device)
for i in range(len(top_idx)):
    row = row_indices[i]
    col = col_indices[i]
    corrupt_mask_LS[row, col] = True

# Setup caching SAE for layer 8
sae_model_cache = saes[layer_2_saelayer[cache_layer]]
sae_model_cache.mean_error = clean_layer_errors[cache_layer]

# Use pre-computed clean activations for layer 8
print("Using pre-computed clean activations for layer 8...")
baseline_acts_layer8 = clean_layer_caches[cache_layer].clone()

# We also know the baseline score from when we computed the clean caches
# But let's get it for reference
with torch.no_grad():
    baseline_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
baseline_score = _patching_metric(baseline_contact_LL)
print(f"Baseline score: {baseline_score:.4f}")

# Now, corrupt layer 4 and cache layer 8 activations
print("Corrupting layer 4 and caching layer 8...")

# Setup corruption hook for layer 4
corrupt_hook = SAEHookProt(
    sae=sae_model_corrupt,
    mask_BL=clean_batch_mask_BL,
    patch_mask_BLS=corrupt_mask_LS.to(device),
    patch_value=corr_layer_caches[corrupt_layer].to(device),
    use_mean_error=True,
)

# Setup caching hook for layer 8
cache_hook = SAEHookProt(
    sae=sae_model_cache,
    mask_BL=clean_batch_mask_BL,
    cache_latents=True,
    layer_is_lm=False,
    calc_error=False,
    use_error=False,
    use_mean_error=True
)

# Register both hooks
corrupt_handle = esm_transformer.esm.encoder.layer[corrupt_layer].register_forward_hook(corrupt_hook)
cache_handle = esm_transformer.esm.encoder.layer[cache_layer].register_forward_hook(cache_hook)

# Forward pass with corruption
with torch.no_grad():
    corrupted_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]

# Clean up hooks
corrupt_handle.remove()
cache_handle.remove()
cleanup_cuda()

# Get the activations after corruption
corrupted_acts_layer8 = sae_model_cache.feature_acts.clone()
corrupted_score = _patching_metric(corrupted_contact_LL)

print(f"Score after corrupting layer {corrupt_layer}: {corrupted_score:.4f}")
print(f"Score change: {corrupted_score - baseline_score:.6f}")

# %%



# Analyze the activation changes for causally important features in layer 8
activation_diff = corrupted_acts_layer8 - baseline_acts_layer8

# Get the top causally relevant pairs for layer 8
k_important = 50
layer8_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[cache_layer]]
layer8_effect_sae_flat = layer8_effect_sae_LS.reshape(-1)
layer8_top_vals, layer8_top_idx = torch.topk(layer8_effect_sae_flat, k=k_important, largest=False, sorted=True)

# Convert to 2D coordinates for layer 8
L8, S8 = layer8_effect_sae_LS.shape
layer8_row_indices = layer8_top_idx // S8
layer8_col_indices = layer8_top_idx % S8

print(f"\nAnalyzing changes in top {k_important} causally important features in layer {cache_layer}:")
print(f"Most important feature effect: {layer8_top_vals[0]:.6f}")

# Extract activation changes for these specific important positions
important_activation_changes = []
for i in range(k_important):
    row = layer8_row_indices[i]
    col = layer8_col_indices[i]
    change = activation_diff[row, col].item()
    causal_effect = layer8_top_vals[i].item()
    important_activation_changes.append((i, row.item(), col.item(), change, causal_effect))

# Sort by magnitude of activation change
important_activation_changes.sort(key=lambda x: abs(x[3]), reverse=True)

print(f"\nTop 15 causally important features with largest activation changes:")
print("Rank | (Token, Latent) | Act Change | Causal Effect | Change/Effect Ratio")
print("-" * 75)
for i in range(min(15, len(important_activation_changes))):
    rank, row, col, change, effect = important_activation_changes[i]
    ratio = change / effect if abs(effect) > 1e-8 else float('inf')
    print(f"{rank+1:4d} | ({row:3d}, {col:4d}) | {change:+9.6f} | {effect:9.6f} | {ratio:+9.3f}")

# Calculate statistics specifically for the causally important features
important_changes = torch.tensor([x[3] for x in important_activation_changes])
important_effects = torch.tensor([x[4] for x in important_activation_changes])

print(f"\nStatistics for causally important features in layer {cache_layer}:")
print(f"Mean absolute activation change: {torch.mean(torch.abs(important_changes)):.6f}")
print(f"Max increase in important features: {torch.max(important_changes):.6f}")
print(f"Max decrease in important features: {torch.min(important_changes):.6f}")
print(f"Fraction of important features that increased: {(important_changes > 0).float().mean():.3f}")
print(f"Fraction of important features that decreased: {(important_changes < 0).float().mean():.3f}")

# Check correlation between causal importance and activation change magnitude
abs_changes = torch.abs(important_changes)
abs_effects = torch.abs(important_effects)
correlation = torch.corrcoef(torch.stack([abs_changes, abs_effects]))[0, 1]
print(f"Correlation between |causal effect| and |activation change|: {correlation:.4f}")

# Global statistics for comparison
print(f"\nGlobal activation change statistics:")
print(f"Mean absolute change (all features): {torch.mean(torch.abs(activation_diff)):.6f}")
print(f"Mean absolute change (important features): {torch.mean(abs_changes):.6f}")
print(f"Important features are {torch.mean(abs_changes) / torch.mean(torch.abs(activation_diff)):.2f}x more affected on average")

# %% Systematic Gating Hypothesis Test

def test_gating_hypothesis(corrupt_layer, downstream_layers, k_corrupt, k_test, num_random_trials=5):
    """
    Test gating hypothesis: corrupt one layer and measure effects on downstream layers.
    
    Args:
        corrupt_layer: Layer to corrupt
        downstream_layers: List of layers to test effects on
        k_corrupt: Number of top features to corrupt in corrupt_layer
        k_test: Number of features to test in downstream layers (both important and random)
        num_random_trials: Number of random control groups to test
    
    Returns:
        results: Dictionary with detailed results for each downstream layer
    """
    
    print(f"\n{'='*60}")
    print(f"GATING HYPOTHESIS TEST: Corrupt Layer {corrupt_layer}")
    print(f"Downstream layers: {downstream_layers}")
    print(f"Corrupting top {k_corrupt} features, testing {k_test} features each")
    print(f"{'='*60}")
    
    results = {}
    
    # Get corruption targets for the corrupt layer
    sae_model_corrupt = saes[layer_2_saelayer[corrupt_layer]]
    sae_model_corrupt.mean_error = clean_layer_errors[corrupt_layer]
    
    corrupt_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[corrupt_layer]]
    corrupt_effect_flat = corrupt_effect_sae_LS.reshape(-1)
    corrupt_top_vals, corrupt_top_idx = torch.topk(corrupt_effect_flat, k=k_corrupt, largest=False, sorted=True)
    
    # Convert to 2D coordinates for corruption
    L_corrupt, S_corrupt = corrupt_effect_sae_LS.shape
    corrupt_row_indices = corrupt_top_idx // S_corrupt
    corrupt_col_indices = corrupt_top_idx % S_corrupt
    
    # Create corruption mask
    corrupt_mask_LS = torch.zeros((L_corrupt, S_corrupt), dtype=torch.bool, device=device)
    for i in range(len(corrupt_top_idx)):
        row = corrupt_row_indices[i]
        col = corrupt_col_indices[i]
        corrupt_mask_LS[row, col] = True
    
    print(f"Corrupting layer {corrupt_layer}: Top effect = {corrupt_top_vals[0]:.6f}")
    
    # Test each downstream layer
    for cache_layer in downstream_layers:
        print(f"\n--- Testing effects on layer {cache_layer} ---")
        
        # Setup caching for this downstream layer
        sae_model_cache = saes[layer_2_saelayer[cache_layer]]
        sae_model_cache.mean_error = clean_layer_errors[cache_layer]
        
        # Get baseline activations (pre-computed)
        baseline_acts = clean_layer_caches[cache_layer].clone()
        
        # Setup hooks: corruption + caching
        corrupt_hook = SAEHookProt(
            sae=sae_model_corrupt,
            mask_BL=clean_batch_mask_BL,
            patch_mask_BLS=corrupt_mask_LS.to(device),
            patch_value=corr_layer_caches[corrupt_layer].to(device),
            use_mean_error=True,
        )
        
        cache_hook = SAEHookProt(
            sae=sae_model_cache,
            mask_BL=clean_batch_mask_BL,
            cache_latents=True,
            layer_is_lm=False,
            calc_error=False,
            use_error=False,
            use_mean_error=True
        )
        
        # Forward pass with corruption
        corrupt_handle = esm_transformer.esm.encoder.layer[corrupt_layer].register_forward_hook(corrupt_hook)
        cache_handle = esm_transformer.esm.encoder.layer[cache_layer].register_forward_hook(cache_hook)
        
        with torch.no_grad():
            _ = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
        
        corrupt_handle.remove()
        cache_handle.remove()
        cleanup_cuda()
        
        # Get corrupted activations
        corrupted_acts = sae_model_cache.feature_acts.clone()
        activation_diff = torch.abs(corrupted_acts - baseline_acts)
        
        # Get important features for this downstream layer
        cache_effect_sae_LS = all_effects_sae_ALS[layer_2_saelayer[cache_layer]]
        cache_effect_flat = cache_effect_sae_LS.reshape(-1)
        cache_top_vals, cache_top_idx = torch.topk(cache_effect_flat, k=k_test, largest=False, sorted=True)
        
        # Convert to 2D coordinates
        L_cache, S_cache = cache_effect_sae_LS.shape
        cache_row_indices = cache_top_idx // S_cache
        cache_col_indices = cache_top_idx % S_cache
        
        # Extract changes for important features
        important_changes = []
        for i in range(k_test):
            row = cache_row_indices[i]
            col = cache_col_indices[i]
            change = activation_diff[row, col].item()
            important_changes.append(change)
        
        important_mean = np.mean(important_changes)
        important_median = np.median(important_changes)
        
        # Test against multiple random control groups
        random_means = []
        random_medians = []
        
        for trial in range(num_random_trials):
            # Generate random (token, latent) pairs
            random_rows = torch.randint(0, L_cache, (k_test,))
            random_cols = torch.randint(0, S_cache, (k_test,))
            
            random_changes = []
            for i in range(k_test):
                row = random_rows[i]
                col = random_cols[i]
                change = activation_diff[row, col].item()
                random_changes.append(change)
            
            random_means.append(np.mean(random_changes))
            random_medians.append(np.median(random_changes))
        
        # Calculate statistics
        random_mean_avg = np.mean(random_means)
        random_median_avg = np.mean(random_medians)
        random_mean_std = np.std(random_means)
        random_median_std = np.std(random_medians)
        
        # Effect sizes
        mean_effect_size = (important_mean - random_mean_avg) / random_mean_std if random_mean_std > 0 else float('inf')
        median_effect_size = (important_median - random_median_avg) / random_median_std if random_median_std > 0 else float('inf')
        
        # Store results
        layer_results = {
            'important_mean': important_mean,
            'important_median': important_median,
            'random_mean_avg': random_mean_avg,
            'random_median_avg': random_median_avg,
            'random_mean_std': random_mean_std,
            'random_median_std': random_median_std,
            'mean_effect_size': mean_effect_size,
            'median_effect_size': median_effect_size,
            'mean_ratio': important_mean / random_mean_avg if random_mean_avg > 0 else float('inf'),
            'median_ratio': important_median / random_median_avg if random_median_avg > 0 else float('inf'),
            'important_changes': important_changes,
            'random_means': random_means,
            'random_medians': random_medians
        }
        
        results[cache_layer] = layer_results
        
        # Print results
        print(f"Important features - Mean: {important_mean:.6f}, Median: {important_median:.6f}")
        print(f"Random features   - Mean: {random_mean_avg:.6f}±{random_mean_std:.6f}, Median: {random_median_avg:.6f}±{random_median_std:.6f}")
        print(f"Effect sizes      - Mean: {mean_effect_size:.3f}, Median: {median_effect_size:.3f}")
        print(f"Ratios            - Mean: {important_mean/random_mean_avg:.3f}x, Median: {important_median/random_median_avg:.3f}x")
        
        # Simple significance test (important > all random trials)
        mean_wins = sum(1 for rm in random_means if important_mean > rm)
        median_wins = sum(1 for rm in random_medians if important_median > rm)
        print(f"Significance      - Mean: {mean_wins}/{num_random_trials}, Median: {median_wins}/{num_random_trials}")
        
    return results

# %% Run Systematic Gating Tests

# Parameters
k_corrupt = 50  # Number of features to corrupt in source layer
k_test = 50     # Number of features to test in downstream layers
num_random_trials = 10  # Number of random control groups

# Target layers from your existing analysis
test_layers = [4, 8, 12, 16, 20, 24, 28]

# Store all results
all_gating_results = {}

# Test each layer on all downstream layers
for corrupt_layer in test_layers:
    downstream_layers = [layer for layer in test_layers if layer > corrupt_layer]
    if len(downstream_layers) > 0:
        results = test_gating_hypothesis(corrupt_layer, downstream_layers, k_corrupt, k_test, num_random_trials)
        all_gating_results[corrupt_layer] = results

# %%






