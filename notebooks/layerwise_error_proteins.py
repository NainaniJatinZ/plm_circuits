# %%
"""
steps:
1. load the esm650 details dict
2. load models and saes
3. define helper functions
4. iterate and store, r0, and n_feats  
"""

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

# Additional imports
import json
from functools import partial
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

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

# Load jump details
with open('../data/jump_details_esm_650.json', "r") as json_file:
    jump_details_esm_650 = json.load(json_file)

set_seed(0)

# %%

def prepare_protein_sequences(protein: str, seq_dict: Dict, ss_dict: Dict, jump_details: Dict) -> Dict:
    """
    Prepare clean and corrupted sequences for a given protein.
    
    Args:
        protein: Protein identifier
        seq_dict: Dictionary mapping protein IDs to sequences
        ss_dict: Dictionary mapping protein IDs to secondary structure positions
        jump_details: Dictionary containing optimal flank lengths for each protein
    
    Returns:
        Dictionary containing all necessary data for analysis
    """
    seq = seq_dict[protein]
    ss1_start = ss_dict[protein][0][0] - 5 
    ss1_end = ss_dict[protein][0][0] + 5 + 1 
    ss2_start = ss_dict[protein][0][1] - 5 
    ss2_end = ss_dict[protein][0][1] + 5 + 1 

    details = jump_details[protein]
    clean_flank_length = details['clean_flank_length']
    clean_recovery = details['clean_recovery']
    corrupted_flank_length = details['corrupted_flank_length']
    corrupted_recovery = details['corrupted_recovery']

    # Get full sequence contact predictions
    full_seq_L = [(1, seq)]
    _, _, batch_tokens_BL = batch_converter(full_seq_L)
    batch_tokens_BL = batch_tokens_BL.to(device)
    batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

    with torch.no_grad():
        full_seq_contact_LL = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]

    # Create patching metric
    _patching_metric = partial(
        patching_metric,
        orig_contact=full_seq_contact_LL,
        ss1_start=ss1_start,
        ss1_end=ss1_end,
        ss2_start=ss2_start,
        ss2_end=ss2_end,
    )

    # Prepare clean sequence (with optimal flanks)
    L = len(seq)
    left_start = max(0, ss1_start - clean_flank_length)
    left_end = ss1_start
    right_start = ss2_end
    right_end = min(L, ss2_end + clean_flank_length)
    unmask_left_idxs = list(range(left_start, left_end))
    unmask_right_idxs = list(range(right_start, right_end))

    clean_seq_L = mask_flanks_segment(seq, ss1_start, ss1_end, ss2_start, ss2_end, unmask_left_idxs, unmask_right_idxs)
    _, _, clean_batch_tokens_BL = batch_converter([(1, clean_seq_L)])
    clean_batch_tokens_BL = clean_batch_tokens_BL.to(device)
    clean_batch_mask_BL = (clean_batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

    with torch.no_grad():
        clean_seq_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]

    # Prepare corrupted sequence (with suboptimal flanks)
    left_start = max(0, ss1_start - corrupted_flank_length)
    left_end = ss1_start
    right_start = ss2_end
    right_end = min(L, ss2_end + corrupted_flank_length)
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

    return {
        'protein': protein,
        'seq': seq,
        'ss1_start': ss1_start,
        'ss1_end': ss1_end,
        'ss2_start': ss2_start,
        'ss2_end': ss2_end,
        'clean_flank_length': clean_flank_length,
        'corrupted_flank_length': corrupted_flank_length,
        'full_seq_contact_LL': full_seq_contact_LL,
        'clean_seq_contact_LL': clean_seq_contact_LL,
        'corr_seq_contact_LL': corr_seq_contact_LL,
        'clean_batch_tokens_BL': clean_batch_tokens_BL,
        'clean_batch_mask_BL': clean_batch_mask_BL,
        'corr_batch_tokens_BL': corr_batch_tokens_BL,
        'corr_batch_mask_BL': corr_batch_mask_BL,
        'baseline_recovery': baseline_recovery,
        'corrupted_recovery': corrupted_recovery,
        '_patching_metric': _patching_metric
    }

# %%

def compute_causal_effects_all_layers(protein_data: Dict, main_layers: List[int], 
                                     selected_layers: Optional[List[int]] = None) -> Tuple:
    """
    Compute causal effects for selected layers using integrated gradients.
    
    Args:
        protein_data: Dictionary containing all protein-specific data
        main_layers: List of all available layer indices
        selected_layers: Optional list of specific layers to analyze. If None, uses all main_layers.
    
    Returns:
        Tuple containing effects and caches for selected layers
    """
    # Determine which layers to process
    layers_to_process = selected_layers if selected_layers is not None else main_layers
    print(f"Computing causal effects using integrated gradients for layers: {layers_to_process}")

    all_effects_sae_ALS = []
    all_effects_err_ABLF = []
    clean_layer_caches = {}
    corr_layer_caches = {}
    clean_layer_errors = {}
    corr_layer_errors = {}

    for layer_idx in layers_to_process:
        print(f"\nProcessing layer {layer_idx}...")
        
        sae_model = saes[layer_2_saelayer[layer_idx]]

        # Get clean cache and error
        clean_handle = None
        try:
            hook = SAEHookProt(sae=sae_model, mask_BL=protein_data['clean_batch_mask_BL'], 
                              cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
            clean_handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
            with torch.no_grad():
                clean_seq_sae_contact_LL = esm_transformer.predict_contacts(
                    protein_data['clean_batch_tokens_BL'], protein_data['clean_batch_mask_BL'])[0]
            clean_cache_LS = sae_model.feature_acts.clone()
            clean_err_cache_BLF = sae_model.error_term.clone()
            clean_contact_recovery = protein_data['_patching_metric'](clean_seq_sae_contact_LL)
        finally:
            if clean_handle is not None:
                clean_handle.remove()
            cleanup_cuda()

        # Get corrupted cache and error
        corr_handle = None
        try:
            hook = SAEHookProt(sae=sae_model, mask_BL=protein_data['corr_batch_mask_BL'], 
                              cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
            corr_handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
            with torch.no_grad():
                corr_seq_sae_contact_LL = esm_transformer.predict_contacts(
                    protein_data['corr_batch_tokens_BL'], protein_data['corr_batch_mask_BL'])[0]
            corr_cache_LS = sae_model.feature_acts.clone()
            corr_err_cache_BLF = sae_model.error_term.clone()
        finally:
            if corr_handle is not None:
                corr_handle.remove()
            cleanup_cuda()
        
        print(f"Layer {layer_idx}: Clean contact recovery: {clean_contact_recovery:.4f}")

        # Run integrated gradients
        effect_sae_LS, effect_err_BLF = integrated_gradients_sae(
            esm_transformer,
            sae_model,
            protein_data['_patching_metric'],
            clean_cache_LS.to(device),
            corr_cache_LS.to(device),
            clean_err_cache_BLF.to(device),
            corr_err_cache_BLF.to(device),
            batch_tokens=protein_data['clean_batch_tokens_BL'],
            batch_mask=protein_data['clean_batch_mask_BL'],
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
    
    return (all_effects_sae_ALS, all_effects_err_ABLF, clean_layer_caches, 
            corr_layer_caches, clean_layer_errors, corr_layer_errors)

# %%

def find_k_for_recovery_threshold(target_layer: int, target_recovery_percent: float, 
                                all_effects_sae_ALS: torch.Tensor,
                                clean_layer_caches: Dict, corr_layer_caches: Dict,
                                clean_layer_errors: Dict, protein_data: Dict,
                                max_k: int = 1000, baseline_recovery: float = 0.0, 
                                r0_percent: float = 0.0) -> Tuple[int, float, float]:
    """
    Find the number of top-k features needed to reach a target recovery threshold.
    
    Args:
        target_layer: Layer to analyze
        target_recovery_percent: Target recovery percentage (e.g., 0.7 for 70%)
        all_effects_sae_ALS: Causal effects tensor
        clean_layer_caches: Clean activation caches
        corr_layer_caches: Corrupted activation caches
        clean_layer_errors: Clean error terms
        protein_data: Dictionary containing protein-specific data
        max_k: Maximum number of features to test
        baseline_recovery: Baseline recovery score
        r0_percent: Percentage for r0 calculation
    
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
        handle = None
        try:
            hook = SAEHookProt(
                sae=sae_model,
                mask_BL=protein_data['clean_batch_mask_BL'],
                patch_mask_BLS=sae_mask_LS.to(device),
                patch_value=corr_layer_caches[target_layer].to(device),
                use_mean_error=True,
            )
            handle = esm_transformer.esm.encoder.layer[target_layer].register_forward_hook(hook)
            
            # Forward pass & metric
            with torch.no_grad():
                preds_LL = esm_transformer.predict_contacts(
                    protein_data['clean_batch_tokens_BL'], protein_data['clean_batch_mask_BL'])[0]
            recovery = protein_data['_patching_metric'](preds_LL)
        finally:
            # Clean up
            if handle is not None:
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
    return best_k, best_recovery, r0

# %%

def analyze_protein_layerwise_error(protein: str, target_recovery_percent: float = 0.7, 
                                   selected_layers: Optional[List[int]] = None) -> Dict:
    """
    Complete analysis pipeline for a single protein.
    
    Args:
        protein: Protein identifier
        target_recovery_percent: Target recovery percentage (default: 0.7 for 70%)
        selected_layers: Optional list of specific layers to analyze. If None, uses all main_layers.
                        E.g., [4, 8, 12, 16] to analyze only the top 4 layers.
    
    Returns:
        Dictionary containing analysis results
    """
    print(f"\n{'='*50}")
    print(f"Analyzing protein: {protein}")
    print(f"{'='*50}")
    
    # Step 1: Prepare sequences
    protein_data = prepare_protein_sequences(protein, seq_dict, ss_dict, jump_details_esm_650)
    
    print(f"Protein: {protein}")
    print(f"Clean Flank length: {protein_data['clean_flank_length']}, Clean Recovery: {protein_data['baseline_recovery']:.4f}")
    print(f"Corrupted Flank length: {protein_data['corrupted_flank_length']}, Corrupted Recovery: {protein_data['corrupted_recovery']:.4f}")
    
    # Check if corrupted performance is too high (> 0.1)
    corrupted_recovery_val = protein_data['corrupted_recovery']
    if isinstance(corrupted_recovery_val, torch.Tensor):
        corrupted_recovery_val = corrupted_recovery_val.cpu().item()
    
    if corrupted_recovery_val > 0.1:
        print(f"Skipping protein {protein}: corrupted recovery {corrupted_recovery_val:.4f} > 0.1 threshold")
        return {
            'protein': protein,
            'skipped': True,
            'reason': f'corrupted_recovery_{corrupted_recovery_val:.4f}_above_threshold',
            'baseline_recovery': protein_data['baseline_recovery'],
            'corrupted_recovery': corrupted_recovery_val
        }
    
    # Step 2: Compute causal effects
    (all_effects_sae_ALS, all_effects_err_ABLF, clean_layer_caches, 
     corr_layer_caches, clean_layer_errors, corr_layer_errors) = compute_causal_effects_all_layers(
         protein_data, main_layers, selected_layers)
    
    # Step 3: Find k for each layer
    layer_circuit_sizes = {}
    layer_circuit_recoveries = {}
    layer_r0_values = {}
    
    baseline_recovery = protein_data['baseline_recovery']
    if isinstance(baseline_recovery, torch.Tensor):
        baseline_recovery = baseline_recovery.cpu().item()
    
    # Determine which layers to process  
    layers_to_process = selected_layers if selected_layers is not None else main_layers
    
    for layer in layers_to_process:
        k, recovery, r0 = find_k_for_recovery_threshold(
            layer, target_recovery_percent, all_effects_sae_ALS,
            clean_layer_caches, corr_layer_caches, clean_layer_errors, 
            protein_data, baseline_recovery=baseline_recovery, 
            r0_percent=0.0
        )
        layer_circuit_sizes[layer] = k
        layer_circuit_recoveries[layer] = recovery
        layer_r0_values[layer] = r0
        
    print(f"\nCircuit sizes for {target_recovery_percent*100:.0f}% baseline recovery:")
    for layer in layers_to_process:
        print(f"Layer {layer}: {layer_circuit_sizes[layer]} features (recovery: {layer_circuit_recoveries[layer]:.4f}, r0: {layer_r0_values[layer]:.4f})")
    
    return {
        'protein': protein,
        'skipped': False,
        'baseline_recovery': baseline_recovery,
        'corrupted_recovery': protein_data['corrupted_recovery'],
        'layer_circuit_sizes': layer_circuit_sizes,
        'layer_circuit_recoveries': layer_circuit_recoveries,
        'layer_r0_values': layer_r0_values,
        'target_recovery_percent': target_recovery_percent,
        'selected_layers': layers_to_process,
        'clean_flank_length': protein_data['clean_flank_length'],
        'corrupted_flank_length': protein_data['corrupted_flank_length']
    }

# %%

# Example usage:
# # Analyze all layers
# results = analyze_protein_layerwise_error('2EK8A', target_recovery_percent=0.7)
# 
# # Analyze only top 4 layers for speed
# results_top4 = analyze_protein_layerwise_error('2EK8A', target_recovery_percent=0.7, 
#                                               selected_layers=[4, 8, 12, 16])
# 
# print(f"Results for {results['protein']}:")
# print(f"Baseline recovery: {results['baseline_recovery']:.4f}")
# print(f"Layer circuit sizes: {results['layer_circuit_sizes']}")
# print(f"Layer r0 values: {results['layer_r0_values']}")

# %%

# For batch processing multiple proteins with layer selection:
all_results = {}
skipped_proteins = []
processed_proteins = []
top_layers = [4, 8, 12, 16]  # Focus on top 4 layers for speed

for protein in list(jump_details_esm_650.keys()):  # Process first 5 proteins
    try:
        results = analyze_protein_layerwise_error(protein, target_recovery_percent=0.7, 
                                                selected_layers=top_layers)
        all_results[protein] = results
        
        if results.get('skipped', False):
            skipped_proteins.append(protein)
            print(f"Skipped {protein}: {results['reason']}")
        else:
            processed_proteins.append(protein)
            print(f"Completed {protein}: {list(results['layer_circuit_sizes'].keys())}")
    except Exception as e:
        print(f"Error processing {protein}: {e}")
        all_results[protein] = {
            'protein': protein,
            'error': str(e),
            'skipped': True,
            'reason': 'processing_error'
        }
        skipped_proteins.append(protein)
        continue

# Save results to JSON
import json
from datetime import datetime

# Create summary statistics
summary = {
    'total_proteins': len(all_results),
    'processed_proteins': len(processed_proteins),
    'skipped_proteins': len(skipped_proteins),
    'skipped_list': skipped_proteins,
    'processed_list': processed_proteins,
    'target_recovery_percent': 0.7,
    'selected_layers': top_layers,
    'timestamp': datetime.now().isoformat()
}

# Prepare data for JSON serialization (convert tensors to floats)
def serialize_results(results):
    serialized = {}
    for protein, data in results.items():
        serialized_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                serialized_data[key] = value.cpu().item()
            elif isinstance(value, dict):
                # Handle nested dictionaries (like layer_circuit_sizes)
                serialized_data[key] = {str(k): (v.cpu().item() if isinstance(v, torch.Tensor) else v) 
                                      for k, v in value.items()}
            else:
                serialized_data[key] = value
        serialized[protein] = serialized_data
    return serialized

serialized_results = serialize_results(all_results)

# Save to JSON file
output_data = {
    'summary': summary,
    'results': serialized_results
}

output_filename = f"layerwise_error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_filename, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n{'='*60}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Total proteins analyzed: {summary['total_proteins']}")
print(f"Successfully processed: {summary['processed_proteins']}")
print(f"Skipped (corrupted > 0.1 or errors): {summary['skipped_proteins']}")
print(f"Results saved to: {output_filename}")
print(f"{'='*60}")

# %%

def load_and_analyze_results(json_filename: str):
    """
    Load and analyze saved results from JSON file.
    
    Args:
        json_filename: Path to the JSON results file
    
    Returns:
        Dictionary with loaded results and summary statistics
    """
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    summary = data['summary']
    results = data['results']
    
    print(f"Loaded results from: {json_filename}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Total proteins: {summary['total_proteins']}")
    print(f"Successfully processed: {summary['processed_proteins']}")
    print(f"Skipped: {summary['skipped_proteins']}")
    print(f"Target recovery: {summary['target_recovery_percent']*100:.0f}%")
    print(f"Analyzed layers: {summary['selected_layers']}")
    
    if summary['processed_proteins'] > 0:
        print(f"\nProcessed proteins:")
        for protein in summary['processed_list']:
            if protein in results and not results[protein].get('skipped', False):
                protein_data = results[protein]
                print(f"  {protein}: baseline={protein_data['baseline_recovery']:.3f}, "
                      f"corrupted={protein_data['corrupted_recovery']:.3f}")
                
                layer_sizes = protein_data['layer_circuit_sizes']
                print(f"    Circuit sizes: {layer_sizes}")
    
    if summary['skipped_proteins'] > 0:
        print(f"\nSkipped proteins:")
        for protein in summary['skipped_list']:
            if protein in results:
                reason = results[protein].get('reason', 'unknown')
                print(f"  {protein}: {reason}")
    
    return data

# Example usage for loading results:
# data = load_and_analyze_results('layerwise_error_analysis_20241219_143022.json')

# %%

def merge_sequence_logos():
    """
    Merge sequence logos from source directories into the current results directory.
    Only copies files that don't already exist in the target directory.
    """
    import os
    import shutil
    from pathlib import Path
    
    # Define paths
    target_dir = Path("../results/sequence_logos_clean/")
    source_dirs = [
        Path("/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/saving/alt_results/results/sequence_logos_clean/"),
        Path("/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/saving/results/sequence_logos_clean/")
    ]
    
    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get existing files in target directory
    existing_files = set(f.name for f in target_dir.glob("*.png"))
    print(f"Target directory has {len(existing_files)} existing files")
    
    copied_count = 0
    skipped_count = 0
    
    # Process each source directory
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Source directory does not exist: {source_dir}")
            continue
            
        print(f"\nProcessing source: {source_dir}")
        source_files = list(source_dir.glob("*.png"))
        print(f"Found {len(source_files)} files in source")
        
        for source_file in source_files:
            target_file = target_dir / source_file.name
            
            if source_file.name in existing_files:
                print(f"  SKIP: {source_file.name} (already exists)")
                skipped_count += 1
            else:
                try:
                    shutil.copy2(source_file, target_file)
                    print(f"  COPY: {source_file.name}")
                    existing_files.add(source_file.name)  # Add to existing set to avoid duplicates from other sources
                    copied_count += 1
                except Exception as e:
                    print(f"  ERROR copying {source_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Files copied: {copied_count}")
    print(f"Files skipped (already exist): {skipped_count}")
    print(f"Total files in target directory: {len(list(target_dir.glob('*.png')))}")
    print(f"{'='*60}")

# Run the merge
merge_sequence_logos()

# %%
