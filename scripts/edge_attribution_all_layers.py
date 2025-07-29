#!/usr/bin/env python3
"""
Edge Attribution Analysis Across All Layer Combinations

This script performs edge attribution analysis between all combinations of SAE layers
in protein language models, extending the notebook-based analysis to systematic 
coverage of the full network.

Author: Converted from edge_attr.py notebook
"""

import sys
import os
import json
import argparse
from functools import partial
from typing import List, Dict, Any, Optional, Tuple, Union, Set, Callable
from collections import defaultdict, Counter
import itertools
from pathlib import Path

import torch
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.figure
import networkx as nx

# Add plm_circuits to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'plm_circuits'))

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


# Type aliases for better readability
TensorDict = Dict[str, torch.Tensor]
LayerIndex = int
FeatureIndex = int
ConnectionTuple = Tuple[FeatureIndex, FeatureIndex, float]
EdgeResults = Dict[str, Any]

# Removed unused _compute_jvp_edge_v2 function - edge attribution is implemented inline in analyze_layer_pair


def setup_models_and_data(
    device: torch.device, 
    main_layers: List[LayerIndex], 
    protein: str = "1PVGA"
) -> Dict[str, Any]:
    """Setup ESM model, SAEs, and protein sequence data"""
    print(f"Setting up models and data for protein {protein}")
    
    # Load ESM-2 model (override WEIGHTS_DIR to use writable location)
    weights_dir = '/mnt/polished-lake/home/connor/plm_circuits/.cache/models'
    esm_transformer, batch_converter, esm2_alphabet = load_esm(33, WEIGHTS_DIR=weights_dir, device=device)
    
    # Load SAEs for multiple layers
    saes = []
    for layer in main_layers:
        sae_model = load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=layer, device=device)
        saes.append(sae_model)
    
    layer_2_saelayer = {layer: layer_idx for layer_idx, layer in enumerate(main_layers)}
    
    # Load sequence data and define protein parameters
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'full_seq_dict.json')
    with open(data_path, "r") as json_file:
        seq_dict = json.load(json_file)
    
    # Define protein-specific parameters
    sse_dict: Dict[str, List[List[int]]] = {"2B61A": [[182, 316]], "1PVGA": [[101, 202]]}
    fl_dict: Dict[str, List[int]] = {"2B61A": [44, 43], "1PVGA": [65, 63]}
    
    seq: str = seq_dict[protein]
    position: List[int] = sse_dict[protein][0]
    
    # Define segment boundaries
    ss1_start: int = position[0] - 5 
    ss1_end: int = position[0] + 5 + 1 
    ss2_start: int = position[1] - 5 
    ss2_end: int = position[1] + 5 + 1 
    
    print(f"Analyzing protein: {protein}")
    print(f"Sequence length: {len(seq)}")
    print(f"Segment 1: {ss1_start}-{ss1_end}")
    print(f"Segment 2: {ss2_start}-{ss2_end}")
    
    return {
        'esm_transformer': esm_transformer,
        'batch_converter': batch_converter,
        'esm2_alphabet': esm2_alphabet,
        'saes': saes,
        'layer_2_saelayer': layer_2_saelayer,
        'seq': seq,
        'ss1_start': ss1_start,
        'ss1_end': ss1_end,
        'ss2_start': ss2_start,
        'ss2_end': ss2_end,
        'fl_dict': fl_dict,
        'protein': protein
    }


def prepare_sequences(
    setup_data: Dict[str, Any], 
    device: torch.device
) -> Dict[str, Union[torch.Tensor, Callable]]:
    """Prepare clean and corrupted sequences"""
    esm_transformer = setup_data['esm_transformer']
    batch_converter = setup_data['batch_converter']
    esm2_alphabet = setup_data['esm2_alphabet']
    seq = setup_data['seq']
    ss1_start, ss1_end = setup_data['ss1_start'], setup_data['ss1_end']
    ss2_start, ss2_end = setup_data['ss2_start'], setup_data['ss2_end']
    fl_dict = setup_data['fl_dict']
    protein = setup_data['protein']
    
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
    
    print(f"Clean flank size: {clean_fl}")
    print(f"Clean sequence contact recovery: {_patching_metric(clean_seq_contact_LL):.4f}")
    print(f"Corrupted flank size: {corr_fl}")
    print(f"Corrupted sequence contact recovery: {_patching_metric(corr_seq_contact_LL):.4f}")
    
    return {
        'clean_batch_tokens_BL': clean_batch_tokens_BL,
        'clean_batch_mask_BL': clean_batch_mask_BL,
        'corr_batch_tokens_BL': corr_batch_tokens_BL,
        'corr_batch_mask_BL': corr_batch_mask_BL,
        'patching_metric': _patching_metric,
        'full_seq_contact_LL': full_seq_contact_LL,
        'clean_seq_contact_LL': clean_seq_contact_LL,
        'corr_seq_contact_LL': corr_seq_contact_LL
    }


def perform_causal_ranking(
    setup_data: Dict[str, Any], 
    sequence_data: Dict[str, Union[torch.Tensor, Callable]], 
    device: torch.device, 
    main_layers: List[LayerIndex]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform causal ranking for all latent-token pairs across layers"""
    print("Starting causal ranking with integrated gradients...")
    
    esm_transformer = setup_data['esm_transformer']
    saes = setup_data['saes']
    layer_2_saelayer = setup_data['layer_2_saelayer']
    clean_batch_tokens_BL = sequence_data['clean_batch_tokens_BL']
    clean_batch_mask_BL = sequence_data['clean_batch_mask_BL']
    corr_batch_tokens_BL = sequence_data['corr_batch_tokens_BL']
    corr_batch_mask_BL = sequence_data['corr_batch_mask_BL']
    _patching_metric = sequence_data['patching_metric']
    
    all_effects_sae_ALS: List[torch.Tensor] = []
    all_effects_err_ABLF: List[torch.Tensor] = []
    
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
    
    return all_effects_sae_ALS, all_effects_err_ABLF


def create_layer_caches(
    setup_data: Dict[str, Any], 
    sequence_data: Dict[str, Union[torch.Tensor, Callable]], 
    device: torch.device, 
    main_layers: List[LayerIndex]
) -> Tuple[TensorDict, TensorDict, TensorDict, TensorDict]:
    """Create layer-wise caches for performance analysis"""
    print("Creating layer-wise caches for performance analysis...")
    
    esm_transformer = setup_data['esm_transformer']
    saes = setup_data['saes']
    layer_2_saelayer = setup_data['layer_2_saelayer']
    clean_batch_tokens_BL = sequence_data['clean_batch_tokens_BL']
    clean_batch_mask_BL = sequence_data['clean_batch_mask_BL']
    corr_batch_tokens_BL = sequence_data['corr_batch_tokens_BL']
    corr_batch_mask_BL = sequence_data['corr_batch_mask_BL']
    _patching_metric = sequence_data['patching_metric']
    
    clean_layer_caches: TensorDict = {}
    corr_layer_caches: TensorDict = {}
    clean_layer_errors: TensorDict = {}
    corr_layer_errors: TensorDict = {}
    
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
    
    return clean_layer_caches, corr_layer_caches, clean_layer_errors, corr_layer_errors


def get_top_features(
    effects: torch.Tensor, 
    k: int = 50
) -> Dict[str, Union[torch.Tensor, Set[int]]]:
    """Get top k features for a layer"""
    effect_flat: torch.Tensor = effects.reshape(-1)
    top_rank_vals: torch.Tensor
    top_idx: torch.Tensor
    top_rank_vals, top_idx = torch.topk(effect_flat, k=k, largest=False, sorted=True)
    
    L: int
    S: int
    L, S = effects.shape
    row_indices: torch.Tensor = top_idx // S
    col_indices: torch.Tensor = top_idx % S
    
    return {
        'values': top_rank_vals,
        'row_indices': row_indices,
        'col_indices': col_indices,
        'feature_indices': set([col_indices[i].item() for i in range(len(col_indices))])
    }


def analyze_layer_pair(
    up_layer: LayerIndex, 
    down_layer: LayerIndex, 
    setup_data: Dict[str, Any], 
    sequence_data: Dict[str, Union[torch.Tensor, Callable]], 
    all_effects_sae_ALS: torch.Tensor,
    clean_layer_caches: TensorDict, 
    clean_layer_errors: TensorDict, 
    device: torch.device, 
    k: int = 50
) -> Optional[EdgeResults]:
    """Analyze edge attribution between a specific pair of layers"""
    print(f"\n=== ANALYZING LAYER {up_layer} → LAYER {down_layer} ===")
    
    esm_transformer = setup_data['esm_transformer']
    saes = setup_data['saes']
    layer_2_saelayer = setup_data['layer_2_saelayer']
    clean_batch_tokens_BL = sequence_data['clean_batch_tokens_BL']
    clean_batch_mask_BL = sequence_data['clean_batch_mask_BL']
    
    up_sae = saes[layer_2_saelayer[up_layer]]
    down_sae = saes[layer_2_saelayer[down_layer]]
    
    up_effects = all_effects_sae_ALS[layer_2_saelayer[up_layer]]
    down_effects = all_effects_sae_ALS[layer_2_saelayer[down_layer]]
    
    # Get top features for each layer
    up_features_data = get_top_features(up_effects, k)
    down_features_data = get_top_features(down_effects, k)
    
    up_feats: List[FeatureIndex] = list(up_features_data['feature_indices'])
    down_feats: List[FeatureIndex] = list(down_features_data['feature_indices'])
    
    print(f"Found {len(up_feats)} upstream features and {len(down_feats)} downstream features")
    
    # Prepare data for edge attribution
    L: int
    S: int
    L, S = up_effects.shape
    up_base: torch.Tensor = clean_layer_caches[up_layer].detach().clone().to(device).requires_grad_()
    patch_mask_LS: torch.Tensor = torch.ones((L, S), dtype=torch.bool, device=device)
    up_error: torch.Tensor = clean_layer_errors[up_layer].to(device)
    down_error: torch.Tensor = clean_layer_errors[down_layer].to(device)
    
    # Forward function for VJP
    def _forward_fn() -> torch.Tensor:
        # up hook that puts patch activations 
        up_sae.mean_error = up_error
        up_hook = SAEHookProt(sae=up_sae, mask_BL=clean_batch_mask_BL, patch_mask_BLS=patch_mask_LS, 
                             patch_value=up_base, use_mean_error=True)
        
        # down hook that records downstream activations
        down_sae.mean_error = down_error
        down_hook = SAEHookProt(sae=down_sae, mask_BL=clean_batch_mask_BL, cache_latents=True, 
                               layer_is_lm=False, calc_error=True, use_error=True, no_detach=True)
        
        # register the hooks
        handle_up = esm_transformer.esm.encoder.layer[up_layer].register_forward_hook(up_hook)
        handle_down = esm_transformer.esm.encoder.layer[down_layer].register_forward_hook(down_hook)
        
        # run the forward pass
        _ = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
        handle_up.remove()
        handle_down.remove()
        feats = down_sae.feature_acts
        if not feats.requires_grad:
            raise RuntimeError(
                "[edge-attr-vjp] downstream activations are detached; "
                "remove `.detach()` inside your SAE hook or clone with "
                "`.requires_grad_()` earlier in the graph."
            )
        return feats
    
    # Run edge attribution analysis
    down_base: torch.Tensor = _forward_fn()
    down_grad: torch.Tensor = down_effects.to(device)
    
    # Container: (down_idx, up_idx) → list[val]
    bucket: Dict[Tuple[FeatureIndex, FeatureIndex], List[torch.Tensor]] = {}
    
    # Loop over downstream features (rows of the Jacobian)
    for d_idx in down_feats:
        # Select the scalar we will back-prop; optionally weight by loss grad
        scalar_field = down_base[..., d_idx]
        if down_grad is not None:
            scalar_field = scalar_field * down_grad[..., d_idx]
        scalar = scalar_field.sum()
        
        # Jᵀ ▽  – gradient w.r.t. *entire* upstream latent tensor
        grad_tensor = torch.autograd.grad(
            scalar,
            up_base,
            retain_graph=True,
            create_graph=False,
        )[0]
        
        # Accumulate entries we care about
        for u_idx in up_feats:
            val = grad_tensor[..., u_idx].sum()
            if val.abs() < 1e-6:
                continue
            bucket.setdefault((d_idx, u_idx), []).append(val.detach().cpu())
        
        if d_idx == down_feats[0] or d_idx % 10 == 0:
            print(f"[edge-attr-vjp] processed downstream idx {d_idx}")
    
    # Assemble sparse COO tensor
    if not bucket:
        print(f"No edges found for {up_layer} → {down_layer}")
        return None
    
    idxs: Tuple[Tuple[FeatureIndex, FeatureIndex], ...]
    vals: Tuple[torch.Tensor, ...]
    idxs, vals = zip(
        *[((d, u), torch.stack(v).mean()) for (d, u), v in bucket.items()]
    )
    idx_mat: torch.Tensor = torch.tensor(list(zip(*idxs)), dtype=torch.long)
    val_mat: torch.Tensor = torch.stack(list(vals))
    
    edge_tensor: torch.Tensor = torch.sparse_coo_tensor(
        idx_mat,
        val_mat,
        size=(len(down_feats), len(up_feats)),
    ).coalesce()
    
    nnz: int = edge_tensor._nnz()
    print(f"[edge-attr-vjp] finished – {nnz} non-zero entries")
    
    # Analyze results
    if nnz > 0:
        indices: torch.Tensor = edge_tensor.indices()
        values: torch.Tensor = edge_tensor.values()
        
        down_indices: List[int] = indices[0].tolist()
        up_indices: List[int] = indices[1].tolist()
        edge_values: List[float] = values.tolist()
        
        # Create all_connections format for this layer pair
        all_connections: List[ConnectionTuple] = []
        for i in range(len(down_indices)):
            # The indices are already the actual feature indices, not indices into the feature lists
            down_idx = down_indices[i]
            up_idx = up_indices[i]
            val = edge_values[i]
            all_connections.append((up_idx, down_idx, val))
        
        # Sort by absolute strength
        all_connections.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print(f"Top 5 connections for {up_layer} → {down_layer}:")
        for i, (up_idx, down_idx, val) in enumerate(all_connections[:5]):
            print(f"  {i+1}. Up {up_idx} -> Down {down_idx}: {val:.6f}")
        
        return {
            'edge_tensor': edge_tensor,
            'all_connections': all_connections,
            'up_feats': up_feats,
            'down_feats': down_feats,
            'up_features_data': up_features_data,
            'down_features_data': down_features_data
        }
    
    return None


def load_interpretable_features() -> Dict[LayerIndex, Dict[FeatureIndex, str]]:
    """Load interpretable feature descriptions"""
    # Load from JSON file if available
    layer_latent_dict_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'layer_latent_dict_2b61.json')
    if os.path.exists(layer_latent_dict_path):
        with open(layer_latent_dict_path, 'r') as f:
            layer_latent_dict = json.load(f)
    else:
        layer_latent_dict = {}
    
    # # Hardcoded interpretable features (from original notebook)
    # l4_latents: Dict[FeatureIndex, str] = {
    #     340: "FX'", 237: "FX'", 3788: "X'XI", 798: "D'XXGN", 1690: "X'XXF", 
    #     2277: "G", 2311: "X'XM", 3634: "X'XXXG", 1682: "PXXXXXX'", 3326: "H"
    # }
    # 
    # l8_latents: Dict[FeatureIndex, str] = {
    #     488: "AB_Hydrolase_fold", 2677: "FAD/NAD", 2775: "Transketolase", 2166: "DHFR"
    # }
    # 
    # l12_latents: Dict[FeatureIndex, str] = {
    #     2112: "AB_Hydrolase_fold", 3536: "SAM_mtases", 1256: "FAM", 2797: "Aldolase", 
    #     3794: "SAM_mtases", 3035: "WD40", 2302: "HotDog"
    # }
    # 
    # l16_latents: Dict[FeatureIndex, str] = {
    #     1504: "AB_Hydrolase_fold"
    # }
    # 
    # l20_latents: Dict[FeatureIndex, str] = {
    #     3615: "AB_Hydrolase_fold", 878: "Kinase"
    # }
    # 
    # l24_latents: Dict[FeatureIndex, str] = {
    #     2586: "Pectin lyase", 1822: "Kinase"
    # }
    
    l4_latents: Dict[FeatureIndex, str] = {
        1509:"E", 2511:"X'XQ", 2112:"YXX'", 3069: "GX'", 3544: "C", 2929: "N", 3170: "X'N", 3717:"V", 527: "DX'", 3229: "IXX'", 1297: "I", 1468: "X'XXN", 1196: "D"
    }
    
    l8_latents: Dict[FeatureIndex, str] = {
        1916: "NX'XXNA", 2529:"Hatpase_C", 3159: "Hatpase_C", 3903: "Hatpase_C", 1055: "Hatpase_C", 2066: "Hatpase_C"
    }
    
    l12_latents: Dict[FeatureIndex, str] = {
        3943: "Hatpase_C", 1796: "Hatpase_C", 1204: "Hatpase_C", 1145:  "Hatpase_C", 1082: "XPG-I", 2472: "Kinesin"
    }
    
    l16_latents: Dict[FeatureIndex, str] = {
        3077: "Hatpase_C", 1353: "Hatpase_C", 1597: "Hatpase_C", 1814: "Hatpase_C", 3994: "Ribosomal", 1166: "Hatpase_C"
    }
    
    l20_latents: Dict[FeatureIndex, str] = {}
    
    l24_latents: Dict[FeatureIndex, str] = {}
    
    # You can extend this to include interpretable features for other layers
    interpretable_features: Dict[LayerIndex, Dict[FeatureIndex, str]] = {
        4: l4_latents,
        8: l8_latents,
        12: l12_latents,
        16: l16_latents,
        20: l20_latents,
        24: l24_latents,
        28: {}
    }
    
    return interpretable_features


def create_layer_network_diagram(
    all_connections: List[ConnectionTuple], 
    up_layer: LayerIndex, 
    down_layer: LayerIndex,
    interpretable_features: Dict[LayerIndex, Dict[FeatureIndex, str]], 
    top_k: int = 100, 
    output_dir: Optional[Union[str, Path]] = None
) -> Optional[Tuple[nx.DiGraph, Dict[Tuple[str, str], float]]]:
    """Create a NetworkX diagram showing connections between interpretable features across layers"""
    
    print(f"\n=== CREATING NETWORKX DIAGRAM FOR L{up_layer}→L{down_layer} ===")
    
    up_interpretable: Dict[FeatureIndex, str] = interpretable_features.get(up_layer, {})
    down_interpretable: Dict[FeatureIndex, str] = interpretable_features.get(down_layer, {})
    
    # Get top k edges
    top_k_edges = all_connections[:top_k]
    print(f"Using top {top_k} edges out of {len(all_connections)} total edges")
    
    def map_node(node_idx: FeatureIndex, layer: LayerIndex, interpretable_dict: Dict[FeatureIndex, str]) -> str:
        """Map node to interpretable feature name or 'Other' group"""
        if node_idx in interpretable_dict:
            return f"L{layer}_{node_idx}: {interpretable_dict[node_idx]}"
        else:
            return f"Other L{layer}"
    
    # Get statistics for scaling
    all_strengths: List[float] = [abs(strength) for _, _, strength in top_k_edges]
    if not all_strengths:
        print("No edges to visualize")
        return None
    
    max_strength = max(all_strengths)
    min_strength = min(all_strengths)
    
    print(f"Edge strength stats (top {top_k}):")
    print(f"  Weakest edge: {min_strength:.8f}")
    print(f"  Strongest edge: {max_strength:.8f}")
    print(f"  Dynamic range: {max_strength/min_strength:.1f}x")
    
    # Process edges and aggregate weights
    edge_weights: Dict[Tuple[str, str], float] = {}
    
    for up_idx, down_idx, strength in top_k_edges:
        up_node = map_node(up_idx, up_layer, up_interpretable)
        down_node = map_node(down_idx, down_layer, down_interpretable)
        
        # Square root scaling for better visualization
        abs_strength = abs(strength)
        scaled_strength = np.sqrt(abs_strength)
        
        edge_key = (up_node, down_node)
        if edge_key in edge_weights:
            edge_weights[edge_key] += scaled_strength
        else:
            edge_weights[edge_key] = scaled_strength
    
    print(f"Created {len(edge_weights)} unique edges after grouping")
    
    # Create NetworkX graph
    G = nx.DiGraph()
    for (up_node, down_node), weight in edge_weights.items():
        G.add_edge(up_node, down_node, weight=weight)
    
    # Separate node types for styling
    up_interpretable_nodes = [node for node in G.nodes() if node.startswith(f"L{up_layer}_") and ":" in node]
    down_interpretable_nodes = [node for node in G.nodes() if node.startswith(f"L{down_layer}_") and ":" in node]
    other_up_nodes = [node for node in G.nodes() if node == f"Other L{up_layer}"]
    other_down_nodes = [node for node in G.nodes() if node == f"Other L{down_layer}"]
    
    print(f"Node counts:")
    print(f"  L{up_layer} interpretable: {len(up_interpretable_nodes)}")
    print(f"  L{down_layer} interpretable: {len(down_interpretable_nodes)}")
    print(f"  Other L{up_layer}: {len(other_up_nodes)}")
    print(f"  Other L{down_layer}: {len(other_down_nodes)}")
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Position nodes in two columns
    pos = {}
    all_up_nodes = up_interpretable_nodes + other_up_nodes
    for i, node in enumerate(all_up_nodes):
        pos[node] = (0, len(all_up_nodes) - i)
    
    all_down_nodes = down_interpretable_nodes + other_down_nodes
    for i, node in enumerate(all_down_nodes):
        pos[node] = (3, len(all_down_nodes) - i)
    
    # Calculate edge widths
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    if weights:
        max_weight = max(weights)
        min_weight = min(weights)
        min_width, max_width = 0.8, 5.0
        
        edge_widths = []
        for w in weights:
            if max_weight > min_weight:
                normalized = (w - min_weight) / (max_weight - min_weight)
                width = min_width + normalized * (max_width - min_width)
            else:
                width = (min_width + max_width) / 2
            edge_widths.append(width)
    else:
        edge_widths = [1.0]
    
    # Draw the graph
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', arrows=True, arrowsize=20,
                          connectionstyle="arc3,rad=0.1")
    
    nx.draw_networkx_nodes(G, pos, nodelist=up_interpretable_nodes, 
                          node_color='lightblue', node_size=1500, 
                          node_shape='o', alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, nodelist=down_interpretable_nodes,
                          node_color='lightcoral', node_size=1500,
                          node_shape='s', alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, nodelist=other_up_nodes,
                          node_color='lightgray', node_size=2000,
                          node_shape='D', alpha=0.9)
    
    nx.draw_networkx_nodes(G, pos, nodelist=other_down_nodes,
                          node_color='lightgray', node_size=2000, 
                          node_shape='D', alpha=0.9)
    
    # Add labels
    labels = {}
    for node in G.nodes():
        if "Other" in node:
            labels[node] = node
        else:
            if ":" in node:
                parts = node.split(": ")
                layer_info = parts[0]
                desc = parts[1]
                labels[node] = f"{layer_info}\n{desc}"
            else:
                labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    plt.title(f"Layer {up_layer} → Layer {down_layer} Edge Attribution Network\n(Top {top_k} Edges, Square Root Scaled)", 
              fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], s=150, c='lightblue', marker='o', label=f'L{up_layer} Interpretable'),
        plt.scatter([], [], s=150, c='lightcoral', marker='s', label=f'L{down_layer} Interpretable'), 
        plt.scatter([], [], s=200, c='lightgray', marker='D', label='Other Features'),
    ]
    
    legend = plt.legend(handles=legend_elements, loc='center', 
                       bbox_to_anchor=(0.5, 0.7), frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"edge_attribution_L{up_layer}_to_L{down_layer}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved diagram to {filepath}")
    
    plt.show()
    
    return G, edge_weights


def save_results(
    results: Dict[Tuple[LayerIndex, LayerIndex], Optional[EdgeResults]], 
    output_dir: Union[str, Path]
) -> Dict[str, Any]:
    """Save all analysis results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    summary: Dict[str, Any] = {
        'layer_pairs_analyzed': [],
        'total_edges_found': 0,
        'layer_pair_stats': {}
    }
    
    all_results: Dict[str, Any] = {}
    
    for (up_layer, down_layer), result in results.items():
        if result is not None:
            pair_key = f"L{up_layer}_to_L{down_layer}"
            summary['layer_pairs_analyzed'].append(pair_key)
            
            num_edges = len(result['all_connections'])
            summary['total_edges_found'] += num_edges
            summary['layer_pair_stats'][pair_key] = {
                'num_edges': num_edges,
                'strongest_edge': float(max([abs(x[2]) for x in result['all_connections']]) if result['all_connections'] else 0),
                'weakest_edge': float(min([abs(x[2]) for x in result['all_connections']]) if result['all_connections'] else 0)
            }
            
            # Save detailed results for this layer pair
            all_results[pair_key] = {
                'all_connections': [(int(up), int(down), float(val)) for up, down, val in result['all_connections']],
                'up_features': list(result['up_feats']),
                'down_features': list(result['down_feats']),
                'num_connections': num_edges
            }
    
    # Save summary
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Edge Attribution Analysis Across All Layer Combinations')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--protein', default='1PVGA', help='Protein to analyze')
    parser.add_argument('--layers', nargs='+', type=int, default=[4, 8, 12, 16, 20, 24, 28], 
                       help='Layers to analyze')
    parser.add_argument('--top-k', type=int, default=50, help='Number of top features per layer')
    parser.add_argument('--output-dir', default='./edge_attribution_results_all_layer_top2', 
                       help='Output directory for results')
    parser.add_argument('--max-pairs', type=int, default=None, 
                       help='Maximum number of layer pairs to analyze (for testing)')
    parser.add_argument('--skip-viz', action='store_true', 
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup models and data
    setup_data = setup_models_and_data(device, args.layers, args.protein)
    sequence_data = prepare_sequences(setup_data, device)
    
    # Perform causal ranking
    all_effects_sae_ALS, all_effects_err_ABLF = perform_causal_ranking(
        setup_data, sequence_data, device, args.layers
    )
    
    # Create layer caches
    clean_layer_caches, corr_layer_caches, clean_layer_errors, corr_layer_errors = create_layer_caches(
        setup_data, sequence_data, device, args.layers
    )
    
    # Load interpretable features
    interpretable_features = load_interpretable_features()
    
    # Analyze all layer pairs
    print(f"\n=== ANALYZING ALL LAYER PAIRS ===")
    results: Dict[Tuple[LayerIndex, LayerIndex], Optional[EdgeResults]] = {}
    layer_pairs: List[Tuple[LayerIndex, LayerIndex]] = list(itertools.combinations(args.layers, 2))
    
    if args.max_pairs:
        layer_pairs = layer_pairs[:args.max_pairs]
        print(f"Limiting analysis to first {args.max_pairs} pairs for testing")
    
    print(f"Will analyze {len(layer_pairs)} layer pairs")
    
    for i, (up_layer, down_layer) in enumerate(layer_pairs):
        print(f"\nProgress: {i+1}/{len(layer_pairs)}")
        
        try:
            result = analyze_layer_pair(
                up_layer, down_layer, setup_data, sequence_data, all_effects_sae_ALS,
                clean_layer_caches, clean_layer_errors, device, k=args.top_k
            )
            results[(up_layer, down_layer)] = result
            
            # Generate visualization if requested and edges were found
            if not args.skip_viz and result is not None:
                create_layer_network_diagram(
                    result['all_connections'], up_layer, down_layer,
                    interpretable_features, top_k=100, output_dir=args.output_dir
                )
                
        except Exception as e:
            print(f"Error analyzing {up_layer} → {down_layer}: {e}")
            results[(up_layer, down_layer)] = None
        
        # Clean up GPU memory
        cleanup_cuda()
    
    # Save results
    summary = save_results(results, args.output_dir)
    
    # Print final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Analyzed {len(layer_pairs)} layer pairs")
    print(f"Found edges in {len(summary['layer_pairs_analyzed'])} pairs")
    print(f"Total edges found: {summary['total_edges_found']}")
    print(f"Results saved to: {args.output_dir}")
    
    # Print top layer pairs by number of edges
    if summary['layer_pair_stats']:
        top_pairs = sorted(summary['layer_pair_stats'].items(), 
                          key=lambda x: x[1]['num_edges'], reverse=True)[:10]
        print(f"\nTop 10 layer pairs by number of edges:")
        for pair, stats in top_pairs:
            print(f"  {pair}: {stats['num_edges']} edges (strongest: {stats['strongest_edge']:.6f})")


if __name__ == "__main__":
    main()