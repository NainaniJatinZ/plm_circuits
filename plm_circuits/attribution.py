"""
attribution.py
~~~~~~~~~~~~~~
Integrated‑gradients attribution + contact‑map recovery metrics.
"""
from __future__ import annotations
from IPython import get_ipython
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from hook_manager import SAEHookProt
from helpers.utils import (
    clear_memory, 
    cleanup_cuda,
)

def activate_autoreload():
    try:
        ipython = get_ipython()
        if ipython is not None:
            ipython.magic("load_ext autoreload")
            ipython.magic("autoreload 2")
            print("In IPython")
            print("Set autoreload")
        else:
            print("Not in IPython")
    except NameError:
        print("`get_ipython` not available. This script is not running in IPython.")

# Call the function during script initialization
activate_autoreload()

# ---------- recovery metrics ----------------------------------------- #
def _norm_sum_mult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """∑(a·b) / ∑(a·a)  — just like in the notebook."""
    return (a * b).sum() / (a * a).sum()

def contact_recovery(
    preds: torch.Tensor,
    target: torch.Tensor,
    ss1_start: int,
    ss1_end: int,
    ss2_start: int,
    ss2_end: int,
) -> torch.Tensor:
    seg_pred = preds[ss1_start:ss1_end, ss2_start:ss2_end]
    seg_true = target[ss1_start:ss1_end, ss2_start:ss2_end]
    return _norm_sum_mult(seg_true, seg_pred)

# ---------- integrated gradients ------------------------------------- #
def integrated_gradients_sae(
    esm_model,
    sae_model,
    _patching_metric,
    clean_cache_LS: torch.Tensor,
    corr_cache_LS: torch.Tensor,
    clean_err: torch.Tensor,
    corr_err: torch.Tensor,
    *,
    batch_tokens,
    batch_mask,
    hook_layer: int,
    steps: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    IG over SAE latents **and** reconstruction error.

    Returns
    -------
    effect_sae_LS : torch.Tensor
        (L, S) attribution for each latent at each position.
    effect_err_BLF : torch.Tensor
        (B, L, F) attribution for error term.
    """
    device = clean_cache_LS.device
    effects_lat = []
    effects_err = []

    for alpha in np.linspace(0, 1, steps, endpoint=False):
        lat = (1 - alpha) * clean_cache_LS + alpha * corr_cache_LS
        err = (1 - alpha) * clean_err + alpha * corr_err
        lat.requires_grad_(True) 
        err.requires_grad_(True)

        sae_model.mean_error = err
        hook = SAEHookProt(
            sae_model,
            mask_BL=batch_mask.to(device),
            patch_latent_S=torch.arange(sae_model.w_enc.shape[1], device=device),
            patch_value=lat,
            use_mean_error=True,
        )
        h = esm_model.esm.encoder.layer[hook_layer].register_forward_hook(hook)

        out = esm_model.predict_contacts(batch_tokens, batch_mask)[0]
        h.remove()
        saes, esm_model = clear_memory([sae_model], esm_model)
        sae_model = saes[0]
        cleanup_cuda()

        score = _patching_metric(out)
        score.backward()
        print(f"ratio: {alpha}, score: {score}")

        # Keep gradients on the same device
        effects_lat.append((lat.grad * (corr_cache_LS - clean_cache_LS)).detach())
        effects_err.append((err.grad * (corr_err - clean_err)).detach())

    # Stack on GPU, then move to CPU if needed
    eff_lat = torch.stack(effects_lat).mean(dim=0)
    eff_err = torch.stack(effects_err).mean(dim=0)
    
    return eff_lat.cpu(), eff_err.cpu()  # Only move to CPU at the very end if needed

# ---------- top-k component selection -------------------------------- #
def topk_sae_err_pt(
    effects_sae_ALS: torch.Tensor,   # (A, L, S)
    effects_err_ALF: torch.Tensor,   # (A, L, F)
    k: int = 10,
    mode: str = "abs",              # "abs" | "pos" | "neg"
) -> List[Dict]:
    """
    Return the *k* most influential elements among

    • SAE latents   → (layer_idx, token_idx, latent_idx)
    • FFN-error sum → (layer_idx, token_idx)

    Parameters
    ----------
    effects_sae_ALS
        (A, L, S) tensor of SAE-latent attributions.
    effects_err_ALF
        (A, L, F) tensor of reconstruction-error attributions.
    k
        Number of entries to return.
    mode
        "abs" → rank by absolute magnitude (default).
        "pos" → rank by *positive* values only.
        "neg" → rank by *negative* values only (most negative).
    """

    if mode not in {"abs", "pos", "neg"}:
        raise ValueError(f"mode must be 'abs', 'pos' or 'neg' – got {mode!r}")

    A, L, S = effects_sae_ALS.shape

    # ------------------------------------------------------------------
    # 1) flatten tensors so we can rank them together -------------------
    sae_flat  = effects_sae_ALS.reshape(-1)                  # A·L·S
    err_flat  = effects_err_ALF.sum(dim=-1).reshape(-1)      # A·L  (collapse F)
    combined  = torch.cat([sae_flat, err_flat], dim=0)       # (A·L·S + A·L)

    # ------------------------------------------------------------------
    # 2) choose ranking criterion --------------------------------------
    if mode == "abs":
        ranking_tensor = combined.abs()
        largest_flag   = True
    elif mode == "pos":
        ranking_tensor = combined
        largest_flag   = True            # standard descending sort
    else:   # mode == "neg"
        ranking_tensor = combined
        largest_flag   = False           # ascending → most negative first

    # ------------------------------------------------------------------
    # 3) top-k according to the selected criterion ---------------------
    top_rank_vals, top_idx = torch.topk(ranking_tensor, k, largest=largest_flag, sorted=True)
    # Retrieve the *original* values (with sign) for output
    top_vals = combined[top_idx]

    # ------------------------------------------------------------------
    # 4) decode indices back to coordinates ----------------------------
    sae_len = sae_flat.numel()
    out = []
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        if idx < sae_len:                                   # SAE element
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
        else:                                               # ERR element
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

# ---------- single-layer performance functions ---------------------- #
def topk_performance_single_layer(
    esm_transformer, 
    saes, 
    all_effects_sae_ALS, 
    all_effects_err_ABLF, 
    k: int,
    mode: str,
    target_layer_idx: int,  # Which layer to focus on (index in main_layers)
    corr_layer_caches, 
    corr_layer_errors,
    clean_layer_errors, 
    layer_2_saelayer, 
    saelayer_2_layer,
    device, 
    clean_batch_tokens_BL, 
    clean_batch_mask_BL, 
    _patching_metric, 
    main_layers, 
    fixed_error: bool = False
):
    """
    Patch only the top-k components from a specific layer.
    
    Parameters
    ----------
    target_layer_idx : int
        Index of the target layer in main_layers (0-based)
    """
    # Get all top-k components across all layers
    topk_circuit_all = topk_sae_err_pt(all_effects_sae_ALS, all_effects_err_ABLF, k=k*len(main_layers), mode=mode)
    
    # Filter to only components from the target layer
    topk_circuit = [entry for entry in topk_circuit_all if entry["layer_idx"] == target_layer_idx][:k]
    
    if not topk_circuit:
        # No components from this layer in top-k, return clean performance
        with torch.no_grad():
            preds_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
        return _patching_metric(preds_LL), topk_circuit
    
    # Get the actual layer number
    target_layer_num = main_layers[target_layer_idx]
    
    # Create mask only for the target layer
    sae_model = saes[layer_2_saelayer[target_layer_num]]
    L, S = sae_model.feature_acts.shape if hasattr(sae_model, 'feature_acts') else (clean_batch_mask_BL.shape[1], 4096)
    F = corr_layer_errors[target_layer_num].shape[-1]
    
    sae_m = torch.ones((L, S), dtype=torch.bool, device=device)  # FALSE → keep clean
    err_m = torch.ones((1, L, F), dtype=torch.bool, device=device)
    
    # Apply patches for components from target layer
    for entry in topk_circuit:
        t = entry["token_idx"]
        if entry["type"] == "SAE":
            u = entry["latent_idx"]
            sae_m[t, u] = False
        else:  # "ERR"
            err_m[0, t, :] = False
    
    # Set up the hook for the target layer only
    sae_model = saes[layer_2_saelayer[target_layer_num]]
    corr_lat_LS = corr_layer_caches[target_layer_num]
    clean_err_LF = clean_layer_errors[target_layer_num]
    corr_err_LF = corr_layer_errors[target_layer_num]
    
    # Choose which values will overwrite the clean forward
    lat_patch_val = corr_lat_LS
    if fixed_error:
        sae_model.mean_error = clean_err_LF
    else:
        sae_model.mean_error = err_m * corr_err_LF + (~err_m) * clean_err_LF

    hook = SAEHookProt(
        sae=sae_model,
        mask_BL=clean_batch_mask_BL,
        patch_mask_BLS=sae_m.to(device),
        patch_value=lat_patch_val.to(device),
        use_mean_error=True,
    )
    handle = esm_transformer.esm.encoder.layer[target_layer_num].register_forward_hook(hook)

    # Forward pass & metric
    with torch.no_grad():
        preds_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
    rec = _patching_metric(preds_LL)

    # Clean up
    handle.remove()
    cleanup_cuda()
    return rec, topk_circuit

def plot_layer_performance_sweep(
    modes: List[str],
    start_k: int,
    end_k: int,
    step_k: int,
    mode2label: Dict[str, str],
    *,
    fixed_error: bool = False,
    corr_layer_caches,
    corr_layer_errors,
    clean_layer_errors,
    esm_transformer,
    saes,
    all_effects_sae_ALS,
    all_effects_err_ABLF,
    layer_2_saelayer,
    saelayer_2_layer,
    device,
    clean_batch_tokens_BL,
    clean_batch_mask_BL,
    _patching_metric,
    main_layers,
    clean_contact_recovery: float,
    **kwargs,
):
    """
    Plot performance curves for each layer separately.
    
    Each layer gets its own curve showing how performance changes 
    as we patch more of its top-k components.
    """
    ks = list(range(start_k, end_k + 1, step_k))
    
    plt.figure(figsize=(12, 8))
    
    # Plot curve for each layer
    for layer_idx, layer_num in enumerate(main_layers):
        print(f"\nProcessing layer {layer_num}...")
        
        layer_recs = {}
        for mode in modes:
            recs = []
            for k in ks:
                rec, _ = topk_performance_single_layer(
                    esm_transformer=esm_transformer,
                    saes=saes,
                    all_effects_sae_ALS=all_effects_sae_ALS,
                    all_effects_err_ABLF=all_effects_err_ABLF,
                    k=k,
                    mode=mode,
                    target_layer_idx=layer_idx,
                    corr_layer_caches=corr_layer_caches,
                    corr_layer_errors=corr_layer_errors,
                    clean_layer_errors=clean_layer_errors,
                    layer_2_saelayer=layer_2_saelayer,
                    saelayer_2_layer=saelayer_2_layer,
                    device=device,
                    clean_batch_tokens_BL=clean_batch_tokens_BL,
                    clean_batch_mask_BL=clean_batch_mask_BL,
                    _patching_metric=_patching_metric,
                    main_layers=main_layers,
                    fixed_error=fixed_error,
                )
                recs.append(rec.item())
            layer_recs[mode] = recs
            
            # Plot this layer's curve for this mode
            linestyle = '--' if mode == 'pos' else '-' if mode == 'neg' else ':'
            plt.plot(ks, recs, marker="o", linestyle=linestyle, 
                    label=f"Layer {layer_num} ({mode2label[mode]})", 
                    linewidth=2, markersize=4)
    
    # Reference lines
    plt.axhline(clean_contact_recovery, linestyle="--", color="black", 
                alpha=0.7, label="Clean baseline")
    plt.axhline(clean_contact_recovery * 0.6, linestyle=":", color="red", 
                alpha=0.7, label="0.6× clean")

    plt.xlabel("k (top-k components patched from single layer)")
    plt.ylabel("Contact recovery")
    plt.title("Layer-wise Performance: Top-k Components from Individual Layers")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return layer_recs
