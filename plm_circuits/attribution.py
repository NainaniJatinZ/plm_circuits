"""
attribution.py
~~~~~~~~~~~~~~
Integrated‑gradients attribution + contact‑map recovery metrics.
"""
from __future__ import annotations
import torch
import numpy as np
from hook_manager import SAEHookProt
from helpers.utils import (
    clear_memory, 
    cleanup_cuda,
)

# ---------- recovery metrics ----------------------------------------- #
def _norm_sum_mult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """∑(a·b) / ∑(a·a)  — just like in the notebook."""
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
            mask_BL=batch_mask,
            patch_latent_S=torch.arange(sae_model.w_enc.shape[1]),
            patch_value=lat,
            use_mean_error=True,
        )
        h = esm_model.esm.encoder.layer[hook_layer].register_forward_hook(hook)

        out = esm_model.predict_contacts(batch_tokens, batch_mask)[0]
        h.remove()
        saes, esm_transformer = clear_memory(saes, esm_transformer)
        cleanup_cuda()

        score = _patching_metric(out)
        score.backward()

        effects_lat.append(lat.grad.detach().cpu() * (corr_cache_LS - clean_cache_LS))
        effects_err.append(err.grad.detach().cpu() * (corr_err - clean_err))

        

    eff_lat = torch.stack(effects_lat).mean(dim=0)
    eff_err = torch.stack(effects_err).mean(dim=0)
    return eff_lat, eff_err
