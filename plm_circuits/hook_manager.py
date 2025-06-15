"""
hook_manager.py
~~~~~~~~~~~~~~~~
Forward‑hook utilities for SAE‑based interventions.

Usage example
-------------
>>> sae = load_sae_prot(...)
>>> hook = SAEHookProt(sae, mask)
>>> handle = model.esm.encoder.layer[layer_idx].register_forward_hook(hook)

Remove the hook when you are done:
>>> handle.remove()
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from helpers.sae_model_interprot import SparseAutoencoder


class SAEHookProt:
    """Patch / scale SAE latent activations during the forward pass.

    Parameters
    ----------
    sae
        The :class:`~interprot.sae_model.SparseAutoencoder` attached to
        the target ESM layer.
    mask
        ``BoolTensor`` of shape *(B, L)* – ``True`` for non‑padding
        tokens. You can obtain this via
        ``(batch_tokens != alphabet.padding_idx)``.
    patch_latent, patch_value
        1‑D index tensor and replacement values. Mutually exclusive with
        *patch_mask*.
    patch_mask
        Boolean tensor of the same shape as the SAE activations; ``True``
        entries will be replaced by *patch_value*.
    layer_lm
        Set to *True* if you are attaching the hook to the LM head rather
        than an encoder layer.
    cache_sae_acts
        Store the activations in ``sae.feature_acts`` for later use.
    calc_error, use_error, use_mean_error
        Compute the difference between the original and reconstructed
        hidden states (``sae.error_term``) and optionally add it back in
        *use_error* or add a *mean* error stored in
        ``sae.mean_error`` (*use_mean_error*).
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        mask_BL: torch.Tensor,
        *,
        patch_latent_S: Optional[torch.Tensor] = None,
        patch_value: Optional[torch.Tensor] = None,
        patch_mask_BLS: Optional[torch.Tensor] = None,
        layer_is_lm: bool = False,
        cache_latents: bool = False,
        calc_error: bool = False,
        use_error: bool = False,
        use_mean_error: bool = False,
    ) -> None:
        self.sae = sae.eval()
        self.mask_BL = mask_BL
        self.patch_latent_S = patch_latent_S
        self.patch_value = patch_value
        self.patch_mask_BLS = patch_mask_BLS
        self.layer_is_lm = layer_is_lm
        self.cache_latents = cache_latents
        self.calc_error = calc_error
        self.use_error = use_error
        self.use_mean_error = use_mean_error

    # ------------------------------------------------------------------ #
    def __call__(
        self,
        _module: nn.Module,
        _inputs: Tuple[torch.Tensor, ...],
        outputs,
    ):
        hidden_BLF = outputs if self.layer_is_lm else outputs[0]
        mod_BLF = hidden_BLF.clone()

        # 1) select valid tokens (no padding) -------------------------- #
        valid_BXF = mod_BLF[self.mask_BL]

        # 2) SAE encode / decode -------------------------------------- #
        x, mu_F, std_F = self.sae.LN(valid_BXF)
        f_BXS = (x - self.sae.b_pre) @ self.sae.w_enc + self.sae.b_enc

        if self.patch_value is not None:
            if self.patch_latent_S is not None:
                f_BXS[:, self.patch_latent_S] = self.patch_value
            elif self.patch_mask_BLS is not None:
                f_BXS[self.patch_mask_BLS] = self.patch_value[self.patch_mask_BLS]

        if self.cache_latents:
            self.sae.feature_acts = f_BXS.detach().cpu()

        topk_BXS = self.sae.topK_activation(f_BXS, self.sae.k)
        recon_BXF = (topk_BXS @ self.sae.w_dec + self.sae.b_pre) * std_F + mu_F

        mod_BLF[self.mask_BL] = recon_BXF

        # 3) error handling ------------------------------------------- #
        if self.calc_error:
            self.sae.error_term = hidden_BLF - mod_BLF
            if self.use_error:
                mod_BLF = mod_BLF + self.sae.error_term
        if self.use_mean_error:
            mod_BLF = mod_BLF + self.sae.mean_error

        return mod_BLF if self.layer_is_lm else (mod_BLF,) + outputs[1:]
