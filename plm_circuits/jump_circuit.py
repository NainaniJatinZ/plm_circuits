# %%
# model, saes 
# load seq, sse_pairs, clean_fl, corr_fl, tokens and mask for all three
# get clean and corr cache
# do attribution + ig for each layer with an sae 
# get top k nodes over all layers (including error nodes)

import json
from functools import partial
import torch
from huggingface_hub import hf_hub_download
from helpers.sae_model_interprot import SparseAutoencoder
from esm import FastaBatchedDataset, pretrained
from transformers import AutoTokenizer, EsmForMaskedLM
from safetensors.torch import load_file

# from .helpers.utils import (
#     load_esm,
#     load_sae_prot,
#     mask_flanks_segment,
#     cleanup_cuda,
#     patching_metric,
# )

# %% helpers 


def load_esm(model_size, WEIGHTS_DIR='/work/pi_jensen_umass_edu/jnainani_umass_edu/ESM_Interp/weights/',device: torch.device = torch.device("cuda")):
    """Return (model, alphabet, batch_converter) for the requested ESM‑2 size."""
    sizes = {6: "esm2_t6_8M_UR50D", 33: "esm2_t33_650M_UR50D", 36: "esm2_t36_3B_UR50D"}
    if model_size not in sizes:
        raise ValueError(f"Unsupported ESM‑2 size: {model_size}")
    
    esm_model_name = sizes[model_size]
    os.environ["TORCH_HOME"] = WEIGHTS_DIR  # Force PyTorch cache directory
    os.environ["HF_HOME"] = WEIGHTS_DIR
     
    _, esm2_alphabet = pretrained.load_model_and_alphabet(esm_model_name)
    esm_transformer = EsmForMaskedLM.from_pretrained(f"facebook/{esm_model_name}", cache_dir=WEIGHTS_DIR).to(device)
    batch_converter = esm2_alphabet.get_batch_converter()
    return esm_transformer, batch_converter, esm2_alphabet

def load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=24, device="cuda"):
    """Load a Sparse Autoencoder trained for a specific ESM layer."""
    checkpoint_path = hf_hub_download(
    repo_id="liambai/InterProt-ESM2-SAEs",
    filename=f"esm2_plm{ESM_DIM}_l{LAYER}_sae{SAE_DIM}.safetensors"
    )
    sae_model = SparseAutoencoder(ESM_DIM, SAE_DIM)
    sae_model.load_state_dict(load_file(checkpoint_path))
    sae_model.to(device)
    return sae_model

def mask_flanks_segment(seq, ss1_start, ss1_end, ss2_start, ss2_end, unmask_left_idxs, unmask_right_idxs):
    """
    seq: original protein sequence (string)
    patch_start, patch_end: the region [patch_start:patch_end] is fully unmasked
    unmask_left_idxs, unmask_right_idxs: sets or lists of indices that we keep unmasked on the flanks
    Returns: masked_seq (string) with <mask> everywhere except patch + chosen flank indices
    """
    L = len(seq)
    seq_list = ["<mask>"] * L
    
    # Always unmask patch
    # Unmask the patch region
    seq_list[ss1_start: ss1_end] = list(seq[ss1_start: ss1_end] )
    seq_list[ss2_start: ss2_end] = list(seq[ss2_start: ss2_end] )
    # Unmask chosen flank residues
    for i in unmask_left_idxs:
        seq_list[i] = seq[i]
    for i in unmask_right_idxs:
        seq_list[i] = seq[i]
    
    return "".join(seq_list)



# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

esm_transformer, batch_converter, esm2_alphabet = load_esm(33, device=device)

main_layers = [4, 8, 12, 16, 20, 24, 28]
saes = []
for layer in main_layers:
    sae_model = load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=layer, device=device)
    saes.append(sae_model)

layer_2_saelayer = {layer: layer_idx  for layer_idx, layer in enumerate(main_layers)}

with open('../data/full_seq_dict.json', "r") as json_file:
    seq_dict = json.load(json_file)

sse_dict = {"2B61A": [[182, 316]],"1PVGA": [[101, 202]]}
fl_dict = {"2B61A": [44, 43], "1PVGA": [65, 63]}


# %%

protein = "2B61A"

seq = seq_dict[protein]
full_seq_L = [(1, seq)]
position = sse_dict[protein][0]

ss1_start = position[0] - 5 
ss1_end = position[0] + 5 + 1 
ss2_start = position[1] - 5 
ss2_end = position[1] + 5 + 1 

_, _, batch_tokens_BL = batch_converter(full_seq_L)
batch_tokens_BL = batch_tokens_BL.to(device)
batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

with torch.no_grad():
    full_seq_contact_LL = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]

# %%
clean_fl = fl_dict[protein][0]
L = len(seq)
left_start = max(0, ss1_start - clean_fl)
left_end   = ss1_start
right_start= ss2_end
right_end = min(L, ss2_end + clean_fl)
unmask_left_idxs  = list(range(left_start, left_end))
unmask_right_idxs = list(range(right_start, right_end))

clean_seq_L = mask_flanks_segment(seq, ss1_start, ss1_end, ss2_start, ss2_end, unmask_left_idxs, unmask_right_idxs)
_, _, clean_batch_tokens_BL = batch_converter([(1, clean_seq_L)])
clean_batch_tokens_BL = clean_batch_tokens_BL.to(device)
clean_batch_mask_BL = (clean_batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

with torch.no_grad():
    clean_seq_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]

# %%

ori_mult_new = full_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end] * clean_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end]
ori_mult_ori = full_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end] * full_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end]
print(ori_mult_new.shape, ori_mult_ori.shape)
print(ori_mult_new.sum(), ori_mult_ori.sum())
print(ori_mult_new.sum() / ori_mult_ori.sum())


# %%


corr_fl = fl_dict[protein][1]
left_start = max(0, ss1_start - corr_fl)
left_end   = ss1_start
right_start= ss2_end
right_end = min(L, ss2_end + corr_fl)
unmask_left_idxs  = list(range(left_start, left_end))
unmask_right_idxs = list(range(right_start, right_end))

corr_seq_L = mask_flanks_segment(seq, ss1_start, ss1_end, ss2_start, ss2_end, unmask_left_idxs, unmask_right_idxs)
_, _, corr_batch_tokens_BL = batch_converter([(1, corr_seq_L)])
corr_batch_tokens_BL = corr_batch_tokens_BL.to(device)
corr_batch_mask_BL = (corr_batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

with torch.no_grad():
    corr_seq_contact_LL = esm_transformer.predict_contacts(corr_batch_tokens_BL, corr_batch_mask_BL)[0]

# %%

ori_mult_new = full_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end] * corr_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end]
ori_mult_ori = full_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end] * full_seq_contact_LL[ss1_start:ss1_end, ss2_start:ss2_end]
print(ori_mult_new.shape, ori_mult_ori.shape)
print(ori_mult_new.sum(), ori_mult_ori.sum())
print(ori_mult_new.sum() / ori_mult_ori.sum())

# %%

def patching_metric(contact_preds, orig_contact, ss1_start, ss1_end, ss2_start, ss2_end):

    seg_cross_contact = contact_preds[ss1_start:ss1_end, ss2_start:ss2_end]
    orig_contact_seg = orig_contact[ss1_start:ss1_end, ss2_start:ss2_end]
    return torch.sum(seg_cross_contact * orig_contact_seg) / torch.sum(orig_contact_seg * orig_contact_seg)

print(patching_metric(full_seq_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end))
print(patching_metric(clean_seq_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end))
print(patching_metric(corr_seq_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end))

# %% hook manager 

from typing import Optional, Tuple
import torch
import torch.nn as nn


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


def cleanup_cuda():
    import gc

    gc.collect()
    torch.cuda.empty_cache()




# %% single layer trial

_patching_metric = partial(
    patching_metric,
    orig_contact=full_seq_contact_LL,  # Specify this as a keyword arg
    ss1_start=ss1_start,
    ss1_end=ss1_end,
    ss2_start=ss2_start,
    ss2_end=ss2_end,
)

layer_idx = 4
sae_model = saes[layer_2_saelayer[layer_idx]]

hook = SAEHookProt(sae=sae_model, mask_BL=clean_batch_mask_BL, cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
with torch.no_grad():
    clean_seq_sae_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
cleanup_cuda()
handle.remove()
clean_cache_LS = sae_model.feature_acts
clean_err_cache_BLF = sae_model.error_term
clean_contact_recovery = patching_metric(clean_seq_sae_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end)
clean_recovery = _patching_metric(clean_seq_sae_contact_LL)

hook = SAEHookProt(sae=sae_model, mask_BL=corr_batch_mask_BL, cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
with torch.no_grad():
    corr_seq_sae_contact_LL = esm_transformer.predict_contacts(corr_batch_tokens_BL, corr_batch_mask_BL)[0]
cleanup_cuda()
handle.remove()
corr_cache_LS = sae_model.feature_acts
corr_err_cache_BLF = sae_model.error_term
corr_contact_recovery = patching_metric(corr_seq_sae_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end)
corr_recovery = _patching_metric(corr_seq_sae_contact_LL)
print(f"Layer {layer_idx}: Clean contact recovery: {clean_contact_recovery:.4f}, Corr contact recovery: {corr_contact_recovery:.4f}")
print(f"Layer {layer_idx}: Clean recovery: {clean_recovery:.4f}, Corr recovery: {corr_recovery:.4f}")

# %%
import numpy as np
from helpers.utils import clear_memory

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

        # Keep gradients on the same device
        effects_lat.append((lat.grad * (corr_cache_LS - clean_cache_LS)).detach())
        effects_err.append((err.grad * (corr_err - clean_err)).detach())

    # Stack on GPU, then move to CPU if needed
    eff_lat = torch.stack(effects_lat).mean(dim=0)
    eff_err = torch.stack(effects_err).mean(dim=0)
    
    return eff_lat.cpu(), eff_err.cpu()  # Only move to CPU at the very end if needed

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

# %%

neg_effects_LxS = - effect_sae_LS.view(-1)
topk_values_neg, topk_indices_neg = torch.topk(neg_effects_LxS, 25)
# Assuming final_effect_sae_LS has shape [420, 4096]
rows, cols = effect_sae_LS.shape

# Convert flattened indices back to 2D indices
row_indices = topk_indices_neg // cols  # Integer division to get row indices
col_indices = topk_indices_neg % cols   # Remainder to get column indices

# Combine row and column indices for better readability
topk_indices_2d = list(zip(row_indices.tolist(), col_indices.tolist()))

print("Row indices:", row_indices)
print("Column indices:", col_indices)
print("Top-k 2D indices:", topk_indices_2d)



# %%


all_effects_sae_ALS = []
all_effects_err_ABLF = []

with torch.no_grad():
    full_seq_contact_LL = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]
cleanup_cuda()

_patching_metric = partial(
    patching_metric,
    full_seq_contact_LL,
    ss1_start=ss1_start,
    ss1_end=ss1_end,
    ss2_start=ss2_start,
    ss2_end=ss2_end,
)

for layer_idx in main_layers:

    sae_model = saes[layer_2_saelayer[layer_idx]]

    hook = SAEHookProt(sae=sae_model, mask=clean_batch_mask_BL, cache_sae_acts=True, layer_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        clean_seq_sae_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
    cleanup_cuda()
    handle.remove()
    clean_cache_LS = sae_model.feature_acts
    clean_err_cache_BLF = sae_model.error_term
    clean_contact_recovery = _patching_metric(clean_seq_sae_contact_LL) 

    hook = SAEHookProt(sae=sae_model, mask=corr_batch_mask_BL, cache_sae_acts=True, layer_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        corr_seq_sae_contact_LL = esm_transformer.predict_contacts(corr_batch_tokens_BL, corr_batch_mask_BL)[0]
    cleanup_cuda()
    handle.remove()
    corr_cache_LS = sae_model.feature_acts
    corr_err_cache_BLF = sae_model.error_term
    corr_contact_recovery = _patching_metric(corr_seq_sae_contact_LL)
    print(f"Layer {layer_idx}: Clean contact recovery: {clean_contact_recovery:.4f}, Corr contact recovery: {corr_contact_recovery:.4f}")

    effect_sae_LS, effect_err_BLF = integrated_gradients_sae(
        esm_transformer,
        sae_model,
        _patching_metric,
        clean_cache_LS,
        corr_cache_LS,
        clean_err_cache_BLF,
        corr_err_cache_BLF,
        batch_tokens=clean_batch_tokens_BL,
        batch_mask=clean_batch_mask_BL,
        hook_layer=layer_idx,
    )

    all_effects_sae_ALS.append(effect_sae_LS)
    all_effects_err_ABLF.append(effect_err_BLF)






import torch
from typing import List, Dict

def topk_sae_err_pt(
    effects_sae_ALS: torch.Tensor,   # (A, L, S)
    effects_err_ALF: torch.Tensor,   # (A, L, F)
    k: int = 10,
) -> List[Dict]:
    """
    Return the top‑k absolute causal‑effect elements across
      • SAE latents   → (layer_idx, token_idx, latent_idx)
      • FFN‑error sum → (layer_idx, token_idx)

    Everything is done with torch ops; works on CPU or GPU.
    """
    A, L, S = effects_sae_ALS.shape
    # 1) |SAE| and |ERR|  (collapse F)
    sae_abs   = effects_sae_ALS.abs()                       # (A, L, S)
    err_sum   = effects_err_ALF.abs().sum(dim=-1)           # (A, L)

    # 2) flatten
    sae_flat  = sae_abs.reshape(-1)                         # A·L·S
    err_flat  = err_sum.reshape(-1)                         # A·L

    # 3) global top‑k
    combined  = torch.cat([sae_flat, err_flat], dim=0)      # (A·L·S + A·L)
    top_vals, top_idx = torch.topk(combined, k, largest=True, sorted=True)

    # 4) decode indices back to coordinates
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


A = 8; L = 420; S = 4096; F = 1280
effects_sae_ALS = torch.randn(A, L, S, device="cuda")   # or "cpu"
effects_err_ALF = torch.randn(A, L, F, device="cuda")

top10 = topk_sae_err_pt(effects_sae_ALS, effects_err_ALF, k=10)





