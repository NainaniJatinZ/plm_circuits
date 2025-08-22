# %%
# model, saes 
# load seq, sse_pairs, clean_fl, corr_fl, tokens and mask for all three
# get clean and corr cache
# do attribution + ig for each layer with an sae 
# get top k nodes over all layers (including error nodes)

import json
from functools import partial
import torch
import os
from huggingface_hub import hf_hub_download
from helpers.sae_model_interprot import SparseAutoencoder
from esm import FastaBatchedDataset, pretrained
from transformers import AutoTokenizer, EsmForMaskedLM
from safetensors.torch import load_file
import numpy as np
from typing import List, Dict, Optional, Tuple
import torch.nn as nn
import matplotlib.pyplot as plt
import collections

# %% helpers 

def clear_memory(saes, model, mask_bool=False):
    """
    Clears out the gradients from the SAEs and the main model
    to avoid accumulation or weird artifacts.

    Args:
        saes (List[SAE]): A list of SAE objects (or similarly structured modules).
        model (nn.Module): The main model whose gradients we want to clear.
        mask_bool (bool): Whether to also clear mask gradients if they exist.
    """
    for sae in saes:
        for param in sae.parameters():
            if param.grad is not None:
                param.grad = None
        if mask_bool and hasattr(sae, 'mask'):
            for param in sae.mask.parameters():
                if param.grad is not None:
                    param.grad = None
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    
    return saes, model

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
    """Load a Sparse Autoencoder trained for a specific ESM layer."""
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

def patching_metric(contact_preds, orig_contact, ss1_start, ss1_end, ss2_start, ss2_end):

    seg_cross_contact = contact_preds[ss1_start:ss1_end, ss2_start:ss2_end]
    orig_contact_seg = orig_contact[ss1_start:ss1_end, ss2_start:ss2_end]
    return torch.sum(seg_cross_contact * orig_contact_seg) / torch.sum(orig_contact_seg * orig_contact_seg)

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

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

esm_transformer, batch_converter, esm2_alphabet = load_esm(33, device=device) # 3b, 650m

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
print(patching_metric(full_seq_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end))
print(patching_metric(clean_seq_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end))
print(patching_metric(corr_seq_contact_LL, full_seq_contact_LL, ss1_start, ss1_end, ss2_start, ss2_end))

# %%

_patching_metric = partial(
    patching_metric,
    orig_contact=full_seq_contact_LL,  # Specify this as a keyword arg
    ss1_start=ss1_start,
    ss1_end=ss1_end,
    ss2_start=ss2_start,
    ss2_end=ss2_end,
)

# %%

all_effects_sae_ALS = []
all_effects_err_ABLF = []

for layer_idx in main_layers:

    sae_model = saes[layer_2_saelayer[layer_idx]]

    hook = SAEHookProt(sae=sae_model, mask_BL=clean_batch_mask_BL, cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        clean_seq_sae_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
    cleanup_cuda()
    handle.remove()
    clean_cache_LS = sae_model.feature_acts
    clean_err_cache_BLF = sae_model.error_term
    clean_contact_recovery = _patching_metric(clean_seq_sae_contact_LL) 

    hook = SAEHookProt(sae=sae_model, mask_BL=corr_batch_mask_BL, cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        corr_seq_sae_contact_LL = esm_transformer.predict_contacts(corr_batch_tokens_BL, corr_batch_mask_BL)[0]
    cleanup_cuda()
    handle.remove()
    corr_cache_LS = sae_model.feature_acts # corr_cache_LS[:, a]
    corr_err_cache_BLF = sae_model.error_term
    corr_contact_recovery = _patching_metric(corr_seq_sae_contact_LL)
    print(f"Layer {layer_idx}: Clean contact recovery: {clean_contact_recovery:.4f}, Corr contact recovery: {corr_contact_recovery:.4f}")

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

all_effects_sae_ALS = torch.stack(all_effects_sae_ALS)
all_effects_err_ABLF = torch.stack(all_effects_err_ABLF)

# %%

clean_contact_recovery
# %%

import torch
from typing import List, Dict

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


# %%

clean_layer_caches = {}
corr_layer_caches = {}
clean_layer_errors = {}
corr_layer_errors = {}

for layer_idx in main_layers:
    sae_model = saes[layer_2_saelayer[layer_idx]]
    hook = SAEHookProt(sae=sae_model, mask_BL=clean_batch_mask_BL, cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        clean_seq_sae_contact_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
    cleanup_cuda()
    handle.remove()
    print(f"layer {layer_idx}, score {_patching_metric(clean_seq_sae_contact_LL)}")
    clean_layer_caches[layer_idx] = sae_model.feature_acts
    clean_layer_errors[layer_idx] = sae_model.error_term

    hook = SAEHookProt(sae=sae_model, mask_BL=corr_batch_mask_BL, cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        corr_seq_sae_contact_LL = esm_transformer.predict_contacts(corr_batch_tokens_BL, corr_batch_mask_BL)[0]
    cleanup_cuda()
    handle.remove()
    print(f"layer {layer_idx}, score {_patching_metric(corr_seq_sae_contact_LL)}")
    corr_layer_caches[layer_idx] = sae_model.feature_acts
    corr_layer_errors[layer_idx] = sae_model.error_term

# %%

saelayer_2_layer = {v: k for k, v in layer_2_saelayer.items()}

# %%

def topk_performance(esm_transformer, 
                     saes, 
                     all_effects_sae_ALS, 
                     all_effects_err_ABLF, 
                     k, 
                     mode, 
                     corr_layer_caches, 
                     corr_layer_errors, 
                     clean_layer_errors, 
                     layer_2_saelayer = layer_2_saelayer, 
                     saelayer_2_layer = saelayer_2_layer,
                     device = device, 
                     clean_batch_tokens_BL = clean_batch_tokens_BL, 
                     clean_batch_mask_BL = clean_batch_mask_BL, 
                     _patching_metric = _patching_metric, 
                     main_layers = main_layers, 
                     fixed_error = False):
    topk_circuit = topk_sae_err_pt(all_effects_sae_ALS, all_effects_err_ABLF, k=k, mode=mode)
    
    layer_masks = {}
    for layer_idx in list(layer_2_saelayer.values()): #main_layers:
        # shapes: (L,S)  and  (1,L,F)
        L, S   = saes[layer_idx].feature_acts.shape
        F      = corr_layer_errors[saelayer_2_layer[layer_idx]].shape[-1]
        sae_m  = torch.ones((L, S), dtype=torch.bool, device=device)  # FALSE → keep clean
        err_m  = torch.ones((1, L, F), dtype=torch.bool, device=device)
        layer_masks[layer_idx] = {"sae": sae_m, "err": err_m}

    for entry in topk_circuit:                     # flip the selected positions to TRUE  (= patch)
        l = entry["layer_idx"]
        t = entry["token_idx"]
        if entry["type"] == "SAE":
            u = entry["latent_idx"]
            layer_masks[l]["sae"][t, u] = False
        else:                              # "ERR"
            layer_masks[l]["err"][0, t, :] = False

    handles = []

    for layer_idx in main_layers:
        sae_model = saes[layer_2_saelayer[layer_idx]]
        base_layer_idx = layer_2_saelayer[layer_idx]
        # --- fetch caches produced earlier for *this* layer ---------------
        corr_lat_LS  = corr_layer_caches[layer_idx]
        clean_err_LF = clean_layer_errors[layer_idx]
        corr_err_LF  = corr_layer_errors[layer_idx]

        m_sae = layer_masks[base_layer_idx]["sae"]             # (1,L,S)  bool
        m_err = layer_masks[base_layer_idx]["err"]             # (1,L,F)  bool

        # choose which values will overwrite the clean forward
        lat_patch_val = corr_lat_LS
        #err_patch_val = corr_err_LF
        if fixed_error:
            # mean-error logic (broadcast from (1,L,F) to (B,L,F))
            sae_model.mean_error = clean_err_LF # m_err * err_patch_val + (~m_err) * clean_err_LF
        else:
            sae_model.mean_error = m_err * corr_err_LF + (~m_err) * clean_err_LF

        h = SAEHookProt(
            sae          = sae_model,
            mask_BL      = clean_batch_mask_BL,            # B,L
            patch_mask_BLS = m_sae.to(device),                       # apply ONLY where m_sae==True
            patch_value    = lat_patch_val.to(device),               # L,S   (broadcast to B,L,S)
            use_mean_error = True,
        )
        hd = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(h)
        handles.append(hd)

    # ---- forward & metric ----------------------------------------------------
    with torch.no_grad():
        preds_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
    rec = _patching_metric(preds_LL)

    # ---- clean up ------------------------------------------------------------
    for hd in handles:
        hd.remove()
    cleanup_cuda()
    return rec, topk_circuit


rec, topk_circuit = topk_performance(esm_transformer, saes, all_effects_sae_ALS, all_effects_err_ABLF, k=2000, mode="neg", corr_layer_caches=corr_layer_caches, corr_layer_errors=corr_layer_errors, clean_layer_errors=clean_layer_errors)

print(rec)

# %%
# Visualisation helper ---------------------------------------------------

def plot_performance_sweep(
    modes: list[str],
    start_k: int,
    end_k: int,
    step_k: int,
    mode2label: dict[str, str],
    *,
    fixed_error: bool = False,
    corr_layer_caches=corr_layer_caches,
    corr_layer_errors=corr_layer_errors,
    clean_layer_errors=clean_layer_errors,
    esm_transformer=esm_transformer,
    saes=saes,
    all_effects_sae_ALS=all_effects_sae_ALS,
    all_effects_err_ABLF=all_effects_err_ABLF,
    clean_contact_recovery: float = clean_contact_recovery.item(),
    **topk_perf_kwargs,
):
    """Run *topk_performance* for a grid of *k* and plot one curve per *mode*.

    Parameters
    ----------
    modes
        List of strings ("abs", "pos", "neg") to evaluate.
    start_k, end_k, step_k
        Range passed to *range()*; *end_k* is inclusive.
    fixed_error, ...
        Forwarded to *topk_performance*.
    clean_contact_recovery
        Reference value for horizontal lines.
    topk_perf_kwargs
        Any extra arguments forwarded to *topk_performance* (e.g. batch tensors).
    """

    ks = list(range(start_k, end_k + 1, step_k))

    plt.figure(figsize=(7, 4))

    for mode in modes:
        recs = []
        for k in ks:
            rec, _ = topk_performance(
                esm_transformer,
                saes,
                all_effects_sae_ALS,
                all_effects_err_ABLF,
                k=k,
                mode=mode,
                corr_layer_caches=corr_layer_caches,
                corr_layer_errors=corr_layer_errors,
                clean_layer_errors=clean_layer_errors,
                fixed_error=fixed_error,
                **topk_perf_kwargs,
            )
            recs.append(rec.item())
        print(recs)
        plt.plot(ks, recs, marker="o", label=mode2label[mode])

    # reference lines ----------------------------------------------------
    plt.axhline(clean_contact_recovery, linestyle="--", color="black", label="clean")
    plt.axhline(clean_contact_recovery * 0.6, linestyle=":", color="black", label="0.6× clean")

    plt.xlabel("k (top-k elements patched)")
    plt.ylabel("contact-recovery")
    plt.title("Recovery vs K component circuit for different topK strategies")
    plt.legend()
    plt.tight_layout()
    plt.show()

mode2label = {"abs": "Absolute", "pos": "Positive", "neg": "Negative"}

# Example usage ----------------------------------------------------------
plot_performance_sweep(["abs", "pos", "neg"], 0, 5000, 500, fixed_error=False, mode2label=mode2label)

# %%

rec, topk_circuit = topk_performance(esm_transformer, saes, all_effects_sae_ALS, all_effects_err_ABLF, k=2000, mode="neg", corr_layer_caches=corr_layer_caches, corr_layer_errors=corr_layer_errors, clean_layer_errors=clean_layer_errors, fixed_error=False)

print(rec)

# %%


import collections
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# 1) tally per-layer counts
sae_per_layer  = collections.Counter()
err_per_layer  = collections.Counter()
unique_tokens  = set()

for entry in topk_circuit:
    layer = entry["layer_idx"]         # 0-based index in all_effects_* tensors
    tok   = entry["token_idx"]
    if entry["type"] == "SAE":
        sae_per_layer[layer] += 1
    else:                              # "ERR"
        err_per_layer[layer] += 1
    unique_tokens.add(tok)

layers_sorted = sorted(set(list(sae_per_layer) + list(err_per_layer)))
sae_counts = [sae_per_layer[l] for l in layers_sorted]
err_counts = [err_per_layer[l] for l in layers_sorted]

# ------------------------------------------------------------------
# 2) stacked-bar plot with *actual* layer numbers (4,8,…) -------------

# convert internal 0-based indices → real encoder layer numbers
actual_layers = [saelayer_2_layer[l] for l in layers_sorted]

x = np.arange(len(actual_layers))

plt.figure(figsize=(8,4))
plt.bar(x, sae_counts, label="SAE", color="#1f77b4")
plt.bar(x, err_counts, bottom=sae_counts, label="ERR", color="#ff7f0e")

plt.xticks(x, actual_layers)
plt.xlabel("Encoder layer number")
plt.ylabel("Component count")
plt.title(f"Component distribution in k={len(topk_circuit)} circuit")
plt.legend()
plt.grid(True, axis="y", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()


# %%
# ------------------------------------------------------------------
# 3) token-frequency plot along the sequence -------------------------

seq_len = len(seq)  # length of the current protein sequence
token_counter = collections.Counter(entry["token_idx"] for entry in topk_circuit)

freq = [token_counter.get(i+1, 0) for i in range(seq_len)]

plt.figure(figsize=(10,3))
plt.bar(range(seq_len), freq, color="#2ca02c")
clean_fl = fl_dict[protein][0]
L = len(seq)
left_start = max(0, ss1_start - clean_fl)
left_end   = ss1_start
right_start= ss2_end
right_end = min(L, ss2_end + clean_fl)
# shaded regions -------------------------------------------------------
# Convert residue indices to bar positions (0-based bars).
def _bar_idx(pos):
    # if freq was built with i+1, shift indexes; otherwise this is no-op
    return pos - 1 if token_counter.get(0) is None else pos

# SSE boxes
plt.axvspan(_bar_idx(ss1_start), _bar_idx(ss1_end) - 1, color="#d62728", alpha=0.3, label="SSE1")
plt.axvspan(_bar_idx(ss2_start), _bar_idx(ss2_end) - 1, color="#1f77b4", alpha=0.3, label="SSE2")

# Clean-flank boxes
plt.axvspan(_bar_idx(left_start), _bar_idx(ss1_start) - 1, color="#9467bd", alpha=0.15, label="clean flank")
plt.axvspan(_bar_idx(ss2_end), _bar_idx(right_end) - 1, color="#9467bd", alpha=0.15)

plt.xlabel("Sequence position (0-based)")
plt.ylabel("Component count")
plt.title("Per-token frequency of circuit components")
plt.tight_layout()
plt.legend()
plt.show()

print(f"Unique sequence positions involved: {len(unique_tokens)}")
# %%

print(len([1 for x in topk_circuit if x["type"] == "SAE"]))
print(len([1 for x in topk_circuit if x["type"] == "ERR"]))


# %%

def topk_ablation(esm_transformer, 
                     saes, 
                     all_effects_sae_ALS, 
                     all_effects_err_ABLF, 
                     k, 
                     mode, 
                     corr_layer_caches, 
                     corr_layer_errors, 
                     clean_layer_errors, 
                     layer_2_saelayer = layer_2_saelayer, 
                     saelayer_2_layer = saelayer_2_layer,
                     device = device, 
                     clean_batch_tokens_BL = clean_batch_tokens_BL, 
                     clean_batch_mask_BL = clean_batch_mask_BL, 
                     _patching_metric = _patching_metric, 
                     main_layers = main_layers, 
                     fixed_error = False):
    topk_circuit = topk_sae_err_pt(all_effects_sae_ALS, all_effects_err_ABLF, k=k, mode=mode)
    
    layer_masks = {}
    for layer_idx in list(layer_2_saelayer.values()): #main_layers:
        # shapes: (L,S)  and  (1,L,F)
        L, S   = saes[layer_idx].feature_acts.shape
        F      = corr_layer_errors[saelayer_2_layer[layer_idx]].shape[-1]
        sae_m  = torch.zeros((L, S), dtype=torch.bool, device=device)  # FALSE → keep clean
        err_m  = torch.zeros((1, L, F), dtype=torch.bool, device=device)
        layer_masks[layer_idx] = {"sae": sae_m, "err": err_m}

    for entry in topk_circuit:                     # flip the selected positions to TRUE  (= patch)
        l = entry["layer_idx"]
        t = entry["token_idx"]
        if entry["type"] == "SAE":
            u = entry["latent_idx"]
            layer_masks[l]["sae"][t, u] = True
        else:                              # "ERR"
            layer_masks[l]["err"][0, t, :] = True

    handles = []

    for layer_idx in main_layers:
        sae_model = saes[layer_2_saelayer[layer_idx]]
        base_layer_idx = layer_2_saelayer[layer_idx]
        # --- fetch caches produced earlier for *this* layer ---------------
        corr_lat_LS  = corr_layer_caches[layer_idx]
        clean_err_LF = clean_layer_errors[layer_idx]
        corr_err_LF  = corr_layer_errors[layer_idx]

        m_sae = layer_masks[base_layer_idx]["sae"]             # (1,L,S)  bool
        m_err = layer_masks[base_layer_idx]["err"]             # (1,L,F)  bool

        # choose which values will overwrite the clean forward
        lat_patch_val = corr_lat_LS
        #err_patch_val = corr_err_LF
        if fixed_error:
            # mean-error logic (broadcast from (1,L,F) to (B,L,F))
            sae_model.mean_error = clean_err_LF # m_err * err_patch_val + (~m_err) * clean_err_LF
        else:
            sae_model.mean_error = m_err * corr_err_LF + (~m_err) * clean_err_LF

        h = SAEHookProt(
            sae          = sae_model,
            mask_BL      = clean_batch_mask_BL,            # B,L
            patch_mask_BLS = m_sae.to(device),                       # apply ONLY where m_sae==True
            patch_value    = lat_patch_val.to(device),               # L,S   (broadcast to B,L,S)
            use_mean_error = True,
        )
        hd = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(h)
        handles.append(hd)

    # ---- forward & metric ----------------------------------------------------
    with torch.no_grad():
        preds_LL = esm_transformer.predict_contacts(clean_batch_tokens_BL, clean_batch_mask_BL)[0]
    rec = _patching_metric(preds_LL)

    # ---- clean up ------------------------------------------------------------
    for hd in handles:
        hd.remove()
    cleanup_cuda()
    return rec, topk_circuit

def plot_ablation_sweep(
    modes: list[str],
    start_k: int,
    end_k: int,
    step_k: int,
    mode2label: dict[str, str],
    *,
    fixed_error: bool = False,
    corr_layer_caches=corr_layer_caches,
    corr_layer_errors=corr_layer_errors,
    clean_layer_errors=clean_layer_errors,
    esm_transformer=esm_transformer,
    saes=saes,
    all_effects_sae_ALS=all_effects_sae_ALS,
    all_effects_err_ABLF=all_effects_err_ABLF,
    clean_contact_recovery: float = clean_contact_recovery.item(),
    **topk_ablation_kwargs,
):
    """Run *topk_ablation* for a grid of *k* and plot one curve per *mode*.

    Parameters
    ----------
    modes
        List of strings ("abs", "pos", "neg") to evaluate.
    start_k, end_k, step_k
        Range passed to *range()*; *end_k* is inclusive.
    fixed_error, ...
        Forwarded to *topk_ablation*.
    clean_contact_recovery
        Reference value for horizontal lines.
    topk_ablation_kwargs
        Any extra arguments forwarded to *topk_ablation* (e.g. batch tensors).
    """

    ks = list(range(start_k, end_k + 1, step_k))

    plt.figure(figsize=(7, 4))

    for mode in modes:
        recs = []
        for k in ks:
            rec, _ = topk_ablation(
                esm_transformer,
                saes,
                all_effects_sae_ALS,
                all_effects_err_ABLF,
                k=k,
                mode=mode,
                corr_layer_caches=corr_layer_caches,
                corr_layer_errors=corr_layer_errors,
                clean_layer_errors=clean_layer_errors,
                fixed_error=fixed_error,
                **topk_ablation_kwargs,
            )
            recs.append(rec.item())
        print(f"{mode} ablation scores:", recs)
        plt.plot(ks, recs, marker="o", label=mode2label[mode])

    # reference lines ----------------------------------------------------
    plt.axhline(clean_contact_recovery, linestyle="--", color="black", label="clean")
    plt.axhline(clean_contact_recovery * 0.6, linestyle=":", color="black", label="0.6× clean")

    plt.xlabel("k (top-k elements ablated)")
    plt.ylabel("contact-recovery")
    plt.title("Recovery vs K component ablation for different topK strategies")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage of ablation sweep
mode2label = {"abs": "Absolute", "pos": "Positive", "neg": "Negative"}
plot_ablation_sweep(["abs", "pos", "neg"], 0, 1000, 100, fixed_error=True, mode2label=mode2label)

rec, topk_circuit = topk_ablation(esm_transformer, saes, all_effects_sae_ALS, all_effects_err_ABLF, k=200, mode="neg", corr_layer_caches=corr_layer_caches, corr_layer_errors=corr_layer_errors, clean_layer_errors=clean_layer_errors, fixed_error=True)

print(rec)

# %%

200/8
# %%
