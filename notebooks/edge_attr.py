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


def _compute_jvp_edge_v2(        # ← name kept so callers don’t change
    model,
    base_saes,
    upstream_sae,
    downstream_sae,
    token_list,
    upstream_features: List[int],
    downstream_features: List[int],
    clean_sae_cache,
    clean_error_cache,
    res_sae_effects,
    labels,
    device,
    *,
    use_mean_error: bool = True,
    edge_includes_loss_grad: bool = True,
    logstats: bool = False,
):
    """
    Reverse-mode **VJP** implementation of layer-to-layer edge attribution.

    For every pair (down_idx, up_idx) we accumulate

        Σ_{b,t}  ∂ y[b,t,down_idx] / ∂ x[b,t,up_idx]
        (or the gradient-weighted version if `edge_includes_loss_grad=True`).

    Compared with the old forward-mode JVP version this is:

      • robust (all ops in PyTorch have a VJP rule)  
      • slightly heavier on memory, but still much faster than
        finite-difference or zero-ablation baselines.
    """
    if logstats:
        print("[edge-attr-vjp] running reverse-mode edge attribution")

    up_hook,   down_hook  = upstream_sae.cfg.hook_name, downstream_sae.cfg.hook_name # TODO saes dont have cfg
    up_feats,  down_feats = upstream_features, downstream_features
    if not up_feats or not down_feats:
        if logstats:
            print(f"[edge-attr-vjp] skip {up_hook}->{down_hook} (no feats)")
        return None

    # ----------------------------------------------------------------------
    # 1. Baseline upstream activation that we will differentiate *through*
    # ----------------------------------------------------------------------
    up_base = (
        clean_sae_cache[up_hook]
        .detach()
        .clone()
        .to(device)
        .requires_grad_()                 # crucial for reverse-mode
    )

    # ----------------------------------------------------------------------
    # 2. Helper: forward pass that returns downstream SAE latents *attached*
    # ----------------------------------------------------------------------
    def _forward_fn() -> torch.Tensor:    # returns [B, L, S_down]
        _, saes_out = run_with_saes( # TODO add the hook for each 1. intervening on upstream, 2. recording downstream
            model,
            base_saes,
            token_list,
            calc_error=False,
            use_error=False,
            fake_activations=(upstream_sae.cfg.hook_layer, up_base),  # TODO saes dont have cfg
            use_mean_error=use_mean_error,
            cache_sae_activations=True,   # we need the graph intact
            no_detach=True,
        )
        feats = saes_out[downstream_sae.cfg.hook_layer].feature_acts
        if not feats.requires_grad:
            raise RuntimeError(
                "[edge-attr-vjp] downstream activations are detached; "
                "remove `.detach()` inside your SAE hook or clone with "
                "`.requires_grad_()` earlier in the graph."
            )
        return feats

    # ----------------------------------------------------------------------
    # 3. Single forward pass (re-used for every downstream feature)
    # ----------------------------------------------------------------------
    down_base = _forward_fn()             # [B,L,S_down]
    down_grad = (
        res_sae_effects[down_hook].to(device)
        if edge_includes_loss_grad else None
    )

    # Container: (down_idx, up_idx) → list[val]
    bucket: Dict[Tuple[int, int], List[torch.Tensor]] = {}

    # ----------------------------------------------------------------------
    # 4. Loop over downstream features (rows of the Jacobian)
    # ----------------------------------------------------------------------
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
            retain_graph=True,   # keep graph for next d_idx
            create_graph=False,  # we only need first-order grads
        )[0]                     # shape [B,L,S_up]

        # Accumulate entries we care about
        for u_idx in up_feats:
            val = grad_tensor[..., u_idx].sum()  # Σ_{b,t}
            if val.abs() < 1e-6:                 # keep/raise threshold as needed
                continue
            bucket.setdefault((d_idx, u_idx), []).append(val.detach().cpu())

        if logstats and (d_idx == down_feats[0] or d_idx % 10 == 0):
            print(f"[edge-attr-vjp] processed downstream idx {d_idx}")

    # ----------------------------------------------------------------------
    # 5. Assemble sparse COO tensor
    # ----------------------------------------------------------------------
    if not bucket:
        return None

    idxs, vals = zip(
        *[((d, u), torch.stack(v).mean()) for (d, u), v in bucket.items()]
    )
    idx_mat = torch.tensor(list(zip(*idxs)), dtype=torch.long)  # [2, N]
    val_mat = torch.stack(list(vals))                           # [N]

    edge_tensor = torch.sparse_coo_tensor(
        idx_mat,
        val_mat,
        size=(len(down_feats), len(up_feats)),
    ).coalesce()

    if logstats:
        nnz = edge_tensor._nnz()
        print(f"[edge-attr-vjp] finished – {nnz} non-zero entries")

    return edge_tensor

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

# %% Testing edge-attr-vjp for upstream layer 4 and downstream layer 8

up_layer = 4
down_layer = 8
print(layer_2_saelayer[up_layer], layer_2_saelayer[down_layer])
up_sae = saes[layer_2_saelayer[up_layer]]
down_sae = saes[layer_2_saelayer[down_layer]]

up_effects = all_effects_sae_ALS[layer_2_saelayer[up_layer]]
down_effects = all_effects_sae_ALS[layer_2_saelayer[down_layer]]


# %%

# getting the top 50 features for each layer

up_effect_flat = up_effects.reshape(-1)
down_effect_flat = down_effects.reshape(-1)

top_rank_vals_up, top_idx_up = torch.topk(up_effect_flat, k=50, largest=False, sorted=True)
top_rank_vals_down, top_idx_down = torch.topk(down_effect_flat, k=50, largest=False, sorted=True)

L, S = up_effects.shape

row_indices_up = top_idx_up // S
col_indices_up = top_idx_up % S

row_indices_down = top_idx_down // S
col_indices_down = top_idx_down % S

# print the top 5 for up and down

for i in range(5):
    print(f"Up: {top_rank_vals_up[i]:.6f}, {row_indices_up[i]}, {col_indices_up[i]}, {up_effects[row_indices_up[i], col_indices_up[i]]:.6f}")

for i in range(5):
    print(f"Down: {top_rank_vals_down[i]:.6f}, {row_indices_down[i]}, {col_indices_down[i]}, {down_effects[row_indices_down[i], col_indices_down[i]]:.6f}")



# %%
up_feats = set([col_indices_up[i].item() for i in range(len(col_indices_up))])
down_feats = set([col_indices_down[i].item() for i in range(len(col_indices_down))])
# up_feats = [col_indices_up[i].item() for i in range(len(col_indices_up))]
# down_feats = [col_indices_down[i].item() for i in range(len(col_indices_down))]

print(len(up_feats), len(down_feats))
print(up_feats, down_feats)

# %%

up_base = clean_layer_caches[up_layer].detach().clone().to(device).requires_grad_()
up_base_corr = corr_layer_caches[up_layer].detach().clone().to(device).requires_grad_()
patch_mask_LS = torch.ones((L, S), dtype=torch.bool, device=device)
up_error = clean_layer_errors[up_layer].to(device)
down_error = clean_layer_errors[down_layer].to(device)
# %%

def _forward_fn() -> torch.Tensor:    # returns [B, L, S_down]

    # up hook that puts patch activations 
    up_sae.mean_error = up_error
    up_hook = SAEHookProt(sae=up_sae, mask_BL=clean_batch_mask_BL, patch_mask_BLS=patch_mask_LS, patch_value=up_base, use_mean_error=True)

    # down hook that records downstream activations
    down_sae.mean_error = down_error
    down_hook = SAEHookProt(sae=down_sae, mask_BL=clean_batch_mask_BL, cache_latents=True, layer_is_lm=False, calc_error=True, use_error=True, no_detach=True)

    # register the hooks
    handle_up = esm_transformer.esm.encoder.layer[up_layer].register_forward_hook(up_hook)
    handle_down = esm_transformer.esm.encoder.layer[down_layer].register_forward_hook(down_hook)

    # run the forward pass
    # _, saes_out = run_with_saes( # TODO add the hook for each 1. intervening on upstream, 2. recording downstream
    #     model,
    #     base_saes,
    #     token_list,
    #     calc_error=False,
    #     use_error=False,
    #     fake_activations=(upstream_sae.cfg.hook_layer, up_base),  # TODO saes dont have cfg
    #     use_mean_error=use_mean_error,
    #     cache_sae_activations=True,   # we need the graph intact
    #     no_detach=True,
    # )
    # feats = saes_out[downstream_sae.cfg.hook_layer].feature_acts
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

# %%
from typing import List, Dict, Any, Optional, Tuple
# ----------------------------------------------------------------------
# 3. Single forward pass (re-used for every downstream feature)
# ----------------------------------------------------------------------
down_base = _forward_fn()             # [B,L,S_down]
down_grad = down_effects.to(device)

# Container: (down_idx, up_idx) → list[val]
bucket: Dict[Tuple[int, int], List[torch.Tensor]] = {}

# %%
logstats = True
# ----------------------------------------------------------------------
# 4. Loop over downstream features (rows of the Jacobian)
# ----------------------------------------------------------------------
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
        retain_graph=True,   # keep graph for next d_idx
        create_graph=False,  # we only need first-order grads
    )[0]                     # shape [B,L,S_up]

    # Accumulate entries we care about
    for u_idx in up_feats:
        val = grad_tensor[..., u_idx].sum()  # Σ_{b,t}
        if val.abs() < 1e-6:                 # keep/raise threshold as needed
            continue
        bucket.setdefault((d_idx, u_idx), []).append(val.detach().cpu())

    if logstats and (d_idx == list(down_feats)[0] or d_idx % 10 == 0):
        print(f"[edge-attr-vjp] processed downstream idx {d_idx}")

# ----------------------------------------------------------------------
# 5. Assemble sparse COO tensor
# ----------------------------------------------------------------------
if not bucket:
    print("No bucket")
else:
    idxs, vals = zip(
        *[((d, u), torch.stack(v).mean()) for (d, u), v in bucket.items()]
    )
    idx_mat = torch.tensor(list(zip(*idxs)), dtype=torch.long)  # [2, N]
    val_mat = torch.stack(list(vals))                           # [N]

    edge_tensor = torch.sparse_coo_tensor(
        idx_mat,
        val_mat,
        size=(len(down_feats), len(up_feats)),
    ).coalesce()

    if logstats:
        nnz = edge_tensor._nnz()
        print(f"[edge-attr-vjp] finished – {nnz} non-zero entries")

# %%

print(edge_tensor.values())

# %%
# Print edge tensor in a readable format
print("\n=== Edge Tensor Analysis ===")
print(f"Edge tensor shape: {edge_tensor.shape}")
print(f"Number of non-zero entries: {edge_tensor._nnz()}")

if edge_tensor._nnz() > 0:
    # Get the indices and values
    indices = edge_tensor.indices()  # [2, nnz] - [down_idx, up_idx]
    values = edge_tensor.values()    # [nnz]
    
    # Convert to lists for easier processing
    down_indices = indices[0].tolist()
    up_indices = indices[1].tolist()
    edge_values = values.tolist()
    
    # Group by upstream feature index
    from collections import defaultdict
    up_to_down = defaultdict(list)
    
    for i in range(len(down_indices)):
        down_idx = down_indices[i]
        up_idx = up_indices[i]
        val = edge_values[i]
        up_to_down[up_idx].append((down_idx, val))
    
    # Sort upstream indices for consistent output
    sorted_up_indices = sorted(up_to_down.keys())
    
    print(f"\nEdge connections (upstream -> downstream):")
    print("="*50)
    
    for up_idx in sorted_up_indices:
        connections = up_to_down[up_idx]
        # Sort connections by absolute value (strongest first)
        connections.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nUpstream feature {up_idx}:")
        for down_idx, val in connections:
            print(f"  -> Downstream {down_idx}: {val:.6f}")
    
    # Also show top connections overall
    print(f"\n\nTop 10 strongest connections overall:")
    print("="*50)
    all_connections = [(up_idx, down_idx, val) for up_idx, connections in up_to_down.items() 
                       for down_idx, val in connections]
    all_connections.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for i, (up_idx, down_idx, val) in enumerate(all_connections[:10]):
        print(f"{i+1:2d}. Up {up_idx} -> Down {down_idx}: {val:.6f}")
        
else:
    print("No edges found!")

# %%