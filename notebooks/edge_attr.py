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
from typing import List, Dict, Any, Optional, Tuple

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

# %% load json file 

with open('/project/pi_annagreen_umass_edu/jatin/plm_circuits/layer_latent_dict_2b61.json', 'r') as f:
    layer_latent_dict = json.load(f)
print(layer_latent_dict.keys())

# %%

l4_latents = {340: "FX'", 237: "FX'", 3788: "X'XI", 798: "D'XXGN", 1690: "X'XXF", 2277: "G", 2311: "X'XM", 3634: "X'XXXG", 1682: "PXXXXXX'", 3326: "H", "Error node":0, "Other":0}

l8_latents = {488:"AB_Hydrolase_fold", 2677:"FAD/NAD", 2775:"Transketolase", 2166:"DHFR", "Error node":0, "Other":0}

# %%





# %%

# %% Analyze top 5 edges and check for l8 latents

print(f"\n\n=== TOP 5 EDGES ANALYSIS ===")
print("="*50)

# Get all connections sorted by absolute value
all_connections = [(up_idx, down_idx, val) for up_idx, connections in up_to_down.items() 
                   for down_idx, val in connections]
all_connections.sort(key=lambda x: abs(x[2]), reverse=True)

# Print top 5 strongest connections
print("Top 5 strongest connections:")
for i, (up_idx, down_idx, val) in enumerate(all_connections[:5]):
    print(f"{i+1}. Up {up_idx} -> Down {down_idx}: {val:.6f}")

print(f"\n=== L8 LATENTS CHECK ===")
print("="*30)

# Check if any l8 latents appear in top 5
l8_latent_keys = set(l8_latents.keys()) - {"Error node", "Other"}  # Remove non-numeric keys
print(f"L8 latent indices to look for: {sorted(l8_latent_keys)}")

top_5_downstream = [down_idx for _, down_idx, _ in all_connections[:5]]
print(f"Top 5 downstream indices: {top_5_downstream}")

l8_in_top5 = []
for i, (up_idx, down_idx, val) in enumerate(all_connections[:5]):
    if down_idx in l8_latent_keys:
        l8_description = l8_latents[down_idx]
        l8_in_top5.append((i+1, up_idx, down_idx, val, l8_description))
        print(f"✓ Edge {i+1}: Up {up_idx} -> Down {down_idx} ({l8_description}): {val:.6f}")

if not l8_in_top5:
    print("✗ No L8 latents found in top 5 edges")
else:
    print(f"\nFound {len(l8_in_top5)} L8 latent(s) in top 5 edges!")

# Also check broader range (top 20) for context
print(f"\n=== L8 LATENTS IN TOP 20 EDGES ===")
print("="*40)

l8_in_top20 = []
for i, (up_idx, down_idx, val) in enumerate(all_connections[:20]):
    if down_idx in l8_latent_keys:
        l8_description = l8_latents[down_idx]
        l8_in_top20.append((i+1, up_idx, down_idx, val, l8_description))

if l8_in_top20:
    print(f"L8 latents found in top 20 edges:")
    for rank, up_idx, down_idx, val, description in l8_in_top20:
        print(f"  Rank {rank:2d}: Up {up_idx} -> Down {down_idx} ({description}): {val:.6f}")
else:
    print("No L8 latents found in top 20 edges")

# %%

# %% Analyze edges specifically between l4_latents and l8_latents

print(f"\n\n=== L4 → L8 INTERPRETABLE FEATURE CONNECTIONS ===")
print("="*60)

# Get the interpretable feature indices (exclude non-numeric keys)
l4_interpretable = {k: v for k, v in l4_latents.items() if isinstance(k, int)}
l8_interpretable = {k: v for k, v in l8_latents.items() if isinstance(k, int)}

print(f"L4 interpretable features: {len(l4_interpretable)}")
for idx, desc in l4_interpretable.items():
    print(f"  {idx}: {desc}")

print(f"\nL8 interpretable features: {len(l8_interpretable)}")
for idx, desc in l8_interpretable.items():
    print(f"  {idx}: {desc}")

# Look for connections between these specific features
l4_to_l8_connections = []

# Check if we have the edge data structure from the previous analysis
if 'up_to_down' in locals():
    print(f"\n=== SEARCHING FOR L4→L8 CONNECTIONS ===")
    print("-" * 50)
    
    # Iterate through upstream features (l4)
    for up_idx in up_to_down.keys():
        if up_idx in l4_interpretable:
            # Check downstream connections for this l4 feature
            connections = up_to_down[up_idx]
            for down_idx, val in connections:
                if down_idx in l8_interpretable:
                    l4_to_l8_connections.append({
                        'l4_idx': up_idx,
                        'l4_desc': l4_interpretable[up_idx],
                        'l8_idx': down_idx,
                        'l8_desc': l8_interpretable[down_idx],
                        'strength': val
                    })
    
    if l4_to_l8_connections:
        print(f"Found {len(l4_to_l8_connections)} connections between L4 and L8 interpretable features:")
        print()
        
        # Sort by absolute strength
        l4_to_l8_connections.sort(key=lambda x: abs(x['strength']), reverse=True)
        
        for i, conn in enumerate(l4_to_l8_connections):
            print(f"{i+1:2d}. L4 {conn['l4_idx']} ({conn['l4_desc']}) → L8 {conn['l8_idx']} ({conn['l8_desc']})")
            print(f"     Strength: {conn['strength']:.6f}")
            print()
            
        # Summary by L4 feature
        print("="*60)
        print("SUMMARY BY L4 FEATURE:")
        print("="*60)
        
        from collections import defaultdict
        l4_summary = defaultdict(list)
        for conn in l4_to_l8_connections:
            l4_summary[conn['l4_idx']].append(conn)
            
        for l4_idx in sorted(l4_summary.keys()):
            l4_desc = l4_interpretable[l4_idx]
            connections = l4_summary[l4_idx]
            connections.sort(key=lambda x: abs(x['strength']), reverse=True)
            
            print(f"\nL4 {l4_idx} ({l4_desc}) connects to:")
            for conn in connections:
                print(f"  → L8 {conn['l8_idx']} ({conn['l8_desc']}): {conn['strength']:.6f}")
                
        # Summary by L8 feature
        print("\n" + "="*60)
        print("SUMMARY BY L8 FEATURE:")
        print("="*60)
        
        l8_summary = defaultdict(list)
        for conn in l4_to_l8_connections:
            l8_summary[conn['l8_idx']].append(conn)
            
        for l8_idx in sorted(l8_summary.keys()):
            l8_desc = l8_interpretable[l8_idx]
            connections = l8_summary[l8_idx]
            connections.sort(key=lambda x: abs(x['strength']), reverse=True)
            
            print(f"\nL8 {l8_idx} ({l8_desc}) receives from:")
            for conn in connections:
                print(f"  ← L4 {conn['l4_idx']} ({conn['l4_desc']}): {conn['strength']:.6f}")
    
    else:
        print("❌ No direct connections found between L4 and L8 interpretable features")
        print("\nThis could mean:")
        print("1. These specific features weren't in the top-k selected for edge analysis")
        print("2. The connections exist but are too weak to be detected")
        print("3. The interpretable features don't directly connect (connections via intermediates)")
        
        # Check if any of our interpretable features were in the selected sets
        print(f"\nDEBUG INFO:")
        print(f"L4 features in analysis: {[idx for idx in l4_interpretable.keys() if idx in up_to_down.keys()]}")
        print(f"L8 features that appear as downstream: {[idx for idx in l8_interpretable.keys() if any(idx in [d for d, v in connections] for connections in up_to_down.values())]}")
        
else:
    print("❌ Edge analysis data not available. Please run the previous edge analysis first.")

# %%

# %% Analyze edges specifically between l4_latents and l8_latents with rankings

print(f"\n\n=== L4 → L8 INTERPRETABLE FEATURE CONNECTIONS WITH RANKINGS ===")
print("="*70)

# Get the interpretable feature indices (exclude non-numeric keys)
l4_interpretable = {k: v for k, v in l4_latents.items() if isinstance(k, int)}
l8_interpretable = {k: v for k, v in l8_latents.items() if isinstance(k, int)}

print(f"L4 interpretable features: {len(l4_interpretable)}")
for idx, desc in l4_interpretable.items():
    print(f"  {idx}: {desc}")

print(f"\nL8 interpretable features: {len(l8_interpretable)}")
for idx, desc in l8_interpretable.items():
    print(f"  {idx}: {desc}")

# Look for connections between these specific features
l4_to_l8_connections = []

# Check if we have the edge data structure from the previous analysis
if 'up_to_down' in locals() and 'all_connections' in locals():
    print(f"\n=== SEARCHING FOR L4→L8 CONNECTIONS IN EXISTING EDGE DATA ===")
    print("-" * 65)
    
    # Iterate through upstream features (l4)
    for up_idx in up_to_down.keys():
        if up_idx in l4_interpretable:
            # Check downstream connections for this l4 feature
            connections = up_to_down[up_idx]
            for down_idx, val in connections:
                if down_idx in l8_interpretable:
                    # Find the rank of this connection in all_connections
                    overall_rank = None
                    for rank, (up_all, down_all, val_all) in enumerate(all_connections, 1):
                        if up_all == up_idx and down_all == down_idx:
                            overall_rank = rank
                            break
                    
                    l4_to_l8_connections.append({
                        'l4_idx': up_idx,
                        'l4_desc': l4_interpretable[up_idx],
                        'l8_idx': down_idx,
                        'l8_desc': l8_interpretable[down_idx],
                        'strength': val,
                        'overall_rank': overall_rank
                    })
    
    if l4_to_l8_connections:
        # Sort by overall rank (best rank first)
        l4_to_l8_connections.sort(key=lambda x: x['overall_rank'] if x['overall_rank'] else float('inf'))
        
        total_edges = len(all_connections)
        
        print(f"Found {len(l4_to_l8_connections)} connections between L4 and L8 interpretable features:")
        print(f"Total edges in analysis: {total_edges}")
        print()
        
        # Display all connections with rankings
        for i, conn in enumerate(l4_to_l8_connections):
            local_rank = i + 1
            overall_rank = conn['overall_rank']
            percentile = (1 - overall_rank / total_edges) * 100 if overall_rank else 0
            
            print(f"LOCAL RANK {local_rank:2d} | GLOBAL RANK {overall_rank:4d}/{total_edges} (Top {percentile:5.1f}%)")
            print(f"   L4 {conn['l4_idx']} ({conn['l4_desc']}) → L8 {conn['l8_idx']} ({conn['l8_desc']})")
            print(f"   Strength: {conn['strength']:.8f}")
            print()
            
        # Summary by L4 feature with rankings
        print("="*70)
        print("🔍 BY L4 FEATURE (motif/pattern detectors) - WITH RANKINGS")
        print("="*70)
        
        from collections import defaultdict
        l4_summary = defaultdict(list)
        for conn in l4_to_l8_connections:
            l4_summary[conn['l4_idx']].append(conn)
            
        for l4_idx in sorted(l4_summary.keys()):
            l4_desc = l4_interpretable[l4_idx]
            connections = l4_summary[l4_idx]
            connections.sort(key=lambda x: x['overall_rank'] if x['overall_rank'] else float('inf'))
            
            print(f"\nL4 {l4_idx} ({l4_desc}) connects to:")
            for conn in connections:
                rank = conn['overall_rank']
                percentile = (1 - rank / total_edges) * 100 if rank else 0
                print(f"  → L8 {conn['l8_idx']} ({conn['l8_desc']}): {conn['strength']:.8f}")
                print(f"     [Global Rank {rank}/{total_edges}, Top {percentile:.1f}%]")
                
        # Summary by L8 feature with rankings
        print("\n" + "="*70)
        print("🎯 BY L8 FEATURE (domain detectors) - WITH RANKINGS")
        print("="*70)
        
        l8_summary = defaultdict(list)
        for conn in l4_to_l8_connections:
            l8_summary[conn['l8_idx']].append(conn)
            
        for l8_idx in sorted(l8_summary.keys()):
            l8_desc = l8_interpretable[l8_idx]
            connections = l8_summary[l8_idx]
            connections.sort(key=lambda x: x['overall_rank'] if x['overall_rank'] else float('inf'))
            
            print(f"\nL8 {l8_idx} ({l8_desc}) receives from:")
            for conn in connections:
                rank = conn['overall_rank']
                percentile = (1 - rank / total_edges) * 100 if rank else 0
                print(f"  ← L4 {conn['l4_idx']} ({conn['l4_desc']}): {conn['strength']:.8f}")
                print(f"     [Global Rank {rank}/{total_edges}, Top {percentile:.1f}%]")
        
        # Top interpretable connections summary
        print("\n" + "="*70)
        print("🏆 TOP INTERPRETABLE CONNECTIONS SUMMARY")
        print("="*70)
        
        top_5 = l4_to_l8_connections[:5]
        for i, conn in enumerate(top_5, 1):
            rank = conn['overall_rank']
            percentile = (1 - rank / total_edges) * 100
            print(f"{i}. L4 {conn['l4_idx']} '{conn['l4_desc']}' → L8 {conn['l8_idx']} '{conn['l8_desc']}'")
            print(f"   Global Rank: {rank}/{total_edges} (Top {percentile:.1f}%) | Strength: {conn['strength']:.8f}")
    
    else:
        print("❌ No direct connections found between L4 and L8 interpretable features")
        print("\nThis could mean:")
        print("1. These specific features weren't in the top-k selected for edge analysis")
        print("2. The connections exist but are too weak to be detected")
        print("3. The interpretable features don't directly connect (connections via intermediates)")
        
        # Check if any of our interpretable features were in the selected sets
        print(f"\nDEBUG INFO:")
        print(f"L4 features in analysis: {[idx for idx in l4_interpretable.keys() if idx in up_to_down.keys()]}")
        print(f"L8 features that appear as downstream: {[idx for idx in l8_interpretable.keys() if any(idx in [d for d, v in connections] for connections in up_to_down.values())]}")
        
else:
    print("❌ Edge analysis data not available. Please run the previous edge analysis first.")

# %%

# %% Create NetworkX diagram for top 200 edges with interpretable features

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

print(f"\n\n=== CREATING NETWORKX DIAGRAM FOR TOP 500 EDGES ===")
print("="*60)

if 'all_connections' in locals():
    # Get top 200 edges
    top_500_edges = all_connections[:100]
    print(f"Using top 100 edges out of {len(all_connections)} total edges")
    
    # Create mapping for nodes
    def map_node(node_idx, layer, interpretable_dict):
        """Map node to interpretable feature name or 'Other' group"""
        if node_idx in interpretable_dict:
            return f"L{layer}_{node_idx}: {interpretable_dict[node_idx]}"
        else:
            return f"Other L{layer}"
    
    # Process edges and aggregate weights for grouped nodes
    edge_weights = {}
    
    print("Processing edges...")
    
    # Get statistics for principled scaling
    all_strengths = [abs(strength) for _, _, strength in top_500_edges]
    max_strength = max(all_strengths)
    min_strength = min(all_strengths)
    
    print(f"Edge strength stats (top 200):")
    print(f"  Weakest edge: {min_strength:.8f}")
    print(f"  Strongest edge: {max_strength:.8f}")
    print(f"  Dynamic range: {max_strength/min_strength:.1f}x")
    
    # Principled scaling approach: Square root compression + normalization
    # This preserves relationships while compressing the dynamic range
    print(f"Using square root scaling to preserve relationships while improving visibility")
    
    for up_idx, down_idx, strength in top_500_edges:
        # Map nodes
        up_node = map_node(up_idx, 4, l4_interpretable)
        down_node = map_node(down_idx, 8, l8_interpretable)
        
        # Principled scaling: square root to compress range while preserving relationships
        abs_strength = abs(strength)
        sqrt_strength = np.sqrt(abs_strength)  # Compress dynamic range
        
        # Aggregate weights for the same edge (in case of grouping)
        edge_key = (up_node, down_node)
        if edge_key in edge_weights:
            edge_weights[edge_key] += sqrt_strength
        else:
            edge_weights[edge_key] = sqrt_strength
    
    print(f"Created {len(edge_weights)} unique edges after grouping")
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add edges with weights
    for (up_node, down_node), weight in edge_weights.items():
        G.add_edge(up_node, down_node, weight=weight)
    
    # Separate node types for styling
    l4_interpretable_nodes = [node for node in G.nodes() if node.startswith("L4_") and ":" in node]
    l8_interpretable_nodes = [node for node in G.nodes() if node.startswith("L8_") and ":" in node]
    other_l4_nodes = [node for node in G.nodes() if node == "Other L4"]
    other_l8_nodes = [node for node in G.nodes() if node == "Other L8"]
    
    print(f"\nNode counts:")
    print(f"  L4 interpretable: {len(l4_interpretable_nodes)}")
    print(f"  L8 interpretable: {len(l8_interpretable_nodes)}")
    print(f"  Other L4: {len(other_l4_nodes)}")
    print(f"  Other L8: {len(other_l8_nodes)}")
    
    # Create layout
    plt.figure(figsize=(16, 12))
    
    # Position nodes in two columns (L4 on left, L8 on right)
    pos = {}
    
    # L4 nodes on the left
    all_l4_nodes = l4_interpretable_nodes + other_l4_nodes
    for i, node in enumerate(all_l4_nodes):
        pos[node] = (0, len(all_l4_nodes) - i)
    
    # L8 nodes on the right  
    all_l8_nodes = l8_interpretable_nodes + other_l8_nodes
    for i, node in enumerate(all_l8_nodes):
        pos[node] = (3, len(all_l8_nodes) - i)
    
    # Principled edge width calculation
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    if weights:
        max_weight = max(weights)
        min_weight = min(weights)
        
        # Ensure the weakest edge (top 200) gets meaningful minimum width
        # and strongest edge gets reasonable maximum width
        min_width, max_width = 0.8, 5.0  # Slightly higher minimum for visibility
        
        edge_widths = []
        for w in weights:
            if max_weight > min_weight:
                # Linear scaling preserves exact relationships
                normalized = (w - min_weight) / (max_weight - min_weight)
                width = min_width + normalized * (max_width - min_width)
            else:
                width = (min_width + max_width) / 2  # All edges same weight
            edge_widths.append(width)
        
        print(f"Scaled edge weights: {min_weight:.6f} to {max_weight:.6f}")
        print(f"Edge width range: {min(edge_widths):.2f} to {max(edge_widths):.2f}")
        
        # Calculate original strength range for comparison  
        original_min = min_strength
        original_max = max_strength
        sqrt_min = np.sqrt(original_min)
        sqrt_max = np.sqrt(original_max)
        compression_ratio = (original_max/original_min) / (sqrt_max/sqrt_min)
        print(f"Dynamic range compression: {compression_ratio:.1f}x (from {original_max/original_min:.1f}x to {sqrt_max/sqrt_min:.1f}x)")
    else:
        edge_widths = [1.0]
    
    # Draw the graph
    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', arrows=True, arrowsize=20,
                          connectionstyle="arc3,rad=0.1")
    
    # Draw L4 interpretable nodes
    nx.draw_networkx_nodes(G, pos, nodelist=l4_interpretable_nodes, 
                          node_color='lightblue', node_size=1500, 
                          node_shape='o', alpha=0.8)
    
    # Draw L8 interpretable nodes  
    nx.draw_networkx_nodes(G, pos, nodelist=l8_interpretable_nodes,
                          node_color='lightcoral', node_size=1500,
                          node_shape='s', alpha=0.8)
    
    # Draw Other nodes with different style
    nx.draw_networkx_nodes(G, pos, nodelist=other_l4_nodes,
                          node_color='lightgray', node_size=2000,
                          node_shape='D', alpha=0.9)
    
    nx.draw_networkx_nodes(G, pos, nodelist=other_l8_nodes,
                          node_color='lightgray', node_size=2000, 
                          node_shape='D', alpha=0.9)
    
    # Add labels
    labels = {}
    for node in G.nodes():
        if "Other" in node:
            labels[node] = node
        else:
            # Shorten labels for interpretable nodes
            if ":" in node:
                parts = node.split(": ")
                layer_info = parts[0]  # L4_340
                desc = parts[1]        # FX'
                labels[node] = f"{layer_info}\n{desc}"
            else:
                labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    plt.title("Layer 4 → Layer 8 Edge Attribution Network\n(Top 100 Edges, Square Root Scaled)", 
              fontsize=16, fontweight='bold')
    
    # Add legend in the middle area (between node columns)
    legend_elements = [
        plt.scatter([], [], s=150, c='lightblue', marker='o', label='L4 Interpretable'),
        plt.scatter([], [], s=150, c='lightcoral', marker='s', label='L8 Interpretable'), 
        plt.scatter([], [], s=200, c='lightgray', marker='D', label='Other Features'),
    ]
    
    # Position legend in the middle of the network
    legend = plt.legend(handles=legend_elements, loc='center', 
                       bbox_to_anchor=(0.5, 0.7), frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add statistics in middle area, below legend
    stats_text = f"""Network Statistics:
Total Edges: {len(edge_weights)}

L4 Interpretable → L8 Interpretable: {sum(1 for (u,v) in edge_weights.keys() if ":" in u and ":" in v)}
L4 Interpretable → Other L8: {sum(1 for (u,v) in edge_weights.keys() if ":" in u and "Other" in v)}
Other L4 → L8 Interpretable: {sum(1 for (u,v) in edge_weights.keys() if "Other" in u and ":" in v)}
Other L4 → Other L8: {sum(1 for (u,v) in edge_weights.keys() if "Other" in u and "Other" in v)}

Scaling: √(strength) compression
Preserves relative relationships"""
    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             horizontalalignment='right', verticalalignment='top', fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.9, edgecolor='gray'))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print detailed edge information
    print(f"\n=== EDGE BREAKDOWN (Square Root Scaled) ===")
    print("="*50)
    print(f"Note: Edge weights are √(original_strength) to compress dynamic range")
    print(f"Original range: {min_strength:.8f} to {max_strength:.8f}")
    print(f"Scaled range: {np.sqrt(min_strength):.6f} to {np.sqrt(max_strength):.6f}")
    print(f"This preserves relative relationships while improving visual clarity")
    print()
    
    interpretable_to_interpretable = []
    interpretable_to_other = []
    other_to_interpretable = []
    other_to_other = []
    
    for (up_node, down_node), weight in edge_weights.items():
        if ":" in up_node and ":" in down_node:
            interpretable_to_interpretable.append((up_node, down_node, weight))
        elif ":" in up_node and "Other" in down_node:
            interpretable_to_other.append((up_node, down_node, weight))
        elif "Other" in up_node and ":" in down_node:
            other_to_interpretable.append((up_node, down_node, weight))
        else:
            other_to_other.append((up_node, down_node, weight))
    
    print(f"L4 Interpretable → L8 Interpretable: {len(interpretable_to_interpretable)} edges")
    for up, down, w in sorted(interpretable_to_interpretable, key=lambda x: x[2], reverse=True):
        original_strength = w * w  # w = sqrt(original), so original = w^2
        print(f"  {up} → {down}: √{original_strength:.8f} = {w:.6f}")
    
    print(f"\nL4 Interpretable → Other L8: {len(interpretable_to_other)} edges") 
    for up, down, w in sorted(interpretable_to_other, key=lambda x: x[2], reverse=True):
        original_strength = w * w
        print(f"  {up} → {down}: √{original_strength:.8f} = {w:.6f}")
        
    print(f"\nOther L4 → L8 Interpretable: {len(other_to_interpretable)} edges")
    for up, down, w in sorted(other_to_interpretable, key=lambda x: x[2], reverse=True):
        original_strength = w * w
        print(f"  {up} → {down}: √{original_strength:.8f} = {w:.6f}")
        
    print(f"\nOther L4 → Other L8: {len(other_to_other)} edges")
    for up, down, w in sorted(other_to_other, key=lambda x: x[2], reverse=True):
        original_strength = w * w
        print(f"  {up} → {down}: √{original_strength:.8f} = {w:.6f}")

else:
    print("❌ No edge data available. Please run the edge analysis first.")

# %%

# %% NetworkX Diagram Function with Flexible Scaling

def create_layer_network_diagram(
    all_connections,
    l4_interpretable,
    l8_interpretable,
    top_k=100,
    scaling_mode='normalized',  # 'normalized' or 'simple'
    simple_multiplier=2.0,
    min_width=0.8,
    max_width=5.0,
    figsize=(16, 12),
    title_suffix="",
    show_stats=True,
    stats_position='top_right'  # 'top_right', 'center', 'bottom_left'
):
    """
    Create a NetworkX diagram showing connections between interpretable features across layers.
    
    Parameters:
    -----------
    all_connections : list
        List of (up_idx, down_idx, strength) tuples sorted by strength
    l4_interpretable : dict
        Dictionary mapping L4 indices to interpretable descriptions
    l8_interpretable : dict
        Dictionary mapping L8 indices to interpretable descriptions
    top_k : int
        Number of top edges to include
    scaling_mode : str
        'normalized': square root scaling with min/max normalization for edge widths
        'simple': square root scaling with simple multiplier, no normalization
    simple_multiplier : float
        Multiplier for simple scaling mode
    min_width, max_width : float
        Edge width range for normalized mode
    figsize : tuple
        Figure size
    title_suffix : str
        Additional text for title
    show_stats : bool
        Whether to show network statistics
    stats_position : str
        Position for statistics box
        
    Returns:
    --------
    fig, ax, G : matplotlib figure, axes, networkx graph
    """
    
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"\n=== CREATING NETWORKX DIAGRAM FOR TOP {top_k} EDGES ===")
    print("="*60)
    
    # Get top k edges
    top_k_edges = all_connections[:top_k]
    print(f"Using top {top_k} edges out of {len(all_connections)} total edges")
    
    # Create mapping for nodes
    def map_node(node_idx, layer, interpretable_dict):
        """Map node to interpretable feature name or 'Other' group"""
        if node_idx in interpretable_dict:
            return f"L{layer}_{node_idx}: {interpretable_dict[node_idx]}"
        else:
            return f"Other L{layer}"
    
    # Get statistics for scaling
    all_strengths = [abs(strength) for _, _, strength in top_k_edges]
    max_strength = max(all_strengths)
    min_strength = min(all_strengths)
    
    print(f"Edge strength stats (top {top_k}):")
    print(f"  Weakest edge: {min_strength:.8f}")
    print(f"  Strongest edge: {max_strength:.8f}")
    print(f"  Dynamic range: {max_strength/min_strength:.1f}x")
    
    if scaling_mode == 'normalized':
        print(f"Using square root scaling with min/max normalization")
    else:
        print(f"Using simple square root scaling with {simple_multiplier}x multiplier")
    
    # Process edges and aggregate weights
    edge_weights = {}
    
    for up_idx, down_idx, strength in top_k_edges:
        # Map nodes
        up_node = map_node(up_idx, 4, l4_interpretable)
        down_node = map_node(down_idx, 8, l8_interpretable)
        
        # Apply scaling based on mode
        abs_strength = abs(strength)
        
        if scaling_mode == 'normalized':
            # Original: square root compression for normalization
            scaled_strength = np.sqrt(abs_strength)
        else:
            # Simple: square root with multiplier, no normalization
            scaled_strength = np.sqrt(abs_strength) * simple_multiplier
        
        # Aggregate weights for the same edge
        edge_key = (up_node, down_node)
        if edge_key in edge_weights:
            edge_weights[edge_key] += scaled_strength
        else:
            edge_weights[edge_key] = scaled_strength
    
    print(f"Created {len(edge_weights)} unique edges after grouping")
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add edges with weights
    for (up_node, down_node), weight in edge_weights.items():
        G.add_edge(up_node, down_node, weight=weight)
    
    # Separate node types for styling
    l4_interpretable_nodes = [node for node in G.nodes() if node.startswith("L4_") and ":" in node]
    l8_interpretable_nodes = [node for node in G.nodes() if node.startswith("L8_") and ":" in node]
    other_l4_nodes = [node for node in G.nodes() if node == "Other L4"]
    other_l8_nodes = [node for node in G.nodes() if node == "Other L8"]
    
    print(f"\nNode counts:")
    print(f"  L4 interpretable: {len(l4_interpretable_nodes)}")
    print(f"  L8 interpretable: {len(l8_interpretable_nodes)}")
    print(f"  Other L4: {len(other_l4_nodes)}")
    print(f"  Other L8: {len(other_l8_nodes)}")
    
    # Create layout
    fig, ax = plt.subplots(figsize=figsize)
    
    # Position nodes in two columns (L4 on left, L8 on right)
    pos = {}
    
    # L4 nodes on the left
    all_l4_nodes = l4_interpretable_nodes + other_l4_nodes
    for i, node in enumerate(all_l4_nodes):
        pos[node] = (0, len(all_l4_nodes) - i)
    
    # L8 nodes on the right  
    all_l8_nodes = l8_interpretable_nodes + other_l8_nodes
    for i, node in enumerate(all_l8_nodes):
        pos[node] = (3, len(all_l8_nodes) - i)
    
    # Calculate edge widths
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    if weights and scaling_mode == 'normalized':
        # Normalized mode: scale to min/max width range
        max_weight = max(weights)
        min_weight = min(weights)
        
        edge_widths = []
        for w in weights:
            if max_weight > min_weight:
                normalized = (w - min_weight) / (max_weight - min_weight)
                width = min_width + normalized * (max_width - min_width)
            else:
                width = (min_width + max_width) / 2
            edge_widths.append(width)
        
        print(f"Scaled edge weights: {min_weight:.6f} to {max_weight:.6f}")
        print(f"Edge width range: {min(edge_widths):.2f} to {max(edge_widths):.2f}")
        
        # Calculate compression stats
        original_min = min_strength
        original_max = max_strength
        sqrt_min = np.sqrt(original_min)
        sqrt_max = np.sqrt(original_max)
        compression_ratio = (original_max/original_min) / (sqrt_max/sqrt_min)
        print(f"Dynamic range compression: {compression_ratio:.1f}x (from {original_max/original_min:.1f}x to {sqrt_max/sqrt_min:.1f}x)")
        
    elif weights:
        # Simple mode: use weights directly as widths (with some scaling for visibility)
        edge_widths = [max(0.3, w) for w in weights]  # Minimum width 0.3
        print(f"Simple scaling: edge widths = √(strength) × {simple_multiplier}")
        print(f"Edge width range: {min(edge_widths):.2f} to {max(edge_widths):.2f}")
    else:
        edge_widths = [1.0]
    
    # Draw the graph
    # Draw edges first
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', arrows=True, arrowsize=20,
                          connectionstyle="arc3,rad=0.1")
    
    # Draw nodes with different styles
    nx.draw_networkx_nodes(G, pos, nodelist=l4_interpretable_nodes, 
                          node_color='lightblue', node_size=1500, 
                          node_shape='o', alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, nodelist=l8_interpretable_nodes,
                          node_color='lightcoral', node_size=1500,
                          node_shape='s', alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, nodelist=other_l4_nodes,
                          node_color='lightgray', node_size=2000,
                          node_shape='D', alpha=0.9)
    
    nx.draw_networkx_nodes(G, pos, nodelist=other_l8_nodes,
                          node_color='lightgray', node_size=2000, 
                          node_shape='D', alpha=0.9)
    
    # Add labels
    labels = {}
    for node in G.nodes():
        if "Other" in node:
            labels[node] = node
        else:
            # Shorten labels for interpretable nodes
            if ":" in node:
                parts = node.split(": ")
                layer_info = parts[0]  # L4_340
                desc = parts[1]        # FX'
                labels[node] = f"{layer_info}\n{desc}"
            else:
                labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Title
    scaling_desc = "Square Root Scaled" if scaling_mode == 'normalized' else f"Simple √ × {simple_multiplier}"
    title = f"Layer 4 → Layer 8 Edge Attribution Network\n(Top {top_k} Edges, {scaling_desc})"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], s=150, c='lightblue', marker='o', label='L4 Interpretable'),
        plt.scatter([], [], s=150, c='lightcoral', marker='s', label='L8 Interpretable'), 
        plt.scatter([], [], s=200, c='lightgray', marker='D', label='Other Features'),
    ]
    
    legend = plt.legend(handles=legend_elements, loc='center', 
                       bbox_to_anchor=(0.5, 0.7), frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add statistics if requested
    if show_stats:
        stats_text = f"""Network Statistics:
Total Edges: {len(edge_weights)}

L4 Interpretable → L8 Interpretable: {sum(1 for (u,v) in edge_weights.keys() if ":" in u and ":" in v)}
L4 Interpretable → Other L8: {sum(1 for (u,v) in edge_weights.keys() if ":" in u and "Other" in v)}
Other L4 → L8 Interpretable: {sum(1 for (u,v) in edge_weights.keys() if "Other" in u and ":" in v)}
Other L4 → Other L8: {sum(1 for (u,v) in edge_weights.keys() if "Other" in u and "Other" in v)}

Scaling: {scaling_desc}
{'Preserves relative relationships' if scaling_mode == 'normalized' else 'Simple multiplier approach'}"""
        
        # Position stats based on parameter
        if stats_position == 'center':
            h_align, v_align = 'center', 'center'
            bbox_anchor = (0.5, 0.4)
        elif stats_position == 'bottom_left':
            h_align, v_align = 'left', 'bottom'
            bbox_anchor = (0.02, 0.02)
        else:  # top_right
            h_align, v_align = 'right', 'top'
            bbox_anchor = (0.98, 0.98)
        
        plt.text(bbox_anchor[0], bbox_anchor[1], stats_text, transform=plt.gca().transAxes, 
                 horizontalalignment=h_align, verticalalignment=v_align, fontsize=9, 
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.9, edgecolor='gray'))
    
    plt.axis('off')
    plt.tight_layout()
    
    return fig, ax, G, edge_weights

# %% Function Usage Examples

print(f"\n\n=== FUNCTION USAGE EXAMPLES ===")
print("="*50)

# Example 1: Reproduce the current diagram (normalized scaling)
print("1. Reproduce current diagram with normalized scaling:")
print("""
fig, ax, G, weights = create_layer_network_diagram(
    all_connections=all_connections,
    l4_interpretable=l4_interpretable,
    l8_interpretable=l8_interpretable,
    top_k=100,
    scaling_mode='normalized',
    min_width=0.8,
    max_width=5.0,
    title_suffix="(Normalized)",
    stats_position='top_right'
)
plt.show()
""")

# Example 2: Simple scaling mode
print("\n2. Simple scaling mode (no normalization, just multiplier):")
print("""
fig, ax, G, weights = create_layer_network_diagram(
    all_connections=all_connections,
    l4_interpretable=l4_interpretable,
    l8_interpretable=l8_interpretable,
    top_k=100,
    scaling_mode='simple',
    simple_multiplier=3.0,
    title_suffix="(Simple)",
    stats_position='center'
)
plt.show()
""")

# Example 3: More edges with different positioning
print("\n3. More edges with center statistics:")
print("""
fig, ax, G, weights = create_layer_network_diagram(
    all_connections=all_connections,
    l4_interpretable=l4_interpretable,
    l8_interpretable=l8_interpretable,
    top_k=200,
    scaling_mode='normalized',
    title_suffix="(200 Edges)",
    stats_position='center'
)
plt.show()
""")

# Example 4: Custom styling
print("\n4. Custom styling:")
print("""
fig, ax, G, weights = create_layer_network_diagram(
    all_connections=all_connections,
    l4_interpretable=l4_interpretable,
    l8_interpretable=l8_interpretable,
    top_k=50,
    scaling_mode='simple',
    simple_multiplier=5.0,
    figsize=(20, 10),
    title_suffix="(Custom)",
    show_stats=False
)
plt.show()
""")

# Actually generate the same graph as current
print(f"\n=== GENERATING SAME GRAPH AS CURRENT ===")
print("="*45)

if 'all_connections' in locals():
    fig, ax, G, weights = create_layer_network_diagram(
        all_connections=all_connections,
        l4_interpretable=l4_interpretable,
        l8_interpretable=l8_interpretable,
        top_k=100,
        scaling_mode='normalized',
        min_width=0.8,
        max_width=5.0,
        title_suffix="(Function Generated)",
        stats_position='top_right'
    )
    plt.show()
    
    print(f"\n=== COMPARING SIMPLE SCALING MODE ===")
    print("="*40)
    
    fig2, ax2, G2, weights2 = create_layer_network_diagram(
        all_connections=all_connections,
        l4_interpretable=l4_interpretable,
        l8_interpretable=l8_interpretable,
        top_k=100,
        scaling_mode='simple',
        simple_multiplier=2.0,
        title_suffix="(Simple Scaling)",
        stats_position='center'
    )
    plt.show()
    
else:
    print("❌ No edge data available. Please run the edge analysis first.")

# %%
