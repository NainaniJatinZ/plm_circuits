#!/usr/bin/env python3
"""
Reconcile AUROC and MWU on a common dataset with length-matched negatives.

You provide a callback that maps (sequence_string, layer, unit) -> 1D array
of per-token activations. The script will aggregate to a per-protein score
(max / top-q% mean / log-sum-exp / mean), then compute:
- AUROC + stratified bootstrap 95% CI
- Mann–Whitney U (one-sided, greater) p-value
- Verify AUC == U / (n_pos * n_neg)
- BH–FDR q-values across the tested (latent,domain,aggregator) rows
- Length correlation diagnostics
- Plots per-aggregator

Outputs saved into ./reconcile_out/
"""

import os, json, math, pathlib, sys
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple, Iterable
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

# -------------------------- User configuration --------------------------

import sys
import os
# Add plm_circuits to path - handle both when run from notebooks/ and from root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
plm_circuits_path = os.path.join(parent_dir, 'plm_circuits')
if os.path.exists(plm_circuits_path):
    sys.path.append(plm_circuits_path)
else:
    # Fallback for when run from different directory
    sys.path.append('../plm_circuits')

import torch
import numpy as np
import json
import time
import os
import random
import pandas as pd
import string
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz
from Bio import SeqIO
import pathlib
import heapq
from collections import namedtuple
import pickle
import logomaker
import matplotlib.pyplot as plt
# Import utility functions
from helpers.utils import load_esm, load_sae_prot, cleanup_cuda
from hook_manager import SAEHookProt

# Setup device and load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Load ESM-2 model
    print("Loading ESM-2 model...")
    esm_transformer, batch_converter, esm2_alphabet = load_esm(33, device=device)
    print("ESM-2 model loaded successfully")

    # Load SAEs for multiple layers
    main_layers = [4, 8, 12, 16, 20, 24, 28]
    saes = []
    print("Loading SAE models...")
    for layer in main_layers:
        print(f"  Loading SAE for layer {layer}...")
        sae_model = load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=layer, device=device)
        saes.append(sae_model)
    print("All SAE models loaded successfully")

    layer_2_saelayer = {layer: layer_idx for layer_idx, layer in enumerate(main_layers)}
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# Path to Swiss-Prot/Reviewed FASTA (negatives sampling pool)
FASTA_SPROT = "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/uniprot_sprot.fasta"

# Full targets list
TARGETS = [
    # metx 
    (8, 488,  "IPR029058", "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR029058.fasta"),
    (8, 2677, "IPR036188",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_circuits/data/protein-matching-IPR036188.fasta"),
    (8, 2775, "IPR009014",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR009014.fasta"),
    (8, 2166, "IPR024072",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR024072.fasta"),
    (12, 2112, "IPR029058",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR029058.fasta"),
    (12, 3536, "IPR029063",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR029063.fasta"),
    (12, 1256, "IPR016181",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR016181.fasta"),
    (12, 2797, "IPR013785",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR013785.fasta"),
    (12, 3794, "IPR029063",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR029063.fasta"),
    (12, 3035, "IPR036322",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR036322.fasta"),

    # top2
    (12, 1082, "PF00867",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-PF00867.fasta"),
    (12, 2472, "IPR036961",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR036961.fasta"),
    (12, 3943, "IPR036890",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR036890.fasta"),
    (12, 1796, "IPR036890",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR036890.fasta"),
    (12, 1204, "IPR036890",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR036890.fasta"),
    (12, 1145, "IPR036890",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR036890.fasta"),
    (16, 1166, "PF13589",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-PF13589.fasta"),
    (16, 3077, "IPR036890",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR036890.fasta"),
    (16, 1353, "IPR036890",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR036890.fasta"),
    (16, 1597, "IPR036890",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR036890.fasta"),
    (16, 1814, "IPR036890",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-IPR036890.fasta"),
    (16, 3994, "PF13589",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-PF13589.fasta"),
    (20, 2311, "PF13589",  "/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_interp/data/paper/protein-matching-PF13589.fasta"),
]

# Aggregators to evaluate
AGGREGATORS = ["max", "topq", "mean", "topk"]

# Aggregator hyperparams
TOPQ = 0.01     # top 1%
LSE_TAU = 2.0   # temperature for log-sum-exp

# Sampling & bootstrap
MAX_SEQ_LEN     = 1022          # amino acids
MAX_POS_SAMPLES = None          # cap positives; None => use all positives available
NEG_MATCH_STRAT = "length"      # "length" or "random"
N_LENGTH_BINS   = 12            # bins for length-matched sampling
SEED            = 42
BOOT_N          = 3000          # bootstrap iterations for CI
ALTERNATIVE     = "greater"     # MWU one-sided; higher score => positive

# Multiple testing
MT_METHOD       = "bh"          # "bh" (FDR) or "bonferroni"
ALPHA           = 0.05

SHOW_BOOT_PROGRESS = True  # set False if too chatty

def compute_token_activations(seq: str, layer_idx: int, unit_idx: int) -> np.ndarray:
    """
    Return a 1D numpy array of per-token activations for 'unit' at 'layer' on 'seq'.
    You should plug in your ESM/SAE hook here and return activations on non-padding tokens.

    Example signature in your codebase:
        acts = get_sae_token_acts(seq, layer, unit)  # shape [L]
        return acts.astype(np.float32)

    DEMO_MODE returns synthetic motif-like activations.
    """
    """Return max activation of *one neuron* over the sequence."""
    sae_layer = saes[layer_2_saelayer[layer_idx]]
    _, _, toks = batch_converter([(1, seq)])
    toks, mask = toks.to(device), (toks != esm2_alphabet.padding_idx).to(device)

    hook = SAEHookProt(
        sae          = sae_layer,
        mask_BL         = mask,
        cache_latents=True,
        layer_is_lm     = False,
        calc_error   = True,
        use_error    = True,
    )
    h = esm_transformer.esm.encoder.layer[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        _ = esm_transformer.predict_contacts(toks, mask)[0]
    h.remove(); torch.cuda.empty_cache()

    result = sae_layer.feature_acts[:, unit_idx].detach().cpu().numpy().astype(np.float32)
    # Clear SAE cache to prevent memory buildup
    if hasattr(sae_layer, 'feature_acts'):
        del sae_layer.feature_acts
        torch.cuda.empty_cache()
    return result

# ------------------------------ FASTA IO ---------------------------------

def parse_fasta(path: str, max_len: int = MAX_SEQ_LEN) -> List[Tuple[str, str]]:
    """
    Returns list of (id, seq) with len(seq) <= max_len.
    Requires Biopython if available; else, uses a simple fallback parser.
    """
    entries = []
    try:
        from Bio import SeqIO  # type: ignore
        for rec in SeqIO.parse(path, "fasta"):
            s = str(rec.seq)
            if len(s) <= max_len:
                entries.append((rec.id, s))
    except Exception:
        # Minimal FASTA reader
        with open(path, "r", encoding="utf-8") as f:
            ident, buf = None, []
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if ident is not None:
                        seq = "".join(buf)
                        if len(seq) <= max_len:
                            entries.append((ident, seq))
                    ident = line[1:].split()[0]
                    buf = []
                else:
                    buf.append(line)
            if ident is not None:
                seq = "".join(buf)
                if len(seq) <= max_len:
                    entries.append((ident, seq))
    return entries

# --------------------------- Aggregators ---------------------------------

def agg_max(tok_acts: np.ndarray) -> float:
    return float(np.max(tok_acts)) if tok_acts.size else float("nan")

def agg_topq(tok_acts: np.ndarray, q: float = TOPQ) -> float:
    if tok_acts.size == 0: return float("nan")
    k = max(1, int(math.ceil(q * tok_acts.size)))
    # top-k mean via partial sort
    part = np.partition(tok_acts, -k)[-k:]
    return float(np.mean(part))

def agg_lse(tok_acts: np.ndarray, tau: float = LSE_TAU) -> float:
    if tok_acts.size == 0: return float("nan")
    a = tok_acts / max(1e-8, tau)
    m = float(np.max(a))
    return float(m + np.log(np.exp(a - m).sum()))

def agg_mean(tok_acts: np.ndarray) -> float:
    return float(np.mean(tok_acts)) if tok_acts.size else float("nan")

def agg_lme(tok_acts, tau=2.0):  # log-mean-exp (length neutral)
    if tok_acts.size == 0: return float("nan")
    a = tok_acts / max(1e-8, tau)
    m = float(np.max(a))
    return float(m + np.log(np.exp(a - m).sum()) - np.log(tok_acts.size))

def agg_topk(tok_acts, K=64):     # fixed top-K mean (length neutral)
    if tok_acts.size == 0: return float("nan")
    k = min(K, tok_acts.size)
    return float(np.partition(tok_acts, -k)[-k:].mean())

AGG_FUNCS = {
    "max": agg_max,
    "topq": lambda x: agg_topq(x, TOPQ),
    "lse": lambda x: agg_lse(x, LSE_TAU),
    "mean": agg_mean,
    "lme": lambda x: agg_lme(x),
    "topk": lambda x: agg_topk(x),
}

# ------------------------- Metrics & bootstrap ---------------------------

def stratified_boot_auc(y: np.ndarray, s: np.ndarray, B: int = BOOT_N, seed: int = SEED,
                        show_progress: bool = False, desc: str = "bootstrap") -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s)
    pos = s[y==1]; neg = s[y==0]
    n1, n0 = pos.size, neg.size
    aucs = np.empty(B, dtype=np.float64)
    iterator = range(B)
    if show_progress:
        iterator = tqdm(iterator, leave=False, dynamic_ncols=True, desc=desc, unit="boot")
    for b in iterator:
        pos_b = pos[rng.integers(0, n1, n1)]
        neg_b = neg[rng.integers(0, n0, n0)]
        y_b = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        s_b = np.r_[pos_b, neg_b]
        aucs[b] = roc_auc_score(y_b, s_b)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    auc_hat = roc_auc_score(y, s)
    return auc_hat, (float(lo), float(hi))

def mwu_test(pos_scores: np.ndarray, neg_scores: np.ndarray, alternative: str = "greater"):
    res = mannwhitneyu(pos_scores, neg_scores, alternative=alternative, method="auto")
    U = float(res.statistic)
    p = float(res.pvalue)
    auc_from_U = U / (pos_scores.size * neg_scores.size)
    return U, p, auc_from_U

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)                 # ascending p
    q = np.empty_like(p)
    cummin = 1.0
    # Traverse from largest p to smallest (step-down)
    for i in range(m-1, -1, -1):
        idx = order[i]
        rank = i + 1
        val = p[idx] * m / rank
        cummin = min(cummin, val)
        q[idx] = cummin
    return np.clip(q, 0.0, 1.0)

# ------------------------- Length matching utils -------------------------

def quantile_bins(lengths: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins+1))
    # ensure strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-6
    return edges

def sample_length_matched_negatives(pos_lengths: np.ndarray, neg_pool_lengths: np.ndarray, rng, n_bins: int = 12) -> np.ndarray:
    edges = quantile_bins(np.r_[pos_lengths, neg_pool_lengths], n_bins)
    idx_pool = np.arange(neg_pool_lengths.size)
    chosen = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        pos_in_bin = ((pos_lengths >= lo) & (pos_lengths <= hi)).sum()
        if pos_in_bin == 0: 
            continue
        cand = idx_pool[(neg_pool_lengths >= lo) & (neg_pool_lengths <= hi)]
        if cand.size == 0:
            continue
        take = min(pos_in_bin, cand.size)
        chosen.append(rng.choice(cand, size=take, replace=False))
    if not chosen:
        return np.array([], dtype=int)
    return np.concatenate(chosen)

def run_experiment(
    targets: List[Tuple[int,int,str,str]],
    fasta_sprot: str,
    aggregators: List[str] = AGGREGATORS,
    neg_match: str = NEG_MATCH_STRAT,
    n_bins: int = N_LENGTH_BINS,
    max_seq_len: int = MAX_SEQ_LEN,
    max_pos_samples: int = MAX_POS_SAMPLES,
    seed: int = SEED,
    boot_n: int = BOOT_N,
    mt_method: str = MT_METHOD,
    alpha: float = ALPHA,
    alternative: str = ALTERNATIVE,
    outdir: str = "./reconcile_out"
):
    out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Load negative pool (Swiss-Prot / reviewed)
    neg_pool = parse_fasta(fasta_sprot, max_len=max_seq_len)
    if len(neg_pool) == 0:
        raise RuntimeError(f"No sequences loaded from {fasta_sprot}")
    neg_pool_ids = np.array([pid for pid,_ in neg_pool], dtype=object)
    neg_pool_seqs = np.array([seq for _,seq in neg_pool], dtype=object)
    neg_pool_lens = np.array([len(s) for s in neg_pool_seqs], dtype=int)

    rows = []
    target_bar = tqdm(targets, desc="targets", unit="latent", dynamic_ncols=True)

    for (layer, unit, domain, pos_fasta) in target_bar:
        target_bar.set_postfix_str(f"L{layer}.{unit} {domain}")

        # Positives
        pos_entries = parse_fasta(pos_fasta, max_len=max_seq_len)
        if len(pos_entries) == 0:
            print(f"[WARN] No positives loaded from {pos_fasta}; skipping {domain}.")
            continue
        if max_pos_samples is not None and len(pos_entries) > max_pos_samples:
            pos_entries = list(rng.choice(pos_entries, size=max_pos_samples, replace=False))
        pos_ids  = np.array([pid for pid,_ in pos_entries], dtype=object)
        pos_seqs = np.array([seq for _,seq in pos_entries], dtype=object)
        pos_lens = np.array([len(s) for s in pos_seqs], dtype=int)

        # Exclude positives from negative pool
        mask_not_pos = ~np.isin(neg_pool_ids, pos_ids)
        neg_ids_pool  = neg_pool_ids[mask_not_pos]
        neg_seqs_pool = neg_pool_seqs[mask_not_pos]
        neg_lens_pool = neg_pool_lens[mask_not_pos]

        # Length-matched negatives
        if neg_match == "length":
            idx = sample_length_matched_negatives(pos_lens, neg_lens_pool, rng, n_bins=n_bins)
            if idx.size < len(pos_seqs):
                # if under-filled, top up by random (without overlap)
                extra = rng.choice(np.setdiff1d(np.arange(len(neg_seqs_pool)), idx), 
                                   size=len(pos_seqs)-idx.size, replace=False)
                idx = np.concatenate([idx, extra])
        else:
            idx = rng.choice(np.arange(len(neg_seqs_pool)), size=len(pos_seqs), replace=False)

        neg_ids  = neg_ids_pool[idx]
        neg_seqs = neg_seqs_pool[idx]
        neg_lens = neg_lens_pool[idx]

        # -------- compute token activations ONCE per sequence (with progress) --------
        pos_tok_acts = []
        for s in tqdm(pos_seqs, desc=f"L{layer}.{unit} {domain}  pos seqs", unit="seq", leave=False, dynamic_ncols=True):
            pos_tok_acts.append(compute_token_activations(s, layer, unit))
        neg_tok_acts = []
        for s in tqdm(neg_seqs, desc=f"L{layer}.{unit} {domain}  neg seqs", unit="seq", leave=False, dynamic_ncols=True):
            neg_tok_acts.append(compute_token_activations(s, layer, unit))

        # Precompute aggregations for each aggregator from the cached token acts
        for agg in aggregators:
            func = AGG_FUNCS[agg]
            pos_scores = np.array([func(x) for x in pos_tok_acts], dtype=np.float64)
            neg_scores = np.array([func(x) for x in neg_tok_acts], dtype=np.float64)

            y = np.r_[np.ones(pos_scores.size, dtype=int), np.zeros(neg_scores.size, dtype=int)]
            s = np.r_[pos_scores, neg_scores]

            # Metrics (with optional bootstrap bar)
            auc, (lo, hi) = stratified_boot_auc(y, s, B=boot_n, seed=seed,
                                                show_progress=SHOW_BOOT_PROGRESS,
                                                desc=f"boot L{layer}.{unit} {domain} [{agg}]")
            U, p, aucU = mwu_test(pos_scores, neg_scores, alternative=alternative)
            agree = abs(auc - aucU) < 1e-6

            rho_pos, _ = spearmanr(pos_scores, pos_lens)
            rho_neg, _ = spearmanr(neg_scores, neg_lens)

            rows.append({
                "layer": layer, "unit": unit, "domain": domain, "aggregator": agg,
                "n_pos": int(pos_scores.size), "n_neg": int(neg_scores.size),
                "auc": float(auc), "ci_lo": float(lo), "ci_hi": float(hi),
                "U": float(U), "p_mwu": float(p), "auc_from_U": float(aucU), "auc_matches_U": bool(agree),
                "rho_len_pos": float(rho_pos), "rho_len_neg": float(rho_neg),
            })

    # ---- assemble & save (unchanged) ----
    df = pd.DataFrame(rows)
    if df.empty:
        print("No results produced; check your inputs.")
        return

    if MT_METHOD.lower() == "bh":
        df["qval"] = benjamini_hochberg(df["p_mwu"].values)
        mt_label = "BH-FDR qval"
    elif MT_METHOD.lower() == "bonferroni":
        m = df.shape[0]
        df["qval"] = np.minimum(1.0, df["p_mwu"].values * m)
        mt_label = "Bonferroni-adjusted p"
    else:
        df["qval"] = df["p_mwu"]; mt_label = "p (uncorrected)"

    out_csv = pathlib.Path(outdir) / "results_reconcile.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Summary (sorted by AUC) ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.sort_values(["aggregator", "auc"], ascending=[True, False]).to_string(index=False,
              formatters={"auc":"{:.3f}".format, "ci_lo":"{:.3f}".format, "ci_hi":"{:.3f}".format,
                          "p_mwu":"{:.2e}".format, "qval":"{:.2e}".format}))

    # plots (unchanged)
    try:
        import matplotlib.pyplot as plt
        for agg in sorted(df["aggregator"].unique()):
            sub = df[df["aggregator"] == agg].sort_values("auc", ascending=True)
            yv = np.arange(sub.shape[0])
            plt.figure(figsize=(6.0, max(2.5, 0.35*len(yv))))
            plt.barh(yv, sub["auc"].values, color="tab:blue", edgecolor="black", height=0.6)
            xerr = np.vstack([sub["auc"].values - sub["ci_lo"].values,
                              sub["ci_hi"].values - sub["auc"].values])
            plt.errorbar(sub["auc"].values, yv, xerr=xerr, fmt="none", ecolor="black", capsize=3, lw=1)
            plt.axvline(0.5, color="grey", ls="--", lw=1)
            labels = [f"L{L}.{U}\n{D}" for L,U,D in zip(sub["layer"], sub["unit"], sub["domain"])]
            plt.yticks(yv, labels, fontsize=9)
            plt.gca().invert_yaxis()
            plt.xlabel("AUROC (higher = more selective)")
            plt.title(f"Selectivity by {agg} (CI = 95% stratified bootstrap)\n{mt_label}: see CSV", fontsize=11)
            plt.tight_layout()
            out_png = pathlib.Path(outdir) / f"plot_{agg}.png"
            plt.savefig(out_png, dpi=200)
            plt.close()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    meta = {
        "fasta_sprot": fasta_sprot,
        "targets": TARGETS,
        "aggregators": AGGREGATORS,
        "params": {
            "TOPQ": TOPQ, "LSE_TAU": LSE_TAU, "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "NEG_MATCH_STRAT": NEG_MATCH_STRAT, "N_LENGTH_BINS": N_LENGTH_BINS,
            "SEED": SEED, "BOOT_N": BOOT_N, "ALTERNATIVE": ALTERNATIVE,
            "MT_METHOD": MT_METHOD, "ALPHA": ALPHA,
        },
    }
    with open(pathlib.Path(outdir) / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: {out_csv}")
    print(f"Plots saved under: {outdir}")

def main():
    """Main function to run the domain correlation experiment."""
    # Check that all FASTA files exist
    if not os.path.exists(FASTA_SPROT):
        raise FileNotFoundError(f"Swiss-Prot FASTA file not found at: {FASTA_SPROT}")

    # Check each target's FASTA file exists and filter out missing ones
    valid_targets = []
    missing_files = []
    for layer, unit, domain_id, fasta_path in TARGETS:
        if os.path.exists(fasta_path):
            valid_targets.append((layer, unit, domain_id, fasta_path))
        else:
            missing_files.append((domain_id, fasta_path))
    
    if missing_files:
        print(f"Warning: {len(missing_files)} FASTA files not found, will be skipped:")
        for domain_id, fasta_path in missing_files:
            print(f"  - {domain_id}: {fasta_path}")
    
    print(f"Found {len(valid_targets)} valid targets out of {len(TARGETS)} total.")
    
    if not valid_targets:
        raise RuntimeError("No valid targets found - all FASTA files are missing!")
        
    # Use only valid targets
    targets_to_use = valid_targets
    
    run_experiment(
        targets=targets_to_use,
        fasta_sprot=FASTA_SPROT,
        aggregators=AGGREGATORS,
        neg_match=NEG_MATCH_STRAT,
        n_bins=N_LENGTH_BINS,
        max_seq_len=MAX_SEQ_LEN,
        max_pos_samples=MAX_POS_SAMPLES,
        seed=SEED,
        boot_n=BOOT_N,
        mt_method=MT_METHOD,
        alpha=ALPHA,
        alternative=ALTERNATIVE,
        outdir="./reconcile_out_main_gpu"
    )

if __name__ == "__main__":
    main()
