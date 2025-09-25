# %%

import argparse
import torch
import json
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from typing import Sequence, Tuple, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
import ruptures as rpt
from sklearn.mixture import GaussianMixture
import time
import os
from scipy.stats import rankdata, mannwhitneyu, fisher_exact
import sys
sys.path.append('../../')
sys.path.append('../../plm_circuits')

# Import utility functions
from helpers.utils import (
    clear_memory,
    load_esm,
    load_sae_prot,
    mask_flanks_segment,
    patching_metric,
    cleanup_cuda,
    set_seed
)

# Import attribution functions
from attribution import (
    integrated_gradients_sae,
    topk_sae_err_pt
)

# Import hook classes
from hook_manager import SAEHookProt

from data.protein_params import sse_dict, fl_dict, protein_name, protein2pdb
from data.feature_clusters_hypotheses import feature_clusters_MetXA, feature_clusters_Top2

# Additional imports
import json
from functools import partial
import os
import torch
import collections
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# Lazy caches to avoid heavy loads on import
_ESM_CACHE = None  # type: ignore[var-annotated]
_SAE_CACHE: Dict[int, object] = {}


# %%

all_acts_mean = torch.load("/project/pi_annagreen_umass_edu/bryn/plm_circuit_enrichment/data/summarized_acts_length_normalized.pt", weights_only=True) 
all_acts_max = torch.load("/project/pi_annagreen_umass_edu/bryn/plm_circuit_enrichment/data/summarized_acts_max.pt", weights_only=True)
all_acts_topq = torch.load("data/summarized_acts_top_q.pt", weights_only=True)

all_properties = torch.load("metadata/ptn_fam_tensor_nonzero.pt", weights_only=True)

with open("metadata/list_of_desired_latents.pkl", 'rb') as f:
    subset_list = list(set(pickle.load(f)))
    subset_list.sort()

interpro_annotations_nonzero = pd.read_csv("metadata/interpro_entry_list_mapping_nonzero.csv")

all_acts_mean = all_acts_mean[:, subset_list]
all_acts_max = all_acts_max[:, subset_list]
all_acts_topq = all_acts_topq[:, subset_list]

print(f"Number of latents: {len(subset_list)}")


with open("/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_circuits/results/layer_latent_dicts/layer_latent_dict_2PKEA_0.70.json", "r") as f:
    pkea_latents = json.load(f)

with open("/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_circuits/results/layer_latent_dicts/layer_latent_dict_MetXA_0.70.json", 'r') as file:
    metx_latents = json.load(file)

with open("/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_circuits/results/layer_latent_dicts/layer_latent_dict_Top2_0.70.json", 'r') as file:
    top2_latents = json.load(file)


# only working on metxa and top2 for now
offsets = {k: i*4096 for i,k in enumerate(metx_latents.keys())}
full_indices_metx = [j+offsets[layer_id] for layer_id in metx_latents.keys() for j in metx_latents[layer_id]]
full_indices_top2 = [j+offsets[layer_id] for layer_id in top2_latents.keys() for j in top2_latents[layer_id]]
# full_indices_pkea = [j+offsets[layer_id] for layer_id in pkea_latents.keys() for j in pkea_latents[layer_id]]
latent_ids_both_proteins = list(set(full_indices_top2+full_indices_metx))
# latent_ids_both_proteins = list(set(full_indices_pkea))
latent_ids_both_proteins.sort()
print(f"Number of latent ids both proteins: {len(latent_ids_both_proteins)}, equal latents? {subset_list == latent_ids_both_proteins}")

# %%

def to_flattened_id(layer_id: int, lat_id: int) -> int:
    if layer_id % 4 != 0:
        raise ValueError("layer_id must be a multiple of 4 (e.g., 4, 8, 12, ...)")
    if not (0 <= lat_id < 4096):
        raise ValueError("lat_id must be in [0, 4096)")
    return ((layer_id // 4) - 1) * 4096 + lat_id

def get_layer_and_latent(latent_i: int, flattened_ids: Sequence[int]) -> Tuple[int, int]:
    """
    Given a latent index and a list/array of flattened IDs,
    return (layer_id, latent_ind).

    Assumes:
    - flattened_id -> layer: ((flattened_id // 4096) + 1) * 4
    - flattened_id -> latent_ind: flattened_id % 4096
    """
    if latent_i < 0 or latent_i >= len(flattened_ids):
        raise IndexError("latent_i out of range for flattened_ids")
    cur_flattened = int(flattened_ids[latent_i])
    layer_id = ((cur_flattened // 4096) + 1) * 4
    latent_ind = cur_flattened % 4096
    return layer_id, latent_ind


def _kneedle_candidate_indices(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return candidate knee indices and scores using the Kneedle heuristic."""
    N = s.size
    if N < 3:
        return np.array([], dtype=int), np.array([], dtype=float)

    x = np.linspace(0.0, 1.0, N)
    y = s.astype(float)
    denom = float(np.max(y) - np.min(y))
    if denom <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    y_norm = (y - np.min(y)) / denom

    diff = y_norm + x - 1.0  # concave, decreasing curve distance to diagonal
    peaks, _ = find_peaks(diff)

    if peaks.size == 0:
        peak_idx = int(np.argmax(diff))
        if peak_idx <= 0 or peak_idx >= N - 1:
            return np.array([], dtype=int), np.array([], dtype=float)
        return np.array([peak_idx], dtype=int), np.array([float(diff[peak_idx])], dtype=float)

    scores = diff[peaks]
    order = np.argsort(scores)[::-1]
    peaks = peaks[order]
    scores = scores[order]

    valid = (peaks > 0) & (peaks < N - 1)
    return peaks[valid], scores[valid]


def _second_derivative_candidate_indices(s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return candidate knees using second-derivative magnitude of a smoothed curve."""
    N = s.size
    if N < 3:
        return np.array([], dtype=int), np.array([], dtype=float)

    window = min(51, N)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        smooth = s.astype(float)
    else:
        polyorder = 3 if window > 3 else 2
        smooth = savgol_filter(s.astype(float), window_length=window, polyorder=polyorder)

    d2 = np.diff(smooth, n=2)
    if d2.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    idxs = np.arange(1, N - 1)
    scores = np.abs(d2)
    order = np.argsort(scores)[::-1]
    idxs = idxs[order]
    scores = scores[order]

    valid = (idxs > 0) & (idxs < N - 1)
    return idxs[valid], scores[valid]


def find_activation_tail_regions(
    layer_i: int,
    latent_ind: int,
    *,
    all_acts=None,
    n_regions: int = 5,
    buffer: int = 50,
    min_drop: Optional[float] = None,
    highlight_span: Optional[int] = None,
    method: str = "drop",
) -> List[dict]:
    """Identify large activation drops ("tail" candidates) for a latent.

    Returns dictionaries describing the top `n_regions` non-overlapping drops,
    ordered by a ranking metric tied to the selected method (`drop`, `kneedle`,
    or `second_derivative`). Each dictionary includes the rank where the drop
    occurs, corresponding tau values, and the absolute / relative drop size to
    help with downstream filtering or plotting.
    """
    flattened_id = to_flattened_id(int(layer_i), int(latent_ind))
    try:
        latent_i = subset_list.index(flattened_id)
    except ValueError as exc:
        raise ValueError(
            f"Latent L{layer_i}-{latent_ind} (flat {flattened_id}) not found in subset_list."
        ) from exc

    acts_source = all_acts if all_acts is not None else globals().get("all_acts")
    if acts_source is None:
        raise ValueError("Activation table `all_acts` must be provided or exist globally.")

    v = to_numpy(acts_source)[:, latent_i].astype(float)
    order = np.argsort(v)[::-1]
    s = v[order]
    N = s.size
    if N < 2:
        return []

    drops = s[:-1] - s[1:]
    if drops.ndim != 1:
        drops = drops.reshape(-1)
    max_val = float(np.max(s)) if np.size(s) else 0.0

    method = method.lower()

    span = highlight_span if highlight_span is not None else max(int(buffer), 1)
    if span <= 0:
        span = 1

    rel_drops = drops / (max_val + 1e-12)

    candidate_scores: dict[int, float]

    if method == "drop":
        order_candidates = np.argsort(rel_drops)[::-1]
        candidate_scores = {int(idx): float(rel_drops[idx]) for idx in order_candidates}
    elif method == "kneedle":
        order_candidates, scores = _kneedle_candidate_indices(s)
        candidate_scores = {int(idx): float(score) for idx, score in zip(order_candidates, scores)}
    elif method in {"second_derivative", "second-derivative", "curvature"}:
        order_candidates, scores = _second_derivative_candidate_indices(s)
        candidate_scores = {int(idx): float(score) for idx, score in zip(order_candidates, scores)}
        method = "second_derivative"
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'drop', 'kneedle', or 'second_derivative'.")

    # ensure candidates correspond to valid drops (need idx + 1 < N)
    order_candidates = [int(idx) for idx in order_candidates if int(idx) < N - 1]

    chosen: List[dict] = []
    chosen_idxs: List[int] = []

    def rank_to_tau(rank: int) -> float:
        return float(max(0.0, 1.0 - rank / N))

    for idx in order_candidates:
        drop_value = float(drops[idx])
        if min_drop is not None and drop_value < float(min_drop):
            break
        if buffer:
            if any(abs(idx - prev_idx) <= buffer for prev_idx in chosen_idxs):
                continue
        rank = idx + 1  # 1-based rank of the last point before the drop
        tail_start_rank = min(N, idx + 2)
        tail_end_rank = min(N, tail_start_rank + span - 1)

        selected = {
            "drop_rank": int(rank),
            "drop_tau": rank_to_tau(rank),
            "tail_start_rank": int(tail_start_rank),
            "tail_start_tau": rank_to_tau(tail_start_rank),
            "tail_end_rank": int(tail_end_rank),
            "tail_end_tau": rank_to_tau(tail_end_rank),
            "drop": drop_value,
            "relative_drop": float(
                drop_value / (max_val + 1e-12)
            ),
            "value_before": float(s[idx]),
            "value_after": float(s[idx + 1]),
            "score": float(candidate_scores.get(idx, 0.0)),
            "method": method,
        }
        chosen.append(selected)
        chosen_idxs.append(int(idx))
        if len(chosen) >= int(n_regions):
            break

    return chosen

def save_activation_curve(
    layer_i,
    latent_ind,
    tau=0.99,
    symlog=False,
    highlight_domain_idx=None,
    highlight_domain_idxs=None,
    highlight_color="orange",
    highlight_alpha=0.8,
    highlight_marker_size=18,
    highlight_edgecolor="black",
    highlight_edgewidth=0.4,
    sample_control=False,
    control_domain_range=(0, 10096),
    control_min_proteins=21,
    control_max_proteins=None,
    control_color="slategray",
    control_alpha=0.75,
    control_marker_size=20,
    control_edgecolor="black",
    control_edgewidth=0.35,
    control_seed=None,
    rng=None,
    domain_jitter=0.02,
    out_dir=None,
    all_acts=None,
    tail_regions: Optional[Sequence[dict]] = None,
    tail_color="mediumpurple",
    tail_alpha=0.18,
    tail_linecolor="mediumpurple",
    tail_linewidth=1.2,
    tail_label="tail drop",
    tail_annotate: bool = True,
    tail_annotation_color: str = "black",
    tail_annotation_fontsize: float = 9.0,
    tail_annotation_offset: float = 0.02,
    show_plt=False,
):
    """Plot a sorted activation curve with optional domain overlays.

    highlight_domain_idx / highlight_domain_idxs can mark multiple columns;
    supply a list/tuple/ndarray for the latter. If highlight_color is a
    sequence, its entries are cycled across those domains.

    When sample_control=True, a control domain is drawn uniformly from
    control_domain_range (inclusive) that has ≥ control_min_proteins members
    and isn’t already highlighted. domain_jitter controls the vertical offset
    applied to each highlighted domain (fraction of the activation range).

    tail_regions can be the output of find_activation_tail_regions to shade
    prominent activation drops. When tail_annotate=True each region is labeled
    with its rank order (1, 2, 3, …) at the corresponding drop location.
    """
    flattened_id = to_flattened_id(int(layer_i), int(latent_ind))
    latent_i = subset_list.index(flattened_id)

    acts_source = all_acts if all_acts is not None else globals().get("all_acts")
    if acts_source is None:
        raise ValueError("Activation table `all_acts` must be provided or exist globally.")

    v = to_numpy(acts_source)[:, latent_i].astype(float)
    order = np.argsort(v)[::-1]
    s = v[order]
    N = s.size

    out_dir_use = globals().get("out_dir", Path("./")) if out_dir is None else Path(out_dir)
    out_dir_use.mkdir(parents=True, exist_ok=True)

    k_top = max(1, int(np.ceil((1 - float(tau)) * N)))

    if N > 1 and (s.max() - s.min()) > 0:
        x = np.linspace(0.0, 1.0, N)
        y = (s - s.min()) / (s.max() - s.min() + 1e-12)
        knee_idx = int(np.argmax(y - (1.0 - x)))
    else:
        knee_idx = 0

    highlight_ids = []
    if highlight_domain_idx is not None:
        highlight_ids.append(int(highlight_domain_idx))
    if highlight_domain_idxs is not None:
        if isinstance(highlight_domain_idxs, np.ndarray):
            highlight_ids.extend(int(i) for i in highlight_domain_idxs.flatten())
        elif isinstance(highlight_domain_idxs, (list, tuple, set)):
            highlight_ids.extend(int(i) for i in highlight_domain_idxs)
        else:
            highlight_ids.append(int(highlight_domain_idxs))
    seen = set()
    ordered_ids = []
    for idx in highlight_ids:
        if idx not in seen:
            ordered_ids.append(idx)
            seen.add(idx)

    domain_specs = [{"idx": idx, "source": "user"} for idx in ordered_ids]

    needs_properties = bool(domain_specs) or sample_control
    properties = None
    if needs_properties:
        properties = globals().get("all_properties")
        if properties is None:
            raise ValueError("Domain highlighting/control requires `all_properties`.")
        properties = to_numpy(properties)
        if properties.ndim != 2:
            raise ValueError("`all_properties` must be a 2-D array.")

    if sample_control and properties is not None:
        rng_use = rng if rng is not None else np.random.default_rng(control_seed)
        lower, upper = control_domain_range
        lower = max(int(lower), 0)
        upper = int(upper)
        if properties.shape[1] == 0:
            raise ValueError("`all_properties` has zero columns.")
        upper = min(upper, properties.shape[1] - 1)
        if lower > upper:
            print(f"No control sampled: range {control_domain_range} outside available columns.")
        else:
            candidates = []
            for col in range(lower, upper + 1):
                if col in seen:
                    continue
                n_present = int((properties[:, col] != 0).sum())
                if n_present < control_min_proteins:
                    continue
                if control_max_proteins is not None and n_present > control_max_proteins:
                    continue
                candidates.append((col, n_present))
            if not candidates:
                print(f"No eligible control domain in [{lower}, {upper}] with ≥{control_min_proteins} proteins.")
            else:
                pick = int((rng_use or np.random.default_rng()).integers(len(candidates)))
                control_idx, control_count = candidates[pick]
                domain_specs.append({"idx": control_idx, "source": "control", "count": control_count})
                seen.add(control_idx)
                print(f"Sampled control domain {control_idx} (n={control_count}).")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, N + 1), s, lw=1.2, color="#1f77b4")
    ax.axvline(k_top, color="crimson", ls="--", alpha=0.7, label=f"top {100*(1-float(tau)):.1f}%")
    ax.axvline(knee_idx + 1, color="teal", ls=":", alpha=0.7, label="knee")

    if tail_regions:
        seen_tail_label = False
        y_span = float(s.max() - s.min()) if np.isfinite(s.max() - s.min()) else 0.0
        if y_span == 0:
            y_span = max(float(np.abs(s).max()), 1.0)
        vertical_offset = float(tail_annotation_offset) * y_span if tail_annotation_offset else 0.0

        for idx_region, region in enumerate(tail_regions, start=1):
            try:
                drop_rank = int(region.get("drop_rank"))
            except (TypeError, AttributeError):
                continue
            if drop_rank < 1 or drop_rank > N:
                continue

            tail_start = int(region.get("tail_start_rank", drop_rank + 1))
            tail_end = int(region.get("tail_end_rank", tail_start))
            if tail_start < 1:
                tail_start = 1
            if tail_end < tail_start:
                tail_end = tail_start
            if tail_end > N:
                tail_end = N

            label = tail_label if not seen_tail_label else None
            if tail_alpha and tail_color and tail_start <= tail_end:
                ax.axvspan(
                    tail_start,
                    tail_end,
                    color=tail_color,
                    alpha=tail_alpha,
                    label=label,
                    zorder=1.8,
                )
                if label is not None:
                    seen_tail_label = True
                label = None  # avoid double legend entries

            if tail_linecolor:
                line_label = tail_label if not seen_tail_label else None
                ax.axvline(
                    drop_rank,
                    color=tail_linecolor,
                    ls="--",
                    lw=tail_linewidth,
                    alpha=0.9,
                    label=line_label,
                    zorder=2.6,
                )
                if line_label is not None:
                    seen_tail_label = True

            if tail_annotate and drop_rank <= N:
                drop_idx = max(0, min(N - 1, drop_rank - 1))
                y_val = float(s[drop_idx]) if np.isfinite(s[drop_idx]) else 0.0
                text_y = y_val + vertical_offset
                ax.text(
                    drop_rank,
                    text_y,
                    str(idx_region),
                    color=tail_annotation_color,
                    fontsize=tail_annotation_fontsize,
                    ha="center",
                    va="bottom",
                    zorder=3.2,
                )

    if domain_specs and properties is not None:
        color_cycle = list(highlight_color) if isinstance(highlight_color, (list, tuple, np.ndarray)) else [highlight_color]
        if not color_cycle:
            color_cycle = ["orange"]

        y_span = float(s.max() - s.min()) if np.isfinite(s.max() - s.min()) else 0.0
        if y_span == 0:
            y_span = max(float(np.abs(s).max()), 1.0)
        jitter_step = float(domain_jitter) * y_span if domain_jitter else 0.0

        for spec_idx, spec in enumerate(domain_specs):
            col = spec["idx"]
            source = spec.get("source", "user")

            if col < 0 or col >= properties.shape[1]:
                ax.scatter([], [], s=highlight_marker_size, color="gray", alpha=0.5, edgecolors="none",
                           label=f"domain {col} (out of range)")
                print(f"Domain index {col} out of range for all_properties (columns={properties.shape[1]}).")
                continue

            mask = properties[:, col] != 0
            mask_sorted = mask[order]
            idxs = np.flatnonzero(mask_sorted)

            n_total = int(mask.sum())
            n_top = int(mask_sorted[:k_top].sum())
            coverage = (n_top / n_total) if n_total else 0.0

            df_ann = globals().get("interpro_annotations_nonzero")
            label_name = None
            if df_ann is not None:
                try:
                    label_name = str(df_ann.iloc[col]["ENTRY_NAME"])
                except Exception:
                    pass

            if source == "control":
                point_color = control_color
                marker_size = control_marker_size
                alpha_val = control_alpha
                edge_color = control_edgecolor
                edge_width = control_edgewidth
                base = f"control {col}" if label_name is None else f"control {label_name} (idx {col})"
            else:
                point_color = color_cycle[spec_idx % len(color_cycle)]
                marker_size = highlight_marker_size
                alpha_val = highlight_alpha
                edge_color = highlight_edgecolor
                edge_width = highlight_edgewidth
                base = f"domain {col}" if label_name is None else f"{label_name} (idx {col})"

            label = f"{base}; n={n_total}; cov@tau={coverage:.0%}"
            offset = (spec_idx - (len(domain_specs) - 1) / 2.0) * jitter_step if jitter_step else 0.0
            scatter_args = dict(
                s=marker_size,
                color=point_color,
                alpha=alpha_val,
                edgecolors=edge_color,
                linewidths=edge_width,
                label=label,
                zorder=3,
            )

            if idxs.size:
                ax.scatter(idxs + 1, s[idxs] + offset, **scatter_args)
            else:
                ax.scatter([], [], **scatter_args)

    ax.set_xlabel("Proteins (sorted by latent activation, rank)")
    ax.set_ylabel("Activation")
    ax.set_xlim(1, N)
    if symlog:
        ax.set_yscale("symlog")
    ax.legend(frameon=False)
    ax.set_title(f"L{layer_i}-{latent_ind} (flat {flattened_id})")
    plt.tight_layout()

    out_path = out_dir_use / f"L{layer_i}_{latent_ind}_sorted_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show_plt:
        plt.show()
    else:
        plt.close(fig)
        print(f"Saved: {out_path.resolve()}")

def to_numpy(a):
    if torch is not None and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def to_numpy(a):
    if torch is not None and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def percentile_ranks(x: np.ndarray) -> np.ndarray:
    """Return percentile ranks in [0,1] with average ranks on ties."""
    # rankdata gives 1..N; convert to 0..1
    r = rankdata(x, method="average")
    return (r - 1) / (x.size - 1) if x.size > 1 else np.zeros_like(r, dtype=float)

def bonferroni(pvals: np.ndarray) -> np.ndarray:
    m = pvals.size
    out = pvals * m
    out[out > 1.0] = 1.0
    return out

def tail_enrichment_table(R: np.ndarray, in_mask: np.ndarray, tau: float = 0.99):
    """
    Build 2x2 counts for Fisher at top tail threshold tau on percentile R.
      a: in-domain & top-tail
      b: out-of-domain & top-tail
      c: in-domain & not top-tail
      d: out-of-domain & not top-tail
    """
    top_mask = R >= np.quantile(R, tau)  # top (1 - tau) fraction; robust to ties
    a = np.sum(in_mask & top_mask)
    b = np.sum(~in_mask & top_mask)
    c = np.sum(in_mask & ~top_mask)
    d = np.sum(~in_mask & ~top_mask)
    return a, b, c, d

def screen_domains_tail_enrichment(
    all_acts,
    all_properties,
    latent_i: int,
    domain_indices: np.ndarray | list | None = None,
    tau: float = 0.99,
    n_in_min: int = 20,
    n_out_min: int = 200,
    use_bonferroni: bool = True,
    return_tau_all_in: bool = True,  # NEW
    coverage_fracs: tuple[float, ...] | None = (0.95, 0.90),  # NEW
):
    X = to_numpy(all_acts)
    D = to_numpy(all_properties)

    assert X.ndim == 2, "all_acts must be 2D [N, L]"
    assert D.ndim == 2, "all_properties must be 2D [N, M]"
    N = X.shape[0]

    # 1) Percentile ranks (scale-agnostic) for the chosen latent
    R = percentile_ranks(X[:, latent_i])

    # Domain set to consider
    if domain_indices is None:
        domain_indices = np.arange(D.shape[1])
    else:
        domain_indices = np.asarray(domain_indices, dtype=int)

    # Binarize domain membership (nonzero -> True)
    D_bin = D[:, domain_indices] != 0

    # Precompute the top-mask once (so Fisher thresholds are consistent across domains)
    # This uses same tau for every domain.
    thr = np.quantile(R, tau)
    top_mask = R >= thr
    not_top_mask = ~top_mask

    # Precompute totals
    top_count = int(top_mask.sum())
    not_top_count = N - top_count

    # Counts per domain (vectorized where possible)
    in_counts = D_bin.sum(axis=0)                    # n_in per domain
    out_counts = N - in_counts                       # n_out per domain

    # Filter domains by minimum sizes
    ok = (in_counts >= n_in_min) & (out_counts >= n_out_min)
    idxs = domain_indices[ok]
    if idxs.size == 0:
        cols = [
            "domain_idx", "n_in", "n_out",
            "a_top_in", "b_top_out", "c_not_top_in", "d_not_top_out",
            "odds_ratio", "fisher_p", "p_adj",
            "median_pos_percentile"
        ]
        if return_tau_all_in:
            cols.append("tau_all_in")
        return pd.DataFrame(columns=cols)

    D_ok = D_bin[:, ok]  # [N, K]
    n_in = in_counts[ok]
    n_out = out_counts[ok]

    # Vectorized a, c via matrix multiplications / sums
    a_vec = (top_mask[:, None] & D_ok).sum(axis=0)        # top & in
    c_vec = (not_top_mask[:, None] & D_ok).sum(axis=0)    # not top & in
    b_vec = top_count - a_vec                              # top & out
    d_vec = not_top_count - c_vec                          # not top & out

    # Fisher p-values and OR (loop per domain; fast enough for ~10k)
    pvals = np.empty(a_vec.size, dtype=float)
    ORs = np.empty(a_vec.size, dtype=float)
    for k in range(a_vec.size):
        a, b, c, d = int(a_vec[k]), int(b_vec[k]), int(c_vec[k]), int(d_vec[k])
        # Add a tiny continuity guard if any cell is zero when computing OR
        if a == 0 or b == 0 or c == 0 or d == 0:
            OR = ((a + 0.5) / (c + 0.5)) / ((b + 0.5) / (d + 0.5))
        else:
            OR = (a / c) / (b / d)
        ORs[k] = OR
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        pvals[k] = p

    p_adj = bonferroni(pvals) if use_bonferroni else pvals  # plug-in BH here if you later switch

    # Median positive percentile per domain
    med_pos = np.empty(a_vec.size, dtype=float)
    for k in range(a_vec.size):
        mask_k = D_ok[:, k]
        r_in = R[mask_k]
        med_pos[k] = float(np.median(r_in)) if r_in.size > 0 else np.nan

    data = {
        "domain_idx": idxs,
        "domain_name": [interpro_annotations_nonzero.iloc[idx]["ENTRY_NAME"] for idx in idxs],
        "n_in": n_in.astype(int),
        "n_out": n_out.astype(int),
        "a_top_in": a_vec.astype(int),
        "b_top_out": b_vec.astype(int),
        "c_not_top_in": c_vec.astype(int),
        "d_not_top_out": d_vec.astype(int),
        "odds_ratio": ORs,
        "fisher_p": pvals,
        "p_adj": p_adj,
        "median_pos_percentile": med_pos,
    }
        # NEW: tau at which ≥p of in-domain are in the top tail (p in coverage_fracs)
    if coverage_fracs:
        coverage_fracs = tuple(sorted(coverage_fracs, reverse=True))  # e.g., (0.95, 0.90)
        sorted_R = np.sort(R)
        denom = max(N - 1, 1)

        # Collect per-domain lower-tail quantiles r_q for q = 1 - p
        r_q = {p: np.empty(a_vec.size, dtype=float) for p in coverage_fracs}

        for k in range(a_vec.size):
            mask_k = D_ok[:, k]
            r_in = R[mask_k]
            if r_in.size == 0:
                for p in coverage_fracs:
                    r_q[p][k] = np.nan
            else:
                for p in coverage_fracs:
                    q = 1.0 - p  # need thr <= q-quantile to cover ≥p of in-domain
                    r_q[p][k] = float(np.quantile(r_in, q))

        for p in coverage_fracs:
            idx = np.searchsorted(sorted_R, r_q[p], side="right") - 1
            tau_cover = np.clip(idx / denom, 0.0, 1.0)
            data[f"tau_cover_{int(p*100)}"] = tau_cover

    # NEW: per-domain largest tau such that c == 0 (all in-domain are in top tail)
    if return_tau_all_in:
        sorted_R = np.sort(R)  # ascending
        # per-domain minimum in-domain percentile
        r_min_in = np.where(D_ok, R[:, None], np.inf).min(axis=0)
        # last index in sorted_R where value <= r_min_in
        last_le_idx = np.searchsorted(sorted_R, r_min_in, side="right") - 1
        # convert to tau using numpy's default quantile definition (linear over indices 0..N-1)
        denom = max(N - 1, 1)
        tau_all_in = np.clip(last_le_idx / denom, 0.0, 1.0)
        data["tau_all_in"] = tau_all_in

    df = pd.DataFrame(data)

    # Suggested default ranking: strongest practical signals first
    df = df.sort_values(
        by=["p_adj", "odds_ratio", "median_pos_percentile"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return df

# ---------- small utilities ----------
def to_numpy(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)

def percentile_ranks(x: np.ndarray) -> np.ndarray:
    r = rankdata(x, method="average")
    return (r - 1) / (x.size - 1) if x.size > 1 else np.zeros_like(r, dtype=float)

def bonferroni_local(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    out = pvals * m
    out[out > 1.0] = 1.0
    return out

# ---------- top-k metrics (precision, lift, Fisher/OR) ----------
def topk_metrics(scores: np.ndarray, in_mask: np.ndarray, k: int):
    """Return precision@k, lift@k, (a,b,c,d), OR (Haldane), Fisher p (one-sided)."""
    scores = to_numpy(scores).astype(float).ravel()
    in_mask = to_numpy(in_mask).astype(bool).ravel()
    N = scores.size
    assert in_mask.size == N and 1 <= k <= N

    order = np.argsort(scores)[::-1]
    top = order[:k]
    rest = order[k:]

    TP = int(in_mask[top].sum())
    FP = k - TP
    FN = int(in_mask[rest].sum())
    TN = N - k - FN

    p = TP / k if k > 0 else np.nan
    prev = in_mask.mean() if N > 0 else np.nan
    lift = (p / prev) if prev > 0 else np.inf

    # Odds ratio with Haldane–Anscombe correction if any zero
    a,b,c,d = TP, FP, FN, TN
    if 0 in (a,b,c,d):
        OR = ((a + 0.5) / (c + 0.5)) / ((b + 0.5) / (d + 0.5))
    else:
        OR = (a / c) / (b / d)

    _, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return {
        "k": k, "precision_k": p, "lift_k": lift,
        "a_top_in": a, "b_top_out": b, "c_not_top_in": c, "d_not_top_out": d,
        "or_topk": OR, "fisher_p_topk": fisher_p
    }

# ---------- MWU (plus optional AUC for interpretability) ----------
def mwu_test(scores_in: np.ndarray, scores_out: np.ndarray, alternative="greater"):
    if scores_in.size == 0 or scores_out.size == 0:
        return np.nan, np.nan
    U, p = mannwhitneyu(scores_in, scores_out, alternative=alternative)
    auc = U / (scores_in.size * scores_out.size)
    return p, auc

# ---------- consolidate per-latent analysis (P1 + P2) ----------
def analyze_latent_per_domain(
    all_acts, all_properties, latent_i: int,
    entry_names=None,
    k_multipliers=(1, 2),       # use k = mult * n_in
    use_bonferroni=True,
    min_in=20, min_out=200,
    summarized="topq"           # just for bookkeeping in the output
):
    """
    Returns a single DataFrame for this latent with:
    - n_in/n_out
    - MWU p and Bonf (P1)
    - precision/lift at k=n_in and 2*n_in (P2)
    - Fisher p + OR on the same 2x2 at each k (P2)
    """
    X = to_numpy(all_acts)
    D = (to_numpy(all_properties) != 0)
    N, M = D.shape

    scores = X[:, latent_i].astype(float)
    prevs = D.sum(axis=0) / N
    n_in = D.sum(axis=0)
    n_out = N - n_in
    ok = (n_in >= min_in) & (n_out >= min_out)

    idxs = np.where(ok)[0]
    if idxs.size == 0:
        cols = ["domain_idx","entry_name","n_in","n_out","mwu_p","mwu_p_adj","mwu_auc"]
        for mult in k_multipliers:
            cols += [f"k_{mult}x","precision","lift","fisher_p","fisher_p_adj","or_topk"]
        return pd.DataFrame(columns=cols)

    # --- P1: MWU per domain (correct within this latent's tested domains) ---
    mwu_p = np.empty(idxs.size, dtype=float)
    mwu_auc = np.empty(idxs.size, dtype=float)
    for j, dom in enumerate(idxs):
        mask = D[:, dom]
        p, auc = mwu_test(scores[mask], scores[~mask], alternative="greater")
        mwu_p[j] = p
        mwu_auc[j] = auc

    mwu_p_adj = bonferroni_local(mwu_p) if use_bonferroni else mwu_p

    # --- P2: top-k metrics at k = n_in and 2*n_in (no knee) ---
    # We also Bonferroni-correct Fisher p's within this same set.
    results = []
    fisher_cols = []
    for j, dom in enumerate(idxs):
        row = {
            "domain_idx": int(dom),
            "entry_name": (entry_names[dom] if entry_names is not None else None),
            "n_in": int(n_in[dom]),
            "n_out": int(n_out[dom]),
            "mwu_p": mwu_p[j],
            "mwu_p_adj": mwu_p_adj[j],
            "mwu_auc": mwu_auc[j],
            "prevalence": float(prevs[dom]),
            "summary_kind": summarized,
        }
        # compute once and store; we'll collect fisher p's to adjust jointly
        fisher_ps_this = []
        for mult in k_multipliers:
            k = int(max(1, min(scores.size, mult * n_in[dom])))
            tm = topk_metrics(scores, D[:, dom], k)
            row.update({
                f"k_{mult}x": k,
                f"precision_{mult}x": tm["precision_k"],
                f"lift_{mult}x": tm["lift_k"],
                f"or_topk_{mult}x": tm["or_topk"],
                f"fisher_p_{mult}x": tm["fisher_p_topk"],
            })
            fisher_ps_this.append(tm["fisher_p_topk"])
            fisher_cols.append((len(results), mult))  # remember where to write adj p
        results.append(row)

    df = pd.DataFrame(results)

    if use_bonferroni:
        for mult in k_multipliers:
            pcol = f"fisher_p_{mult}x"
            df[f"fisher_p_adj_{mult}x"] = bonferroni_local(df[pcol].to_numpy())
    else:
        for mult in k_multipliers:
            pcol = f"fisher_p_{mult}x"
            df[f"fisher_p_adj_{mult}x"] = df[pcol].to_numpy()

    # convenience sort: by mwu_p_adj then precision at 1x
    if f"precision_{k_multipliers[0]}x" in df.columns:
        df = df.sort_values(
            by=["mwu_p_adj", f"precision_{k_multipliers[0]}x", f"lift_{k_multipliers[0]}x"],
            ascending=[True, False, False]
        ).reset_index(drop=True)

    return df

# %%

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

def domain_overlap_report(
    all_properties: np.ndarray,
    domain_indices: list[int],
    entry_names: list[str] | None = None,
    protein_ids: list[str] | None = None,   # optional: to list overlapping proteins by ID
):
    """
    Build overlap summaries for a given list of InterPro domain indices.
    Returns:
      summary_df    : per-domain counts & prevalence
      pairwise_df   : long-form pairwise overlaps (A,B)
      multiway_info : dict with union/intersection masks & sizes
    """
    D = (np.asarray(all_properties) != 0)
    N, M = D.shape

    idx = np.array(domain_indices, dtype=int)
    if np.any((idx < 0) | (idx >= M)):
        bad = idx[(idx < 0) | (idx >= M)]
        raise IndexError(f"Invalid domain indices: {bad.tolist()} (M={M})")

    # Per-domain masks
    masks = {i: D[:, i] for i in idx}
    names = {i: (entry_names[i] if entry_names is not None else str(i)) for i in idx}

    # Per-domain summary
    n_in = {i: int(masks[i].sum()) for i in idx}
    prev = {i: n_in[i] / N for i in idx}
    summary_df = pd.DataFrame({
        "domain_idx": idx,
        "entry_name": [names[i] for i in idx],
        "n_in": [n_in[i] for i in idx],
        "prevalence": [prev[i] for i in idx],
    }).sort_values("prevalence", ascending=False, kind="mergesort").reset_index(drop=True)

    # Pairwise overlaps
    rows = []
    for i_pos, i in enumerate(idx):
        Ai = masks[i]
        ni = n_in[i]
        for j in idx[i_pos+1:]:
            Aj = masks[j]
            nj = n_in[j]

            inter = int(np.logical_and(Ai, Aj).sum())
            union = int(np.logical_or(Ai, Aj).sum())
            only_i = ni - inter
            only_j = nj - inter
            neither = N - union

            jacc = inter / union if union > 0 else np.nan
            pA_given_B = inter / nj if nj > 0 else np.nan
            pB_given_A = inter / ni if ni > 0 else np.nan

            # Fisher exact on 2×2 table: [[Ai∩Aj, Ai∩¬Aj],[¬Ai∩Aj, ¬Ai∩¬Aj]]
            table = [[inter, only_i],[only_j, neither]]
            _, fisher_p = fisher_exact(table, alternative="greater")

            rows.append({
                "A_idx": i, "A_name": names[i], "A_n": ni,
                "B_idx": j, "B_name": names[j], "B_n": nj,
                "intersect": inter, "union": union,
                "only_A": only_i, "only_B": only_j, "neither": neither,
                "jaccard": jacc,
                "P(A|B)": pA_given_B, "P(B|A)": pB_given_A,
                "fisher_p": fisher_p,
            })
    pairwise_df = pd.DataFrame(rows).sort_values(
        ["jaccard", "intersect"], ascending=[False, False], kind="mergesort"
    ).reset_index(drop=True)

    # Multiway intersection/union across ALL provided domains
    if len(idx) > 0:
        union_mask = np.logical_or.reduce([masks[i] for i in idx])
        inter_mask = np.logical_and.reduce([masks[i] for i in idx])
    else:
        union_mask = np.zeros(N, dtype=bool)
        inter_mask = np.zeros(N, dtype=bool)

    multiway_info = {
        "N": int(N),
        "k_domains": int(len(idx)),
        "union_size": int(union_mask.sum()),
        "intersection_size": int(inter_mask.sum()),
        "union_mask": union_mask,         # reuse for downstream filtering
        "intersection_mask": inter_mask,  # idem
    }

    # Optional: list the protein IDs that are in the intersection
    if protein_ids is not None:
        ids = np.asarray(protein_ids)
        multiway_info["intersection_protein_ids"] = ids[inter_mask].tolist()
        multiway_info["union_protein_ids"] = ids[union_mask].tolist()

    return summary_df, pairwise_df, multiway_info


# %%
layer_i = 8
latent_ind = 2677
buffer = 500
n_regions = 3

tail_regions_drop = find_activation_tail_regions(
    layer_i,
    latent_ind,
    all_acts=all_acts_topq,
    n_regions=n_regions,
    buffer=buffer,
    method="drop",
)

tail_regions_kneedle = find_activation_tail_regions(
    layer_i,
    latent_ind,
    all_acts=all_acts_topq,
    n_regions=n_regions,
    buffer=buffer,
    method="kneedle",
)

tail_regions_second = find_activation_tail_regions(
    layer_i,
    latent_ind,
    all_acts=all_acts_topq,
    n_regions=n_regions,
    buffer=buffer,
    method="second_derivative",
)

print("Tail region candidates (drop heuristic):")
for region in tail_regions_drop:
    print(region)

print("\nTail region candidates (Kneedle):")
for region in tail_regions_kneedle:
    print(region)

print("\nTail region candidates (second derivative):")
for region in tail_regions_second:
    print(region)

save_activation_curve(
    layer_i,
    latent_ind,
    tau=0.99,
    symlog=False,
    all_acts=all_acts_topq,
    tail_regions=tail_regions_drop, # CONNOR: get tail in this format 
    tail_label="drop tail",
    tail_color="mediumseagreen",
    tail_linecolor="mediumseagreen",
    show_plt=True,
)


save_activation_curve(
    layer_i,
    latent_ind,
    tau=0.99,
    symlog=False,
    all_acts=all_acts_topq,
    tail_regions=tail_regions_kneedle,
    tail_label="kneedle tail",
    tail_color="mediumseagreen",
    tail_linecolor="mediumseagreen",
    show_plt=True,
)

save_activation_curve(
    layer_i,
    latent_ind,
    tau=0.99,
    symlog=False,
    all_acts=all_acts_topq,
    tail_regions=tail_regions_second,
    tail_label="second derivative tail",
    tail_color="mediumseagreen",
    tail_linecolor="mediumseagreen",
    show_plt=True,
)

# %%

SUMMARY_KIND = "topq"            # just a label in the output
ALL_LATENTS = list(range(len(subset_list)))  # iterate over your ~383 latents

# Optional: domain names
entry_names = list(interpro_annotations_nonzero["ENTRY_NAME"])

# Tail enrichment output location
TAIL_RESULTS_DIR = Path("results/domain_correlation_results/tail_enrichv2")
TAIL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tail_enrichment_frames: list[pd.DataFrame] = []
tail_region_summaries: list[dict] = []
significant_domain_frames: list[pd.DataFrame] = []
errors: list[tuple[int, str]] = []

MWU_ALPHA = 0.05

high_precision_df = pd.DataFrame()
strong_tail_df = pd.DataFrame()

# --- main loop ---
for latent_i in tqdm(ALL_LATENTS, desc="Analyzing latents"):
    try:
        layer_id, latent_ind = get_layer_and_latent(latent_i, subset_list)
        flattened_id = int(subset_list[latent_i])

        scores = to_numpy(all_acts_topq)[:, latent_i].astype(float)
        scores_sorted = np.sort(scores)[::-1]
        max_val = float(scores_sorted[0]) if scores_sorted.size else 0.0
        min_val = float(scores_sorted[-1]) if scores_sorted.size else 0.0
        span_val = float(max_val - min_val)

        df_stats = analyze_latent_per_domain( # CONNOR: mwu + precision @ n_domain
            all_acts_topq,
            all_properties,
            latent_i,
            entry_names=entry_names,
            summarized=SUMMARY_KIND,
        )

        if df_stats.empty:
            continue

        df_sig = df_stats[df_stats["mwu_p_adj"] <= MWU_ALPHA]
        if df_sig.empty:
            continue
        # print(df_sig)
        # break
        domain_idxs = df_sig["domain_idx"].to_numpy(dtype=int)

        baseline_df = screen_domains_tail_enrichment( # CONNOR: odds ratio, fisher p, median_pos_percentile, tau_cover_95, tau_cover_90, tau_all_in 
            all_acts_topq,
            all_properties,
            latent_i=latent_i,
            domain_indices=domain_idxs,
            tau=0.99,
            use_bonferroni=True,
        )

        baseline_keep_cols = [
            "domain_idx",
            "median_pos_percentile",
        ]
        for cov_col in ("tau_cover_95", "tau_cover_90", "tau_all_in"):
            if cov_col in baseline_df.columns:
                baseline_keep_cols.append(cov_col)

        baseline_df = baseline_df[baseline_keep_cols]

        df_sig = df_sig.merge(baseline_df, on="domain_idx", how="left")
        df_sig["latent_i"] = latent_i
        df_sig["layer_id"] = layer_id
        df_sig["latent_ind"] = latent_ind
        df_sig["flattened_id"] = flattened_id

        significant_domain_frames.append(df_sig)

        tail_regions = find_activation_tail_regions(
            layer_id,
            latent_ind,
            all_acts=all_acts_topq,
            n_regions=3,
            buffer=200,
            method="second_derivative",
        )

        if not tail_regions:
            continue

        for rank_in_list, region in enumerate(tail_regions, start=1):
            tail_tau = float(region.get("tail_start_tau", 0.99))
            tail_tau = float(np.clip(tail_tau, 1e-6, 1.0 - 1e-6))

            tail_start_rank = int(region.get("tail_start_rank", 1))
            tail_start_idx = max(0, min(scores_sorted.size, tail_start_rank) - 1)
            above_slice = scores_sorted[:tail_start_idx]
            below_slice = scores_sorted[tail_start_idx:]
            mean_above = float(above_slice.mean()) if above_slice.size else float(max_val)
            mean_below = float(below_slice.mean()) if below_slice.size else float(min_val)
            mean_delta = float(mean_above - mean_below)

            tail_region_summaries.append(
                {
                    "latent_i": latent_i,
                    "layer_id": layer_id,
                    "latent_ind": latent_ind,
                    "flattened_id": flattened_id,
                    "tail_rank": rank_in_list,
                    "drop_rank": region.get("drop_rank"),
                    "drop_tau": region.get("drop_tau"),
                    "tail_start_rank": region.get("tail_start_rank"),
                    "tail_start_tau": tail_tau,
                    "tail_end_rank": region.get("tail_end_rank"),
                    "tail_end_tau": region.get("tail_end_tau"),
                    "drop": region.get("drop"),
                    "relative_drop": region.get("relative_drop"),
                    "score": region.get("score"),
                    "method": region.get("method"),
                    "mean_above": mean_above,
                    "mean_below": mean_below,
                    "mean_delta": mean_delta,
                    "max_min_span": span_val,
                }
            )

            df_tail = screen_domains_tail_enrichment( # CONNOR: odds ratio for each tail
                all_acts_topq,
                all_properties,
                latent_i=latent_i,
                domain_indices=domain_idxs,
                tau=tail_tau,
                use_bonferroni=True,
            )

            if df_tail.empty:
                continue

            tail_cols_keep = {
                "domain_idx": "domain_idx",
                "a_top_in": "a_tail",
                "b_top_out": "b_tail",
                "c_not_top_in": "c_tail",
                "d_not_top_out": "d_tail",
                "odds_ratio": "tail_odds_ratio",
                "fisher_p": "tail_fisher_p",
                "p_adj": "tail_p_adj",
            }
            df_tail = df_tail[list(tail_cols_keep.keys())].copy()
            df_tail.rename(columns=tail_cols_keep, inplace=True)
            df_tail = df_tail.merge(
                df_sig[["domain_idx", "entry_name"]],
                on="domain_idx",
                how="left",
            )
            df_tail["latent_i"] = latent_i
            df_tail["layer_id"] = layer_id
            df_tail["latent_ind"] = latent_ind
            df_tail["flattened_id"] = flattened_id
            df_tail["tail_rank"] = rank_in_list
            df_tail["tail_tau"] = tail_tau
            df_tail["tail_drop_rank"] = region.get("drop_rank")
            df_tail["tail_drop_tau"] = region.get("drop_tau")
            df_tail["tail_relative_drop"] = region.get("relative_drop")
            df_tail["tail_method"] = region.get("method")
            df_tail["tail_mean_above"] = mean_above
            df_tail["tail_mean_below"] = mean_below
            df_tail["tail_mean_delta"] = mean_delta
            df_tail["tail_max_min_span"] = span_val

            tail_enrichment_frames.append(df_tail)

    except Exception as exc:
        errors.append((latent_i, str(exc)))

if tail_region_summaries:
    tail_region_df = pd.DataFrame(tail_region_summaries)
    tail_region_path = TAIL_RESULTS_DIR / "tail_region_summaries_second_derivative.csv"
    tail_region_df.to_csv(tail_region_path, index=False)
    print(f"Saved tail region summaries to {tail_region_path}")

if significant_domain_frames:
    sig_domain_df = pd.concat(significant_domain_frames, ignore_index=True)
    sig_domain_path = TAIL_RESULTS_DIR / "significant_domain_pairs.csv"
    sig_domain_df.to_csv(sig_domain_path, index=False)
    print(f"Saved significant domain pairs to {sig_domain_path}")

if tail_enrichment_frames:
    tail_enrich_df = pd.concat(tail_enrichment_frames, ignore_index=True)
    tail_enrich_path = TAIL_RESULTS_DIR / "tail_enrichment_second_derivative.csv"
    tail_enrich_df.to_csv(tail_enrich_path, index=False)
    print(f"Saved tail enrichment results to {tail_enrich_path}")

if errors:
    print("Encountered errors for latents:")
    for latent_idx, msg in errors:
        print(f"  latent {latent_idx}: {msg}")

# %%
if significant_domain_frames:
    sig_domain_df = pd.concat(significant_domain_frames, ignore_index=True)
    high_precision_mask = sig_domain_df.get("precision_1x", pd.Series(dtype=float)) > 0.5
    high_precision_df = sig_domain_df.loc[high_precision_mask].copy()
    high_precision_cols = [
        "latent_i",
        "layer_id",
        "latent_ind",
        "flattened_id",
        "domain_idx",
        "entry_name",
        "precision_1x",
        # "mwu_p",
        # "mwu_p_adj",
        "median_pos_percentile",
    ]
    high_precision_cols = [col for col in high_precision_cols if col in high_precision_df.columns]
    high_precision_df = high_precision_df[high_precision_cols]
    high_precision_df.sort_values(["latent_i", "domain_idx"], inplace=True)

    if not high_precision_df.empty:
        high_precision_path = TAIL_RESULTS_DIR / "high_precision_latent_domain_pairs.csv"
        high_precision_df.to_csv(high_precision_path, index=False)
        print(f"High-precision latent/domain pairs saved to {high_precision_path}")
        print(
            "High-precision latents:",
            high_precision_df[["layer_id", "latent_ind"]].drop_duplicates().shape[0],
        )
    else:
        print("High-precision latents: 0")

if tail_enrichment_frames:
    tail_enrich_df = pd.concat(tail_enrichment_frames, ignore_index=True)
    tail_filter = (
        (tail_enrich_df["tail_tau"] > 0.85)
        & (tail_enrich_df["tail_p_adj"] < 0.05)
        & ((tail_enrich_df["a_tail"] > tail_enrich_df["c_tail"]) | (tail_enrich_df["a_tail"] > tail_enrich_df["b_tail"]))
        & (tail_enrich_df["tail_mean_delta"] > 0.5 * tail_enrich_df["tail_max_min_span"])
    )
    strong_tail_df = tail_enrich_df.loc[tail_filter].copy()
    strong_tail_cols = [
        "latent_i",
        "layer_id",
        "latent_ind",
        "flattened_id",
        "domain_idx",
        "entry_name",
        "tail_rank",
        "tail_tau",
        "tail_drop_rank",
        "tail_drop_tau",
        "tail_relative_drop",
        "a_tail",
        "b_tail",
        "c_tail",
        "d_tail",
        "tail_odds_ratio",
        "tail_fisher_p",
        "tail_p_adj",
        "tail_mean_above",
        "tail_mean_below",
        "tail_mean_delta",
        "tail_max_min_span",
    ]
    strong_tail_cols = [col for col in strong_tail_cols if col in strong_tail_df.columns]
    strong_tail_df = strong_tail_df[strong_tail_cols]
    strong_tail_df.sort_values(["latent_i", "domain_idx", "tail_rank"], inplace=True)

    if not strong_tail_df.empty:
        strong_tail_path = TAIL_RESULTS_DIR / "strong_tail_latent_domain_pairs.csv"
        strong_tail_df.to_csv(strong_tail_path, index=False)
        print(f"Strong tail latent/domain pairs saved to {strong_tail_path}")
        print(
            "Strong-tail latents:",
            strong_tail_df[["layer_id", "latent_ind"]].drop_duplicates().shape[0],
        )
    else:
        print("Strong-tail latents: 0")

    if significant_domain_frames:
        overlap_df = high_precision_df.merge(
            strong_tail_df,
            on=["latent_i", "domain_idx"],
            suffixes=("_sig", "_tail"),
        )
        union_df = pd.merge(
            high_precision_df,
            strong_tail_df,
            on=["latent_i", "domain_idx"],
            suffixes=("_sig", "_tail"),
            how="outer",
        )

        if not overlap_df.empty:
            overlap_path = TAIL_RESULTS_DIR / "latent_domain_overlap.csv"
            overlap_df.to_csv(overlap_path, index=False)
            print(f"Overlap latent/domain pairs saved to {overlap_path}")
            print(
                "Overlap latents:",
                overlap_df[["layer_id_sig", "latent_ind_sig"]].dropna().drop_duplicates().shape[0],
            )

        if not union_df.empty:
            union_path = TAIL_RESULTS_DIR / "latent_domain_union.csv"
            union_df.to_csv(union_path, index=False)
            print(f"Union latent/domain pairs saved to {union_path}")
            dedup_union = union_df.copy()
            combined_layer = dedup_union[["layer_id_sig", "layer_id_tail"]].bfill(axis=1).iloc[:, 0]
            combined_latent = dedup_union[["latent_ind_sig", "latent_ind_tail"]].bfill(axis=1).iloc[:, 0]
            combined_df = pd.DataFrame({
                "layer_id": combined_layer,
                "latent_ind": combined_latent,
            }).dropna().drop_duplicates()
            print("Union latents:", combined_df.shape[0])
        else:
            print("Union latents: 0")
    elif not high_precision_df.empty:
        overlap_df = pd.DataFrame(columns=["latent_i", "domain_idx"])
        union_df = high_precision_df.copy()
        union_path = TAIL_RESULTS_DIR / "latent_domain_union.csv"
        union_df.to_csv(union_path, index=False)
        print(f"Union latent/domain pairs saved to {union_path}")
        print(
            "Union latents:",
            union_df[["layer_id", "latent_ind"]].drop_duplicates().shape[0],
        )
    else:
        print("Union latents: 0")


# %%
n_regions = 3
buffer = 200
for latent_i in tqdm(ALL_LATENTS, desc="Analyzing latents"):
    try:
        layer_id, latent_ind = get_layer_and_latent(latent_i, subset_list)
        flattened_id = int(subset_list[latent_i])
        tail_regions_second = find_activation_tail_regions(
            layer_id,
            latent_ind,
            all_acts=all_acts_topq,
            n_regions=n_regions,
            buffer=buffer,
            method="second_derivative",
        )
        save_activation_curve(
            layer_id,
            latent_ind,
            tau=0.99,
            symlog=False,
            all_acts=all_acts_topq,
            tail_regions=tail_regions_second,
            tail_label="second derivative tail",
            tail_color="mediumseagreen",
            tail_linecolor="mediumseagreen",
            show_plt=False,
            out_dir="/work/pi_jensen_umass_edu/jnainani_umass_edu/plm_circuits/notebooks/domain_corr/results/domain_correlation_results/act_curves/",
        )
    except Exception as exc:
        pass
# %%
