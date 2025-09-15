import sys
sys.path.append('../')
sys.path.append('../plm_circuits')

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
import numpy as np
import matplotlib.pyplot as plt
import collections
from typing import Dict, List, Tuple, Optional

# Lazy caches to avoid heavy loads on import
_ESM_CACHE = None  # type: ignore[var-annotated]
_SAE_CACHE: Dict[int, object] = {}

import requests
import pandas as pd
def fetch_uniprot_entry(acc: str) -> dict:
    url = f"https://rest.uniprot.org/uniprotkb/{acc}"
    r = requests.get(url, headers={"Accept": "application/json"}, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_uniprot_fasta(acc: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text

def get_sae(layer: int):
    global _SAE_CACHE
    if layer in _SAE_CACHE:
        return _SAE_CACHE[layer]
    try:
        sae_model = load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=layer, device=get_device())
    except NameError as e:
        raise RuntimeError("load_sae_prot(...) not found. Edit the import block near the top to point to your module.") from e
    _SAE_CACHE[layer] = sae_model
    return sae_model

def get_esm():
    # Uses your predefined loader
    global _ESM_CACHE
    if _ESM_CACHE is not None:
        return _ESM_CACHE
    try:
        _ESM_CACHE = load_esm(33, device=get_device())
    except NameError as e:
        raise RuntimeError("load_esm(...) not found. Edit the import block near the top to point to your module.") from e
    return _ESM_CACHE

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_latent_activations_on_sequence(
    sequence: str,
    layer: int,
    latent_idx: int,
    use_masked_latents: bool = True,
    use_error: bool = False,
) -> np.ndarray:
    """Runs your SAE hook at the target layer and returns a 1D vector (length L) for the chosen latent."""
    esm_transformer, batch_converter, esm2_alphabet = get_esm()
    device = get_device()
    # tokenization
    _, _, batch_tokens_BL = batch_converter([(1, sequence)])
    batch_tokens_BL = batch_tokens_BL.to(device)
    batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)
    cache_latents = True
    if use_masked_latents:
        cache_latents = False
    # SAE hook
    sae_model = get_sae(layer)
    try:
        hook = SAEHookProt(
            sae=sae_model,
            mask_BL=batch_mask_BL,
            cache_latents=cache_latents,
            cache_masked_latents=use_masked_latents,
            layer_is_lm=False,
            calc_error=False,
            use_error=use_error,
        )
    except NameError as e:
        raise RuntimeError("SAEHookProt not found. Edit imports at the top to your module.") from e
    handle = esm_transformer.esm.encoder.layer[layer].register_forward_hook(hook)
    with torch.no_grad():
        _ = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]
    handle.remove()
    # choose source array
    arr = None
    if use_error and hasattr(sae_model, "error_term") and sae_model.error_term is not None:
        arr = sae_model.error_term  # shape [B, L, F]
        arr = np.asarray(arr)[0]  # [L, F]
        # print(f"error_term shape: {arr.shape}")
    elif use_masked_latents and hasattr(sae_model, "masked_latents") and sae_model.masked_latents is not None:
        arr = sae_model.masked_latents  # [L, F] or [B, L, F]
        # print(f"masked_latents shape: {arr.shape}")
    elif cache_latents and hasattr(sae_model, "feature_acts") and sae_model.feature_acts is not None:
        arr = sae_model.feature_acts  # [L, F] or [B, L, F]
        # print(f"feature_acts shape: {arr.shape}")
    # else:
    #     if use_masked_latents and hasattr(sae_model, "cache_masked_latents") and sae_model.cache_masked_latents is not None:
    #         arr = sae_model.cache_masked_latents  # [L, F] or [B, L, F]
    #     else:
    #         arr = sae_model.feature_acts  # [L, F] or [B, L, F]
    arr = np.asarray(arr.cpu())
    if arr.ndim == 3:
        arr = arr[0]
    L, F = arr.shape
    if not (0 <= latent_idx < F):
        raise ValueError(f"latent_idx {latent_idx} out of range [0, {F-1}]")
    return arr[:, latent_idx]

def fetch_interpro_protein(acc: str, page_size: int = 500) -> dict:
    # All database entries (Pfam + others) mapped to this UniProt accession
    # url = f"https://www.ebi.ac.uk/interpro/api/protein/uniprot/{acc}" https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/I1RF61
    url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{acc}"
    r = requests.get(url, timeout=30) #r = requests.get(url, params={"page_size": page_size}, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_interpro_domains(interpro_json: dict) -> pd.DataFrame:
    rows = []
    for rec in interpro_json.get("results", []):
        meta = rec.get("metadata", {}) or {}
        entry = meta.get("accession")
        name = meta.get("name")
        db = meta.get("source_database")

        # NEW schema: entry → proteins[] → entry_protein_locations[] → fragments[]
        if "proteins" in rec:
            for prot in rec.get("proteins") or []:
                for loc in prot.get("entry_protein_locations") or []:
                    for frag in loc.get("fragments") or []:
                        rows.append({
                            "entry": entry,
                            "name": name,
                            "db": db,
                            "start": int(frag.get("start")) if frag.get("start") is not None else None,
                            "end": int(frag.get("end")) if frag.get("end") is not None else None,
                        })
        # OLD schema fallback (if you ever switch back)
        else:
            for loc in rec.get("entry_protein_locations", []) or []:
                for frag in loc.get("fragments") or []:
                    rows.append({
                        "entry": entry,
                        "name": name,
                        "db": db,
                        "start": int(frag.get("start")) if frag.get("start") is not None else None,
                        "end": int(frag.get("end")) if frag.get("end") is not None else None,
                    })

    if not rows:
        return pd.DataFrame(columns=["entry", "name", "db", "start", "end"])
    df = pd.DataFrame(rows).sort_values(["start", "end"]).reset_index(drop=True)
    return df

import helpers.protein_viz_utils as viz
def render_sequence_heatmap(
    seq: str,
    values: np.ndarray,
    colormap_fn=viz.mono_colormap_fn,
    wrap: int = 120,  # ignored (kept for API compatibility)
    vmin: float | None = None,
    vmax: float | None = None,
    height: int | None = None,
    title: str = "Sequence heatmap",
):
    import html, uuid
    from IPython.display import HTML
    
    L = len(seq) if seq else 0
    if values is None or L == 0:
        print("No sequence/activations to render.")
        return
    vmin = float(np.min(values)) if vmin is None else float(vmin)
    vmax = float(np.max(values)) if vmax is None else float(vmax)

    # Unique id so multiple boxes don't clash across reruns
    uid = f"seqbox-{uuid.uuid4().hex[:8]}"

    spans = []
    for i, (aa, val) in enumerate(zip(seq, values)):
        color = colormap_fn(float(val), vmin, vmax)
        aa_esc = html.escape(aa)
        # data-* attrs let JS show an index+value tooltip
        spans.append(
            f'<span class="res" data-i="{i}" data-val="{float(val):.6f}" '
            f'style="background:{color}">{aa_esc}</span>'
        )

    rows_html = "".join(spans)
    # Single-row strip, fixed compact height
    h = height if height is not None else 82

    html_block = f'''
    <div id="{uid}" class="seqbox">
      <div class="title">{html.escape(title)}</div>
      <div class="strip">{rows_html}</div>
      <div class="tip" style="display:none;"></div>
    </div>
    <style>
    #{uid}.seqbox {{
        overflow-x: auto; overflow-y: hidden;
        border: 1px solid #eee; border-radius: 8px; padding: 6px 8px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        background: #fff;
    }}
    #{uid} .title {{ font-size: 0.9rem; color: #444; margin-bottom: 6px; white-space: nowrap; }}
    #{uid} .strip {{
        white-space: nowrap;
    }}
    #{uid} .strip .res {{
        display:inline-block; margin:0 1px 0 0; padding:1px 2px; border-radius:3px;
        color:#111; line-height:1.3rem; font-size:0.95rem; cursor:default;
    }}
    #{uid} .strip .res:hover {{
        outline: 1px solid rgba(0,0,0,0.25);
    }}
    #{uid} .tip {{
        position: fixed; z-index: 9999;
        background: rgba(0,0,0,0.85); color: #fff; padding: 4px 6px;
        border-radius: 4px; font-size: 12px; pointer-events: none;
        box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }}
    </style>
    <script>
    (function() {{
        const root = document.getElementById("{uid}");
        if (!root) return;
        const tip = root.querySelector(".tip");
        const strip = root.querySelector(".strip");
        const show = (x, y, text) => {{
            tip.style.display = "block";
            tip.style.left = (x + 12) + "px";
            tip.style.top  = (y + 12) + "px";
            tip.textContent = text;
        }};
        const hide = () => {{ tip.style.display = "none"; }};
        strip.addEventListener("mousemove", (e) => {{
            const t = e.target;
            if (t && t.classList && t.classList.contains("res")) {{
                const i = t.getAttribute("data-i");
                const v = Number(t.getAttribute("data-val")).toFixed(3);
                show(e.clientX, e.clientY, i + ": " + v);
            }} else {{
                hide();
            }}
        }});
        strip.addEventListener("mouseleave", hide);
    }})();
    </script>
    '''
    return HTML(html_block)

def draw_sequence_tracks(
    L: int,
    domains_df: pd.DataFrame,
    activations: Optional[np.ndarray] = None,
    coverage_segments: Optional[List[Tuple[int, int]]] = None,
    title: str = "Sequence tracks",
    sequence_letters: Optional[str] = None,
    show_bars_when_activations: bool = True,
    coverage_prefix: int = 0,
    return_png_base64: bool = False,
    dpi: int = 150,
):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import textwrap

    # Greedy lane assignment to avoid overlapping rectangles
    intervals = []  # (idx, start, end, label)
    if domains_df is not None and len(domains_df) > 0:
        for idx, r in domains_df.iterrows():
            try:
                s, e = int(r.start), int(r.end)
            except Exception:
                continue
            if pd.isna(s) or pd.isna(e):
                continue
            label = r.get("name") or r.get("entry") or ""
            intervals.append((idx, int(s), int(e), str(label)))

    intervals_sorted = sorted(intervals, key=lambda x: (x[1], x[2]))
    lane_end_positions: List[int] = []
    row_to_lane: Dict[int, int] = {}
    for idx, s, e, _ in intervals_sorted:
        placed = False
        for lane_idx, last_end in enumerate(lane_end_positions):
            # Non-overlap if current start is strictly greater than last end (inclusive ranges)
            if s > last_end:
                row_to_lane[idx] = lane_idx
                lane_end_positions[lane_idx] = e
                placed = True
                break
        if not placed:
            row_to_lane[idx] = len(lane_end_positions)
            lane_end_positions.append(e)

    num_lanes = max(row_to_lane.values()) + 1 if row_to_lane else 1

    # Layout scaling - make rectangles very small and stack them
    rect_height = 0.08  # Very small height for each domain rectangle
    lane_spacing = 0.12  # Small spacing between lanes
    domains_section_height = max(0.5, num_lanes * lane_spacing)  # Minimum 0.5, scales with lanes
    
    # Dynamic plot height calculation
    base_h = 2.5 if activations is None else 4.5
    extra_h = max(0, domains_section_height - 0.5)  # Extra height for domains beyond base
    fig, ax = plt.subplots(figsize=(12, base_h + extra_h))

    # Draw main sequence line at the bottom of domains section
    sequence_line_y = domains_section_height - 0.1
    ax.plot([1, L], [sequence_line_y, sequence_line_y], lw=4, alpha=0.2, color='gray')
    
    # Draw stacked domain rectangles
    for idx, r in domains_df.iterrows():
        if pd.isna(r.start) or pd.isna(r.end):
            continue
        s, e = int(r.start)+coverage_prefix, int(r.end)+coverage_prefix
        
        # Get lane for this domain and calculate y position
        lane = row_to_lane.get(idx, 0)
        rect_y = lane * lane_spacing
        
        # Draw small rectangle in assigned lane
        ax.add_patch(
            plt.Rectangle((s, rect_y), max(1, e - s + 1), rect_height, alpha=0.6)
        )
        
        # Position label inside the rectangle center
        label_y = rect_y + rect_height / 2
        txt = str(r.get("name") or r.get("entry") or "")
        
        # Truncate overly long labels roughly to fit the rectangle span
        try:
            span = max(1, e - s + 1)
            max_chars = int(max(6, min(30, span * 0.3)))
        except Exception:
            max_chars = 20
        try:
            disp_txt = textwrap.shorten(txt, width=max_chars, placeholder="…")
        except Exception:
            disp_txt = txt
            
        ax.text(
            (s + e) / 2,
            label_y,
            disp_txt,
            ha="center",
            va="center",
            fontsize=7,
            clip_on=False,
            # path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
            color='black',
            weight='bold',
            alpha=0.5
        )
    
    # Draw coverage segments at the very bottom
    if coverage_segments:
        for s, e in coverage_segments:
            ax.add_patch(plt.Rectangle((s, -0.05), max(1, e - s + 1), 0.03, color='red', alpha=0.7))
    
    ax.set_xlim(1, L)
    ax.set_ylim(-0.1, domains_section_height + (0.8 if activations is not None else 0.2))
    ax.set_yticks([])
    
    # Handle activations on separate axis
    if activations is not None:
        xs = np.arange(1, L + 1)
        if show_bars_when_activations:
            # Draw bars on a twin axis, optionally label with letters
            ax2 = ax.twinx()
            vals = np.asarray(activations, dtype=float)
            # Align values length to L
            if vals.shape[0] > L:
                vals = vals[:L]
            elif vals.shape[0] < L:
                pad = np.full(L - vals.shape[0], np.nan, dtype=float)
                vals = np.concatenate([vals, pad], axis=0)
            # Normalize letters length to L if provided
            seq_letters = sequence_letters
            if seq_letters is not None:
                if len(seq_letters) < L:
                    seq_letters = seq_letters + ("X" * (L - len(seq_letters)))
                elif len(seq_letters) > L:
                    seq_letters = seq_letters[:L]
            colors = None
            if seq_letters is not None:
                uniq = sorted(set(seq_letters))
                color_map = {aa: f"C{i % 10}" for i, aa in enumerate(uniq)}
                colors = [color_map.get(aa, "C0") for aa in seq_letters]
            ax2.bar(xs, vals, color=colors, width=0.8, alpha=0.9)
            if seq_letters is not None:
                for x, aa, v in zip(xs, seq_letters, vals):
                    va = "bottom" if (np.isnan(v) or v >= 0) else "top"
                    offset = 0.01 if (np.isnan(v) or v >= 0) else -0.01
                    if v > 0:
                        ax2.text(x, (0.0 if np.isnan(v) else v) + offset, aa, ha="center", va=va, fontsize=7)
            ax2.set_ylabel("Activation")
            ax2.grid(True, axis="y", alpha=0.2)
        else:
            ax2 = ax.twinx()
            ax2.plot(xs, activations, lw=1)
            ax2.set_ylabel("Activation")
            ax2.grid(True, alpha=0.2)
    ax.set_title(title)
    if return_png_base64:
        import io, base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    else:
        plt.show()

def draw_uniprot_features(
    L: int,
    features_list: List[Dict],
    activations: Optional[np.ndarray] = None,
    title: str = "UniProt Features",
    sequence_letters: Optional[str] = None,
    show_bars_when_activations: bool = True,
    coverage_prefix: int = 0,
    return_png_base64: bool = False,
    dpi: int = 150,
):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.patches import Patch
    import textwrap
    
    # Extract intervals from UniProt features format
    intervals = []  # (idx, start, end, label, feature_type)
    for idx, feature in enumerate(features_list):
        try:
            # Extract start and end from nested location structure
            start_val = feature['location']['start']['value']
            end_val = feature['location']['end']['value']
            s, e = int(start_val), int(end_val)
        except (KeyError, ValueError, TypeError):
            continue
            
        # Get feature type and description
        feature_type = feature.get('type', '')
        description = feature.get('description', '')
        
        # Use description as the label, fallback to feature_type
        label = description if description else feature_type
            
        intervals.append((idx, s, e, str(label), feature_type))

    # Greedy lane assignment to avoid overlapping rectangles
    intervals_sorted = sorted(intervals, key=lambda x: (x[1], x[2]))
    lane_end_positions: List[int] = []
    row_to_lane: Dict[int, int] = {}
    
    for idx, s, e, _, _ in intervals_sorted:
        placed = False
        for lane_idx, last_end in enumerate(lane_end_positions):
            # Non-overlap if current start is strictly greater than last end
            if s > last_end:
                row_to_lane[idx] = lane_idx
                lane_end_positions[lane_idx] = e 
                placed = True
                break
        if not placed:
            row_to_lane[idx] = len(lane_end_positions)
            lane_end_positions.append(e)

    num_lanes = max(row_to_lane.values()) + 1 if row_to_lane else 1

    # Layout scaling - make rectangles very small and stack them
    rect_height = 0.08  # Very small height for each feature rectangle
    lane_spacing = 0.12  # Small spacing between lanes
    features_section_height = max(0.5, num_lanes * lane_spacing)  # Minimum 0.5, scales with lanes
    
    # Dynamic plot height calculation
    base_h = 2.5 if activations is None else 4.5
    extra_h = max(0, features_section_height - 0.5)  # Extra height for features beyond base
    fig, ax = plt.subplots(figsize=(14, base_h + extra_h))  # Wider to accommodate legends

    # Draw main sequence line at the bottom of features section
    sequence_line_y = features_section_height - 0.1
    ax.plot([1, L], [sequence_line_y, sequence_line_y], lw=4, alpha=0.2, color='gray')
    
    # Color map for different feature types
    feature_types = sorted(list(set([f[4] for f in intervals])))  # Sort for consistent legend order
    color_map = {ftype: f"C{i % 10}" for i, ftype in enumerate(feature_types)}
    
    # Draw stacked feature rectangles with description labels on top
    for idx, s, e, label, feature_type in intervals:
        # Get lane for this feature and calculate y position
        lane = row_to_lane.get(idx, 0)
        rect_y = lane * lane_spacing
        s, e = s+coverage_prefix, e+coverage_prefix
        # Get color based on feature type
        rect_color = color_map.get(feature_type, 'C0')
        
        # Draw small rectangle in assigned lane
        ax.add_patch(
            plt.Rectangle((s, rect_y), max(1, e - s + 1), rect_height, 
                         alpha=0.5, color=rect_color)
        )
        
        # Add description text above the rectangle
        label_y = rect_y + 0.01  # Position above the rectangle
        
        # Truncate overly long labels to fit the rectangle span
        try:
            span = max(1, e - s + 1)
            max_chars = int(max(10, min(50, span * 0.75)))
        except Exception:
            max_chars = 15
        try:
            disp_txt = textwrap.shorten(label, width=max_chars, placeholder="")
        except Exception:
            disp_txt = label
            
        # Only show text if rectangle is wide enough
        if e - s + 1 >= 8:  # Only show text for rectangles >= 8 units wide
            ax.text(
                (s + e) / 2,
                label_y,
                disp_txt,
                ha="center",
                va="bottom",
                fontsize=9,
                clip_on=False,
                path_effects=[pe.withStroke(linewidth=1, foreground="white")],
                color='black',
                weight='bold',
                alpha=0.5
            )
        else:
            x_center = (s + e) / 2
            # Keep the label within the x-limits; nudge to the right of the feature
            x_text = min(L - 0.5, e + 2)
            y_text = rect_y + rect_height + 0.04
            ax.annotate(
                disp_txt,
                xy=(x_center, rect_y + rect_height / 2),
                xytext=(x_text, y_text),
                textcoords="data",
                ha="left",
                va="bottom",
                fontsize=9,
                color='black',
                alpha=0.8,
                path_effects=[pe.withStroke(linewidth=1, foreground="white")],
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, shrinkA=0, shrinkB=0),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5),
                clip_on=False,
            )
    ax.set_xlim(1, L)
    ax.set_ylim(-0.1, features_section_height + (0.8 if activations is not None else 0.2))
    ax.set_yticks([])
    
    # Create legend for feature types only
    if feature_types:
        legend_elements = [Patch(facecolor=color_map[ftype], alpha=0.7, label=ftype) 
                          for ftype in feature_types]
        legend1 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
                           title="Feature Types", fontsize=9, title_fontsize=10)
        ax.add_artist(legend1)
    
    # Handle activations on separate axis
    ax2 = None
    if activations is not None:
        xs = np.arange(1, L + 1)
        if show_bars_when_activations:
            # Draw bars on a twin axis, optionally label with letters
            ax2 = ax.twinx()
            vals = np.asarray(activations, dtype=float)
            # Align values length to L
            if vals.shape[0] > L:
                vals = vals[:L]
            elif vals.shape[0] < L:
                pad = np.full(L - vals.shape[0], np.nan, dtype=float)
                vals = np.concatenate([vals, pad], axis=0)
            # Normalize letters length to L if provided
            seq_letters = sequence_letters
            if seq_letters is not None:
                if len(seq_letters) < L:
                    seq_letters = seq_letters + ("X" * (L - len(seq_letters)))
                elif len(seq_letters) > L:
                    seq_letters = seq_letters[:L]
            colors = None
            if seq_letters is not None:
                uniq = sorted(set(seq_letters))
                color_map_seq = {aa: f"C{i % 10}" for i, aa in enumerate(uniq)}
                colors = [color_map_seq.get(aa, "C0") for aa in seq_letters]
            ax2.bar(xs, vals, color=colors, width=0.8, alpha=0.9)
            if seq_letters is not None:
                for x, aa, v in zip(xs, seq_letters, vals):
                    va = "bottom" if (np.isnan(v) or v >= 0) else "top"
                    offset = 0.01 if (np.isnan(v) or v >= 0) else -0.01
                    if v > 0:
                        ax2.text(x, (0.0 if np.isnan(v) else v) + offset, aa, ha="center", va=va, fontsize=7)
            ax2.set_ylabel("Activation")
            ax2.grid(True, axis="y", alpha=0.2)
        else:
            ax2 = ax.twinx()
            ax2.plot(xs, activations, lw=1)
            ax2.set_ylabel("Activation")
            ax2.grid(True, alpha=0.2)
    
    ax.set_title(title)
    plt.tight_layout()  # Better layout management
    if return_png_base64:
        import io, base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    else:
        plt.show()

def generate_latent_html_report(
    layer: int,
    latent_idx: int,
    interpro_ids: List[str],
    samples_per_domain: int = 2,
    seed: Optional[int] = 42,
    out_path: Optional[str] = None,
    page_title: Optional[str] = None,
    max_domains: Optional[int] = None,
) -> str:
    """
    Generate a single HTML page for a given (layer, latent) across InterPro IDs.

    For each InterPro ID, sample `samples_per_domain` reviewed UniProt proteins,
    and for each protein render:
      1) Sequence heatmap (HTML snippet)
      2) InterPro domain tracks (PNG image embedded)
      3) UniProt feature tracks (PNG image embedded)
      4) 3D structure viewer (HTML snippet)

    Returns the complete HTML as a string. If `out_path` is provided, writes
    the HTML to that file as well.
    """
    import html as html_escape
    from datetime import datetime

    rng_seed = seed
    # HTML header and simple styles
    title = page_title or f"Latent report: L{layer} - {latent_idx}"
    parts: List[str] = []
    parts.append(
        f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html_escape.escape(title)}</title>
  <style>
    body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; color:#1a1a1a; background:#fafafa; margin: 0; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 16px 18px 48px; }}
    h1 {{ font-size: 1.75rem; margin: 12px 0 6px; }}
    h2 {{ font-size: 1.35rem; margin: 24px 0 8px; color:#333; }}
    h3 {{ font-size: 1.05rem; margin: 14px 0 6px; color:#444; }}
    .domain-block {{ background:#fff; border:1px solid #ececec; border-radius: 10px; padding: 12px 14px; margin: 18px 0; }}
    .protein-block {{ border-top:1px dashed #eee; padding-top: 10px; margin-top: 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    .img-wrap img {{ max-width: 100%; height: auto; border:1px solid #f0f0f0; border-radius: 6px; background:#fff; }}
    .meta {{ font-size: 0.9rem; color:#666; margin:4px 0 10px; }}
    .subtle {{ color:#777; }}
    .badge {{ display:inline-block; background:#eef6ff; color:#0b62b8; border:1px solid #d6e9ff; border-radius:999px; padding:2px 8px; font-size: 0.8rem; margin-left: 8px; }}
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>{html_escape.escape(title)}</h1>
    <div class=\"meta\">Generated {html_escape.escape(datetime.now().isoformat(timespec='seconds'))}</div>
  """
    )

    domain_iter = interpro_ids[:max_domains] if (max_domains is not None) else interpro_ids
    for domain_id in domain_iter:
        # Domain header with name and hyperlink
        try:
            db, acc = _infer_db_and_acc(domain_id)
        except Exception:
            db, acc = ("interpro", str(domain_id))
        domain_url = f"https://www.ebi.ac.uk/interpro/entry/{db}/{requests.utils.quote(acc, safe='')}"  # noqa: E501
        domain_name = None
        try:
            if acc.upper().startswith("IPR"):
                meta = interpro_entry_meta(acc)
                domain_name = meta.get("name") or meta.get("short_name")
        except Exception:
            domain_name = None
        name_html = f"{html_escape.escape(domain_name)} " if domain_name else ""
        id_html = f"<a href=\"{domain_url}\" target=\"_blank\" rel=\"noopener\">{html_escape.escape(acc)}</a>"
        parts.append(
            f"<div class=\"domain-block\"><h2>Domain {name_html}<span class=\"badge\">{id_html}</span></h2>"
        )

        # Sample proteins containing this domain
        try:
            sampled = sample_uniprot_ids(domain_id, n=samples_per_domain, seed=rng_seed)
        except Exception as e:
            parts.append(f"<div class=\"subtle\">Sampling error: {html_escape.escape(str(e))}</div>")
            sampled = []

        if not sampled:
            parts.append("<div class=\"subtle\">No proteins sampled.</div>")
            parts.append("</div>")
            continue

        for uniprot_acc in sampled:
            # Fetch metadata and sequence
            try:
                fasta = fetch_uniprot_fasta(uniprot_acc)
                lines = [ln for ln in fasta.splitlines() if ln and not ln.startswith(">")]
                uniprot_seq = "".join(lines).strip()
            except Exception as e:
                parts.append(f"<div class=\"protein-block\"><h3>{html_escape.escape(uniprot_acc)} <span class=\"badge\">fetch failed</span></h3><div class=\"subtle\">{html_escape.escape(str(e))}</div></div>")
                continue

            try:
                uniprot_meta = fetch_uniprot_entry(uniprot_acc)
            except Exception as e:
                uniprot_meta = {"features": []}

            try:
                interpro_json = fetch_interpro_protein(uniprot_acc)
                interpro_df = parse_interpro_domains(interpro_json)
            except Exception:
                interpro_df = pd.DataFrame(columns=["entry","name","db","start","end"])

            # Compute latent activations
            try:
                computed_act = compute_latent_activations_on_sequence(
                    sequence=uniprot_seq,
                    layer=layer,
                    latent_idx=int(latent_idx),
                    use_masked_latents=True,
                    use_error=False,
                )
            except Exception as e:
                parts.append(f"<div class=\"protein-block\"><h3>{html_escape.escape(uniprot_acc)} <span class=\"badge\">activation failed</span></h3><div class=\"subtle\">{html_escape.escape(str(e))}</div></div>")
                continue

            # Align with raw sequence (drop BOS/EOS)
            vals = np.asarray(computed_act)[1:-1]
            L = len(uniprot_seq)
            if vals.shape[0] > L:
                vals = vals[:L]
            elif vals.shape[0] < L:
                pad = np.full(L - vals.shape[0], np.nan, dtype=float)
                vals = np.concatenate([vals, pad], axis=0)

            # Build per-protein block with UniProt hyperlink
            up_url = f"https://www.uniprot.org/uniprotkb/{requests.utils.quote(uniprot_acc, safe='')}"
            up_html = f"<a href=\"{up_url}\" target=\"_blank\" rel=\"noopener\">{html_escape.escape(uniprot_acc)}</a>"
            parts.append(f"<div class=\"protein-block\"><h3>{up_html}</h3>")

            # 1) Sequence heatmap (HTML snippet)
            try:
                hm_html_obj = render_sequence_heatmap(
                    uniprot_seq,
                    vals,
                    title=f"Sequence heatmap for {uniprot_acc} (L{layer}-{latent_idx})",
                )
                hm_html = getattr(hm_html_obj, "data", str(hm_html_obj))
                parts.append(hm_html)
            except Exception as e:
                parts.append(f"<div class=\"subtle\">Heatmap error: {html_escape.escape(str(e))}</div>")

            # 2) InterPro domain tracks (PNG)
            try:
                png_b64 = draw_sequence_tracks(
                    L,
                    interpro_df,
                    activations=vals,
                    sequence_letters=uniprot_seq,
                    title=f"InterPro features for {uniprot_acc}, under L{layer}-{latent_idx}",
                    return_png_base64=True,
                )
                parts.append(f"<div class=\"img-wrap\"><img alt=\"InterPro tracks\" src=\"data:image/png;base64,{png_b64}\" /></div>")
            except Exception as e:
                parts.append(f"<div class=\"subtle\">InterPro plot error: {html_escape.escape(str(e))}</div>")

            # 3) UniProt features tracks (PNG)
            try:
                feat_list = uniprot_meta.get("features", []) if isinstance(uniprot_meta, dict) else []
                png_b64 = draw_uniprot_features(
                    L,
                    feat_list,
                    activations=vals,
                    sequence_letters=uniprot_seq,
                    title=f"UniProt features for {uniprot_acc}, under L{layer}-{latent_idx}",
                    return_png_base64=True,
                )
                parts.append(f"<div class=\"img-wrap\"><img alt=\"UniProt features\" src=\"data:image/png;base64,{png_b64}\" /></div>")
            except Exception as e:
                parts.append(f"<div class=\"subtle\">UniProt plot error: {html_escape.escape(str(e))}</div>")

            # 4) 3D structure viewer (HTML snippet)
            try:
                v = viz.view_single_protein(
                    uniprot_id=uniprot_acc,
                    chain_id="A",
                    values_to_color=list(np.nan_to_num(vals, nan=0.0)),
                    colormap_fn=viz.mono_colormap_fn,
                    default_color="white",
                    pymol_params={"width": 700, "height": 520},
                )
                struct_html = v._make_html()
                parts.append(struct_html)
            except Exception as e:
                parts.append(f"<div class=\"subtle\">3D viewer error: {html_escape.escape(str(e))}</div>")

            parts.append("</div>")  # end protein-block

        parts.append("</div>")  # end domain-block

    parts.append("</div></body></html>")

    html_out = "\n".join(parts)
    if out_path:
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_out)
        except Exception:
            pass
    return html_out
    
import time, random, requests
from typing import Iterable, List, Tuple, Optional

BASE = "https://www.ebi.ac.uk/interpro/api"

def _infer_db_and_acc(identifier: str, db: Optional[str] = None) -> Tuple[str, str]:
    """Infer InterPro member DB from the identifier if not provided."""
    if db:
        return db.lower(), identifier
    x = identifier.upper()
    if x.startswith("PF"):      # Pfam family
        return "pfam", x
    if x.startswith("IPR"):     # InterPro integrated entry
        return "interpro", x
    if x.startswith("G3DSA"):   # CATH-Gene3D superfamily
        return "cathgene3d", identifier
    if x.startswith("SSF"):     # SUPERFAMILY (SCOP-derived)
        return "ssf", x
    raise ValueError("Couldn't infer database; pass db= one of {'pfam','interpro','cathgene3d','ssf'}.")

def _get_json(url: str, *, max_tries: int = 6, backoff: float = 0.8):
    """GET with polite retries for 202/5xx (InterPro often warms caches)."""
    for i in range(max_tries):
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            return r.json()
        # InterPro may return 202 Accepted or transient 5xx while caching
        if r.status_code in (202, 502, 503, 504):
            time.sleep(backoff * (2 ** i) + random.random() * 0.2)
            continue
        r.raise_for_status()
    raise RuntimeError(f"InterPro API not ready after {max_tries} tries: {url}")

def iter_reviewed_proteins_by_entry(db: str, acc: str, page_size: int = 200, max_pages: int = 10) -> Iterable[dict]:
    """
    Yield protein records (JSON 'metadata' blocks) for reviewed UniProt proteins
    matching an entry (Pfam/IPR/CATH-Gene3D/SUPERFAMILY).
    """
    url = f"{BASE}/protein/reviewed/entry/{db}/{requests.utils.quote(acc, safe='')}/?page_size={page_size}"
    pages = 0
    while url and pages < max_pages:
        data = _get_json(url)
        for row in data.get("results", []):
            yield row.get("metadata", row)
        url = data.get("next")
        pages += 1

def sample_uniprot_ids(
    identifier: str,
    n: int = 50,
    db: Optional[str] = None,
    page_size: int = 200,
    max_pages: int = 10,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Return up to n reviewed UniProt accessions for proteins containing the entry.
    Sampling is reservoir-style over up to `max_pages` pages for speed.
    """
    if seed is not None:
        random.seed(seed)
    db, acc = _infer_db_and_acc(identifier, db)

    reservoir: List[str] = []
    seen = 0
    for meta in iter_reviewed_proteins_by_entry(db, acc, page_size=page_size, max_pages=max_pages):
        up = meta.get("accession") or meta.get("acc")  # defensive
        if not up:
            continue
        seen += 1
        if len(reservoir) < n:
            reservoir.append(up)
        else:
            j = random.randint(0, seen - 1)
            if j < n:
                reservoir[j] = up

    return reservoir
import random, requests
from urllib.parse import quote

import requests, time, random

def interpro_entry_meta(ipr_id: str):
    """
    Return {'name', 'short_name', 'type', 'accession'} for an InterPro ID.
    Uses JSON (not TSV) and is robust to minor schema differences.
    """
    ipr_id = ipr_id.strip().upper()
    url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{ipr_id}/"
    data = _get_json(url)

    # Some responses are the object itself; others wrap in 'metadata'
    meta = data.get("metadata", data) if isinstance(data, dict) else {}

    # 'name' can be a string or an object with {'name', 'short'}
    name = meta.get("name")
    short_name = meta.get("short_name")
    if isinstance(name, dict):
        short_name = short_name or name.get("short") or name.get("short_name")
        name = name.get("name") or name.get("value") or name.get("full")

    # Defensive clean-up in case the server echoes the accession as “name”
    if name and name.strip('"\'' ).upper() == ipr_id:
        name = None  # fall back to None if it’s not a real name

    return {
        "accession": meta.get("accession") or ipr_id,
        "name": name,
        "short_name": short_name,
        "type": meta.get("type"),
    }

if __name__ == "__main__":
    import argparse
    import sys as _sys

    parser = argparse.ArgumentParser(description="Generate HTML reports for latents across InterPro domains.")
    subparsers = parser.add_subparsers(dest="cmd")

    # Single report mode
    p_single = subparsers.add_parser("single", help="Generate one report for a given (layer, latent) and domain IDs")
    p_single.add_argument("--layer", type=int, required=True, help="ESM layer index (e.g., 8)")
    p_single.add_argument("--latent", type=int, required=True, help="Latent index within SAE (e.g., 2677)")
    p_single.add_argument(
        "--interpro", "--domains", dest="interpro_ids", nargs="+", required=True,
        help="One or more InterPro/Pfam/etc identifiers (e.g., IPR001557 PF00005)"
    )
    p_single.add_argument("--samples-per-domain", type=int, default=2, help="Proteins to sample per domain (default: 2)")
    p_single.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    p_single.add_argument("--out", type=str, default=None, help="Output HTML file path; prints to stdout if not set")
    p_single.add_argument("--title", type=str, default=None, help="Optional page title")
    p_single.add_argument("--max-domains", type=int, default=10, help="Max number of domains to include (default: 10)")

    # Batch mode
    p_batch = subparsers.add_parser("batch", help="Generate multiple reports from a CSV of domains per (layer, latent)")
    p_batch.add_argument("--csv", type=str, required=True, help="Path to domain assoc CSV")
    p_batch.add_argument("--layer-col", type=str, default="Layer", help="CSV column for layer")
    p_batch.add_argument("--latent-col", type=str, default="Latent_ID", help="CSV column for latent ID")
    p_batch.add_argument("--domain-col", type=str, default="InterPro ID", help="CSV column for InterPro IDs")
    p_batch.add_argument("--latents", type=str, nargs="*", default=None, help="Optional list like LAYER:LATENT pairs (e.g., 8:2677 12:3035)")
    p_batch.add_argument("--samples-per-domain", type=int, default=2)
    p_batch.add_argument("--seed", type=int, default=42)
    p_batch.add_argument("--out-dir", type=str, required=True, help="Directory to write HTML files")
    p_batch.add_argument("--title-prefix", type=str, default="Latent report", help="Title prefix for each page")
    p_batch.add_argument("--max-domains", type=int, default=10, help="Max number of domains to include per latent (default: 10)")

    args = parser.parse_args()

    if args.cmd in (None, "single"):
        # Backward-compatible: allow running without explicit subcommand
        if args.cmd is None:
            # Shim: convert top-level args into single-mode by re-parsing
            # If no subcommand provided, argparse stored them on 'args'
            # Expect attributes to exist; if missing, raise helpful error
            missing = [x for x in ("layer", "latent", "interpro_ids") if not hasattr(args, x)]
            if missing:
                parser.error("single mode requires --layer, --latent and --interpro IDs")

        interpro_ids: List[str] = []
        for tok in args.interpro_ids:
            interpro_ids.extend([t for t in tok.split(",") if t])

        html = generate_latent_html_report(
            layer=args.layer,
            latent_idx=args.latent,
            interpro_ids=interpro_ids,
            samples_per_domain=args.samples_per_domain,
            seed=args.seed,
            out_path=args.out,
            page_title=args.title,
            max_domains=args.max_domains,
        )
        if args.out is None:
            try:
                _sys.stdout.write(html)
            except Exception:
                pass
    elif args.cmd == "batch":
        import pandas as _pd
        import os as _os

        df = _pd.read_csv(args.csv)
        # Determine (layer, latent) pairs
        if args.latents:
            targets: List[Tuple[int, int]] = []
            for tok in args.latents:
                try:
                    lay_s, lat_s = tok.split(":", 1)
                    targets.append((int(lay_s), int(lat_s)))
                except Exception:
                    continue
        else:
            grp = df.groupby([args.layer_col, args.latent_col]).size().reset_index().iloc[:, :2]
            targets = [(int(r[args.layer_col]), int(r[args.latent_col])) for _, r in grp.iterrows()]

        _os.makedirs(args.out_dir, exist_ok=True)

        for (lay, lat) in targets:
            mask = (df[args.layer_col] == lay) & (df[args.latent_col] == lat)
            dom_series = df.loc[mask, args.domain_col].dropna().astype(str)
            interpro_ids: List[str] = []
            for cell in dom_series:
                for tok in str(cell).replace(";", ",").split(","):
                    tok = tok.strip()
                    if tok:
                        interpro_ids.append(tok)
            if not interpro_ids:
                continue

            title = f"{args.title_prefix}: L{lay}-{lat}"
            out_path = _os.path.join(args.out_dir, f"latent_l{lay}_{lat}.html")
            generate_latent_html_report(
                layer=lay,
                latent_idx=lat,
                interpro_ids=interpro_ids,
                samples_per_domain=args.samples_per_domain,
                seed=args.seed,
                out_path=out_path,
                page_title=title,
                max_domains=args.max_domains,
            )