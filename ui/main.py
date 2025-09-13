"""
Streamlit app: one-pane protein inspector for PDB/UniProt + domains + activations + on-the-fly SAE compute

What it does (MVP++):
- Input: PDB ID (+chain) or UniProt accession
- Maps PDB→UniProt via PDBe SIFTS, fetches UniProt metadata/FASTA and InterPro domain spans (no scraping)
- **On-the-fly activations**: choose a layer and latent index and compute per-residue activations via your ESM+SAE hooks (predefined)
- Optional **multi-variant** panel: compute Full / Clean / Corrupted sequences with user-specified flanks; shows contact recovery metrics (predefined)
- Visualizes 3D structure with py3Dmol (uses your predefined utilities) and draws a sequence/domain track

Run:
    pip install streamlit requests biopython py3Dmol matplotlib numpy pandas torch
    streamlit run streamlit_app.py

Notes:
- Heavy compute uses your own functions (marked "predefined"). Edit the import block below to point to your modules if needed.
- For UniProt input, we use AlphaFold v4 model and expect activations to match UniProt length.
- For PDB input, activations are computed on the mapped UniProt sequence and projected onto the displayed chain via SIFTS.
"""

from __future__ import annotations
import io
import json
import textwrap
from typing import Dict, List, Tuple, Optional
import os
print(os.getcwd())
import sys
sys.path.append('../')
sys.path.append('../plm_circuits/')
import numpy as np
import pandas as pd
import requests
import streamlit as st
# try:
#     from Bio.PDB.Polypeptide import three_to_one as _bio_three_to_one
# except Exception:
#     _bio_three_to_one = None

_RES3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "MSE":"M","SEC":"U","PYL":"O","HSD":"H","HSE":"H","HSP":"H",
    "ASX":"X","GLX":"X","XLE":"X","UNK":"X"
}
def three_to_one(resname: str) -> str:
    res = (resname or "").strip().upper()
    return _RES3_TO_1.get(res, "X")
import torch

# === Your existing utility module ===
# Place protein_viz_utils.py next to this file and import as viz
import helpers.protein_viz_utils as viz  # "predefined"

# === Predefined loaders/hooks/utilities from your environment ===
# Edit these imports to match your project layout if needed.
# try:
from helpers.utils import (  # <-- change to your module; or comment out if already on PYTHONPATH
    load_esm,            # predefined
    load_sae_prot,       # predefined
    patching_metric,     # predefined
    mask_flanks_segment  # predefined
)
from hook_manager import SAEHookProt

# except Exception:
#     # Fallback: assume they exist in globals when you run `streamlit run` from your repo env
#     pass

# -----------------------
# API helpers (cached)
# -----------------------

@st.cache_data(show_spinner=False)
def fetch_pdbe_sifts_mappings(pdb_id: str) -> dict:
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/{pdb_id.lower()}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get(pdb_id.lower(), {})

@st.cache_data(show_spinner=False)
def fetch_uniprot_entry(acc: str) -> dict:
    url = f"https://rest.uniprot.org/uniprotkb/{acc}"
    r = requests.get(url, headers={"Accept": "application/json"}, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_uniprot_fasta(acc: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text

@st.cache_data(show_spinner=False)
def fetch_interpro_protein(acc: str, page_size: int = 500) -> dict:
    # All database entries (Pfam + others) mapped to this UniProt accession
    # url = f"https://www.ebi.ac.uk/interpro/api/protein/uniprot/{acc}"
    url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/reviewed/{acc}"
    r = requests.get(url, timeout=30) #r = requests.get(url, params={"page_size": page_size}, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def load_full_seq_dict() -> dict:
    """Load optional PDB+chain specific sequences, keyed like '2b61a'."""
    try:
        here = os.path.dirname(__file__)
        path = os.path.normpath(os.path.join(here, "../data/full_seq_dict.json"))
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

# -----------------------
# Model loading (cached resources)
# -----------------------

MAIN_LAYERS_DEFAULT = [4, 8, 12, 16, 20, 24, 28]

@st.cache_resource(show_spinner=False)
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=True)
def get_esm():
    # Uses your predefined loader
    try:
        esm_transformer, batch_converter, esm2_alphabet = load_esm(33, device=get_device())  # predefined
    except NameError as e:
        raise RuntimeError("load_esm(...) not found. Edit the import block near the top to point to your module.") from e
    return esm_transformer, batch_converter, esm2_alphabet

@st.cache_resource(show_spinner=True)
def get_sae(layer: int):
    try:
        sae_model = load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=layer, device=get_device())  # predefined
    except NameError as e:
        raise RuntimeError("load_sae_prot(...) not found. Edit the import block near the top to point to your module.") from e
    return sae_model

# -----------------------
# Mapping helpers
# -----------------------

def chain_residue_index_map(pdb_id: str, chain_id: str) -> Tuple[List[int], List[str]]:
    """Return two lists for the selected (PDB, chain):
    - residue_numbers: author residue numbers (res.id[1]) per position 0..N-1
    - residue_one_letter: one-letter AA ("X" if unknown)
    """
    chain = viz.get_single_chain_pdb_structure(pdb_id, chain_id)  # predefined
    res_nums, res_aa = [], []
    for res in chain.get_residues():
        res_nums.append(res.id[1])
        try:
            aa = three_to_one(res.get_resname())
        except KeyError:
            aa = "X"
        res_aa.append(aa)
    return res_nums, res_aa


def sifts_build_uniprot_pos_map(
    sifts: dict, pdb_id: str, chain_id: str, chain_res_nums: List[int]
) -> Tuple[Optional[str], Dict[int, int]]:
    """From PDBe SIFTS mappings, build a dict: chain_index -> UniProt position (1-based).
    Returns (uniprot_acc, mapping_dict).
    """
    unp_section = sifts.get("UniProt", {})
    resid_to_index = {resn: i for i, resn in enumerate(chain_res_nums)}
    chain_to_unp: Dict[int, int] = {}
    chosen_acc = None

    for acc, payload in unp_section.items():
        for m in payload.get("mappings", []):
            if m.get("chain_id") != chain_id:
                continue
            pdb_start = m.get("start", {}).get("residue_number")
            pdb_end = m.get("end", {}).get("residue_number")
            unp_start = m.get("unp_start")
            unp_end = m.get("unp_end")
            if None in (pdb_start, pdb_end, unp_start, unp_end):
                continue
            seg_len = min(pdb_end - pdb_start, unp_end - unp_start)
            for k in range(seg_len + 1):
                pdb_resno = pdb_start + k
                unp_pos = unp_start + k
                if pdb_resno in resid_to_index:
                    chain_to_unp[resid_to_index[pdb_resno]] = unp_pos
            chosen_acc = acc

    return chosen_acc, chain_to_unp


def map_uniprot_activations_to_chain(
    act_unp: np.ndarray, chain_len: int, chain_to_unp: Dict[int, int], fill_value: float = 0.0
) -> np.ndarray:
    out = np.full(chain_len, fill_value, dtype=float)
    L_unp = int(act_unp.shape[0])
    for chain_idx, unp_pos in chain_to_unp.items():
        if 1 <= unp_pos <= L_unp:
            out[chain_idx] = float(act_unp[unp_pos - 1])
    return out

# -----------------------
# InterPro parsing & plotting helpers
# -----------------------

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


def draw_sequence_tracks(
    L: int,
    domains_df: pd.DataFrame,
    activations: Optional[np.ndarray] = None,
    coverage_segments: Optional[List[Tuple[int, int]]] = None,
    title: str = "Sequence tracks",
    sequence_letters: Optional[str] = None,
    show_bars_when_activations: bool = True,
):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    # Greedy lane assignment to avoid overlapping labels
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

    # Layout scaling
    lane_dy = 0.25  # vertical offset per lane in axes data units
    base_h = 2.5 if activations is None else 4.5
    extra_h = max(0, num_lanes - 1) * 0.60
    fig, ax = plt.subplots(figsize=(12, base_h + extra_h))

    ax.plot([1, L], [0.5, 0.5], lw=6, alpha=0.1)
    for idx, r in domains_df.iterrows():
        if pd.isna(r.start) or pd.isna(r.end):
            continue
        s, e = int(r.start), int(r.end)
        ax.add_patch(
            plt.Rectangle((s, 0.2), max(1, e - s + 1), 0.6, alpha=0.35)
        )
        # Lane-based label position to avoid collisions
        lane = row_to_lane.get(idx, 0)
        label_y = 0.60 + lane * lane_dy
        txt = str(r.get("name") or r.get("entry") or "")
        # Truncate overly long labels roughly to fit the rectangle span
        try:
            span = max(1, e - s + 1)
            max_chars = int(max(8, min(40, span * 0.4)))
        except Exception:
            max_chars = 24
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
            fontsize=8,
            clip_on=False,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.6)
        )
    if coverage_segments:
        for s, e in coverage_segments:
            ax.add_patch(plt.Rectangle((s, 0.05), max(1, e - s + 1), 0.1))
    ax.set_xlim(1, L)
    extra_top = max(0, num_lanes - 1) * lane_dy
    ax.set_ylim(0, 1.0 + extra_top + (0.8 if activations is not None else 0))
    ax.set_yticks([])
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
                    ax2.text(x, (0.0 if np.isnan(v) else v) + offset, aa, ha="center", va=va, fontsize=7)
            ax2.set_ylabel("Activation")
            ax2.grid(True, axis="y", alpha=0.2)
        else:
            ax2 = ax.twinx()
            ax2.plot(xs, activations, lw=1)
            ax2.set_ylabel("Activation")
            ax2.grid(True, alpha=0.2)
    ax.set_title(title)
    st.pyplot(fig)

# -----------------------
# Sequence heatmap helper
# -----------------------
def render_sequence_heatmap(
    seq: str,
    values: np.ndarray,
    colormap_fn=viz.rwb_colormap_fn,
    wrap: int = 120,  # ignored (kept for API compatibility)
    vmin: float | None = None,
    vmax: float | None = None,
    height: int | None = None,
    title: str = "Sequence heatmap",
):
    import html, uuid
    L = len(seq) if seq else 0
    if values is None or L == 0:
        st.info("No sequence/activations to render.")
        return
    vmin = float(np.min(values)) if vmin is None else float(vmin)
    vmax = float(np.max(values)) if vmax is None else float(vmax)

    # Unique id so multiple boxes don’t clash across reruns
    uid = f"seqbox-{uuid.uuid4().hex[:8]}"

    spans = []
    for i, (aa, val) in enumerate(zip(seq, values)):
        color = colormap_fn(float(val), vmin, vmax)
        aa_esc = html.escape(aa)
        # data-* attrs let JS show an index+value tooltip
        spans.append(
            f'<span class="res" data-i="{i+1}" data-val="{float(val):.6f}" '
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
    st.components.v1.html(html_block, height=h)

# -----------------------
# On-the-fly compute helpers
# -----------------------

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
            calc_error=True,
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
    else:
        if use_masked_latents and hasattr(sae_model, "cache_masked_latents") and sae_model.cache_masked_latents is not None:
            arr = sae_model.cache_masked_latents  # [L, F] or [B, L, F]
        else:
            arr = sae_model.feature_acts  # [L, F] or [B, L, F]
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[0]
    L, F = arr.shape
    if not (0 <= latent_idx < F):
        raise ValueError(f"latent_idx {latent_idx} out of range [0, {F-1}]")
    return arr[:, latent_idx]

# -----------------------
# Sequence token helpers for plotting
# -----------------------

def tokens_to_residue_letters(seq_text: str, target_len: Optional[int] = None) -> str:
    """Collapse special tokens like "<mask>" to a single 'X' so we can label bars.
    Ensures returned string is exactly target_len if provided (trim or pad with 'X').
    """
    if not seq_text:
        return "" if not target_len else "X" * target_len
    out = []
    i = 0
    n = len(seq_text)
    while i < n:
        if seq_text[i] == '<' and seq_text.startswith('<mask>', i):
            out.append('X')
            i += len('<mask>')
        else:
            ch = seq_text[i]
            if ch.isalpha():
                out.append(ch.upper())
            # ignore other characters
            i += 1
        if target_len is not None and len(out) == target_len:
            # stop early if we've reached the required length
            break
    if target_len is not None:
        if len(out) < target_len:
            out.extend(['X'] * (target_len - len(out)))
        elif len(out) > target_len:
            out = out[:target_len]
    return ''.join(out)

# -----------------------
# Canonical sequence + activation alignment helpers
# -----------------------

def get_pdb_chain_sequence(pdb_id: Optional[str], chain_id: Optional[str]) -> Optional[str]:
    """Return the PDB chain sequence (one-letter AA) using the structure fetch.
    Falls back to None if unavailable.
    """
    try:
        if not pdb_id or not chain_id:
            return None
        _, chain_aa = chain_residue_index_map(str(pdb_id), str(chain_id))
        if not chain_aa:
            return None
        return "".join(chain_aa)
    except Exception:
        return None


def resolve_canonical_sequence(
    source_mode: str,
    pdb_id: Optional[str],
    chain_id: Optional[str],
    uniprot_seq: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Standardize sequence selection across the app.

    Priority:
    - If mode==PDB: use local dict sequence for key f"{pdb_id}{chain_id}" if present → kind="pdb_local"
      else use PDB chain sequence via structure → kind="pdb_chain"
    - If mode==UniProt: use UniProt FASTA string → kind="uniprot"
    Returns (sequence or None, kind or None).
    """
    if source_mode == "PDB" and pdb_id and chain_id:
        seq_dict = load_full_seq_dict()
        key_uc = f"{str(pdb_id).upper()}{str(chain_id).upper()}"
        local_seq = seq_dict.get(key_uc)
        if isinstance(local_seq, str) and len(local_seq) > 0:
            return local_seq, "pdb_local"
        pdb_seq = get_pdb_chain_sequence(pdb_id, chain_id)
        if pdb_seq:
            return pdb_seq, "pdb_chain"
        # Final fallback if nothing else: use UniProt if provided
        if uniprot_seq:
            return uniprot_seq, "uniprot"
        return None, None
    if source_mode == "UniProt" and uniprot_seq:
        return uniprot_seq, "uniprot"
    return None, None


def align_activations_to_sequence_length(values: Optional[np.ndarray], sequence_length: Optional[int]) -> Optional[np.ndarray]:
    """Trim potential BOS/EOS and ensure activations length matches sequence_length when provided."""
    if values is None:
        return None
    if sequence_length is None:
        return values
    n = int(len(values))
    L = int(sequence_length)
    if n == L + 2:
        return values[1:-1]
    if n == L + 1:
        return values[1:]
    if n > L:
        return values[:L]
    return values


def find_subsequence_offset(container: Optional[str], sub: Optional[str]) -> Optional[int]:
    """Return 0-based offset where sub appears in container, or None if not found."""
    if not container or not sub:
        return None
    idx = container.find(sub)
    return idx if idx >= 0 else None


def _clip_to_range(start: int, end: int, L: int) -> Optional[Tuple[int, int]]:
    s = max(1, int(start))
    e = min(L, int(end))
    if s > e:
        return None
    return (s, e)


def shift_interpro_domains_to_canonical(domains_df: pd.DataFrame, offset: int, L_canonical: int) -> pd.DataFrame:
    """Shift InterPro domains (1-based UniProt coords) by offset into canonical coords and clip to [1, L_canonical]."""
    if domains_df is None or len(domains_df) == 0:
        return pd.DataFrame(columns=["entry", "name", "db", "start", "end"])
    rows = []
    for _, r in domains_df.iterrows():
        s = (int(r.start) if pd.notna(r.start) else None)
        e = (int(r.end) if pd.notna(r.end) else None)
        if s is None or e is None:
            continue
        s2, e2 = s + offset, e + offset
        clipped = _clip_to_range(s2, e2, L_canonical)
        if clipped is None:
            continue
        cs, ce = clipped
        rows.append({
            "entry": r.get("entry"),
            "name": r.get("name"),
            "db": r.get("db"),
            "start": cs,
            "end": ce,
        })
    if not rows:
        return pd.DataFrame(columns=["entry", "name", "db", "start", "end"])
    return pd.DataFrame(rows).sort_values(["start", "end"]).reset_index(drop=True)


def shift_coverage_segments_to_canonical(segments: Optional[List[Tuple[int, int]]], offset: int, L_canonical: int) -> List[Tuple[int, int]]:
    if not segments:
        return []
    out: List[Tuple[int, int]] = []
    for s, e in segments:
        s2, e2 = int(s) + offset, int(e) + offset
        clipped = _clip_to_range(s2, e2, L_canonical)
        if clipped is not None:
            out.append(clipped)
    return out

# -----------------------
# UI
# -----------------------

st.set_page_config(page_title="Protein One‑Pane", layout="wide")
st.title("🧬 Protein One‑Pane (MVP++)")
st.caption("PDB/UniProt → domains + metadata + 3D + your SAE activations — in one place")

# st.sidebar.markdown("**Runtime**")
# st.sidebar.write(f"Device: `{get_device()}`")

colA, colB = st.columns([1, 1])
with colA:
    source_mode = st.radio("Input mode", ["PDB", "UniProt"], horizontal=True)
    pdb_id = chain_id = uniprot_acc = None
    sifts = {}

    if source_mode == "PDB":
        pdb_id = st.text_input("PDB ID", value="2B61").strip()
        if pdb_id:
            try:
                sifts = fetch_pdbe_sifts_mappings(pdb_id)
            except Exception as e:
                st.error(f"Failed to fetch SIFTS mappings: {e}")
            chains = sorted({m.get("chain_id") for v in sifts.get("UniProt", {}).values() for m in v.get("mappings", []) if m.get("chain_id")})
            if not chains:
                chains = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            chain_id = st.selectbox("Chain", chains, index=0)
            try:
                chain_res_nums, chain_aa = chain_residue_index_map(pdb_id, chain_id)
                uniprot_acc, chain_to_unp = sifts_build_uniprot_pos_map(sifts, pdb_id, chain_id, chain_res_nums)
            except Exception as e:
                chain_res_nums, chain_aa, chain_to_unp, uniprot_acc = [], [], {}, None
                st.error(f"Mapping error: {e}")
            if uniprot_acc:
                st.success(f"Mapped to UniProt: {uniprot_acc}")
            else:
                st.warning("Could not resolve UniProt accession from SIFTS for this chain.")
    else:
        uniprot_acc = st.text_input("UniProt accession", value="P69905").strip()

with colB:
    st.markdown("**Activation source**")
    act_source = st.radio("Source", ["Compute via your pipeline", "Upload (optional)", "None"], index=0)

    latent_idx = 0
    selected_layer = st.selectbox("Layer", options=MAIN_LAYERS_DEFAULT, index=1, help="Your SAE-trained layers")
    latent_idx = st.number_input("Latent index (integer)", value=2677, min_value=0, step=1, help="Index into SAE dim (e.g., 0..4095)")
    use_masked_latents = st.checkbox("Use masked latents after top‑k", value=False)
    use_error = st.checkbox("Use reconstruction error instead of features", value=False)

    # Optional plotting normalization for tracks/3D colorbar feel
    norm_mode = st.selectbox("Normalization", ["none", "zscore", "minmax [0,1]"])
    clip_val = st.number_input("Clip |value| ≤", value=0.0, min_value=0.0, step=0.1)

# Fetch metadata and domains if we have a UniProt accession
uniprot_meta = None
interpro_df = pd.DataFrame()
uniprot_seq = None
if uniprot_acc:
    try:
        uniprot_meta = fetch_uniprot_entry(uniprot_acc)
        fasta = fetch_uniprot_fasta(uniprot_acc)
        lines = [ln for ln in fasta.splitlines() if ln and not ln.startswith(">")]
        uniprot_seq = "".join(lines).strip()
        interpro_json = fetch_interpro_protein(uniprot_acc)
        interpro_df = parse_interpro_domains(interpro_json)
    except Exception as e:
        st.error(f"Failed to fetch UniProt/InterPro data: {e}")

# Canonical sequence for this session
canonical_seq: Optional[str] = None
canonical_kind: Optional[str] = None  # "pdb_local" | "pdb_chain" | "uniprot" | None
canonical_seq, canonical_kind = resolve_canonical_sequence(source_mode, pdb_id, chain_id, uniprot_seq)

# -----------------------
# Left panel: metadata & domains
# -----------------------

meta_col, viz_col = st.columns([0.9, 1.1])
with meta_col:
    st.subheader("Metadata & Domains")
    if uniprot_meta:
        primary_name = uniprot_meta.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value")
        org = uniprot_meta.get("organism", {}).get("scientificName")
        length = uniprot_meta.get("sequence", {}).get("length")
        st.markdown(
            f"**{primary_name or '—'}**"
            f"UniProt: `{uniprot_acc or '—'}`  |  Organism: `{org or '—'}`  |  Length: `{length or (len(uniprot_seq) if uniprot_seq else '—')}`"
        )

    if not interpro_df.empty:
        st.dataframe(interpro_df, use_container_width=True, hide_index=True)
    else:
        st.info("No InterPro domain spans found (or not loaded yet).")

    # Sequence tracks (will be filled after compute below)
    # Placeholder draw later with actual activations

# -----------------------
# Right panel: 3D view + compute controls
# -----------------------
computed_act = None

with viz_col:
    st.subheader("3D Structure + color‑mapped activations")

    compute_clicked = st.button("Compute activations (selected layer & latent)")

    # Optionally accept upload (if user ever wants it later)
    upload_arr = None
    if act_source == "Upload (optional)":
        up = st.file_uploader(".npy (1D) or .csv (1 col)", type=["npy", "csv"])
        if up is not None:
            try:
                if up.name.endswith(".npy"):
                    upload_arr = np.load(io.BytesIO(up.read()))
                else:
                    upload_arr = np.loadtxt(io.BytesIO(up.read()), delimiter=",")
                upload_arr = np.squeeze(upload_arr).astype(float)
                st.success(f"Loaded activation vector of length {len(upload_arr)}")
            except Exception as e:
                st.error(f"Failed to load array: {e}")

    # Decide activations source for downstream visualization: canonical sequence
    if compute_clicked and canonical_seq:
        with st.spinner("Running ESM+SAE (predefined) ..."):
            try:
                computed_act = compute_latent_activations_on_sequence(
                    sequence=canonical_seq,
                    layer=selected_layer,
                    latent_idx=int(latent_idx),
                    use_masked_latents=True,
                    use_error=use_error,
                )
                print(f"computed_act: {computed_act.shape}")
                print(f"length of canonical_seq: {len(canonical_seq)}")
            except Exception as e:
                st.error(f"Compute error: {e}")

    # Normalize/clip
    def normalize_for_plot(arr: Optional[np.ndarray]):
        if arr is None:
            return None
        arr = arr.astype(float)
        if clip_val > 0:
            arr = np.clip(arr, -clip_val, clip_val)
        if norm_mode == "zscore":
            mu = arr.mean() if arr.size else 0.0
            sd = arr.std() if arr.size else 1.0
            arr = (arr - mu) / (sd if sd else 1.0)
        elif norm_mode == "minmax [0,1]":
            mn, mx = float(arr.min()), float(arr.max())
            arr = (arr - mn) / (mx - mn + 1e-12)
        return arr

    act_array = None
    if computed_act is not None:
        act_array = normalize_for_plot(computed_act)
    elif upload_arr is not None:
        act_array = normalize_for_plot(upload_arr)

    # Align activation length to canonical sequence (handles BOS/EOS)
    if act_array is not None and canonical_seq is not None:
        act_array = align_activations_to_sequence_length(act_array, len(canonical_seq))

    # 3D + sequence track visualization
    tracks_drawn = False
    try:
        if source_mode == "UniProt":
            if uniprot_seq is None:
                st.warning("Provide a valid UniProt accession to fetch sequence/structure.")
            else:
                L = len(uniprot_seq)
                print(f"L: {L}")
                vals = act_array if act_array is not None else np.zeros(L, dtype=float)
                render_sequence_heatmap(
                    uniprot_seq, vals,
                    colormap_fn=viz.mono_colormap_fn, # mono_colormap_fn,
                    wrap=120,
                    title="Residue activations (UniProt coords)"
                )
                # Always draw sequence tracks for UniProt, regardless of 3D success
                seq_letters_for_tracks = tokens_to_residue_letters(uniprot_seq, target_len=L)
                act_for_tracks_local = act_array if (act_array is not None) else None
                draw_sequence_tracks(
                    L,
                    interpro_df if interpro_df is not None else pd.DataFrame(columns=["entry","name","db","start","end"]),
                    activations=act_for_tracks_local,
                    coverage_segments=[],
                    title="UniProt domains (± coverage/activations)",
                    sequence_letters=seq_letters_for_tracks,
                    show_bars_when_activations=True,
                )
                tracks_drawn = True
                if act_array is not None and len(act_array) != L:
                    st.error("Activation length must equal UniProt sequence length for AlphaFold view.")
                else:
                    v = viz.view_single_protein(
                        uniprot_id=uniprot_acc,
                        chain_id="A",
                        values_to_color=vals.tolist(),
                        colormap_fn=viz.mono_colormap_fn, #rwb_colormap_fn,
                        default_color="white",
                        pymol_params={"width": 700, "height": 520},
                    )
                    html = v._make_html()
                    st.components.v1.html(html, height=540)

        else:  # PDB + chain
            if pdb_id and chain_id:
                chain_res_nums, chain_aa = chain_residue_index_map(pdb_id, chain_id)
                chain_len = len(chain_res_nums)
                if chain_len == 0:
                    st.error("No residues found on this chain.")
                else:
                    vals_chain = np.zeros(chain_len, dtype=float)
                    if act_array is not None:
                        if canonical_kind in ("pdb_local", "pdb_chain") and len(act_array) == chain_len:
                            vals_chain = np.asarray(act_array, dtype=float)
                        elif canonical_kind == "uniprot" and uniprot_acc is not None:
                            _, chain_to_unp = sifts_build_uniprot_pos_map(fetch_pdbe_sifts_mappings(pdb_id), pdb_id, chain_id, chain_res_nums)
                            vals_chain = map_uniprot_activations_to_chain(act_array, chain_len, chain_to_unp, fill_value=0.0)

                    act_seq_letters = tokens_to_residue_letters(canonical_seq or "", target_len=(len(act_array) if act_array is not None else (len(canonical_seq) if canonical_seq else 0)))
                    render_sequence_heatmap(
                        act_seq_letters, (act_array if act_array is not None else np.zeros(len(act_seq_letters), dtype=float)),
                        colormap_fn=viz.mono_colormap_fn, #rwb_colormap_fn,
                        wrap=120,
                        title=f"Residue activations (Chain {chain_id})"
                    )
                    v = viz.view_single_protein(
                        pdb_id=pdb_id,
                        chain_id=chain_id,
                        values_to_color=vals_chain.tolist(),
                        colormap_fn=viz.mono_colormap_fn, #rwb_colormap_fn,
                        default_color="white",
                        pymol_params={"width": 700, "height": 520},
                    )
                    html = v._make_html()
                    st.components.v1.html(html, height=540)
    except Exception as e:
        st.error(f"Visualization error: {e}")

# -----------------------
# Sequence tracks under the 3D panel
# -----------------------
# Tracks under 3D: always show domain/coverage context, shifted to canonical coordinates if needed
if canonical_seq and not tracks_drawn:
    L_can = len(canonical_seq)
    # Build UniProt coverage if available
    coverage_unp = []
    if uniprot_seq and 'sifts' in locals() and uniprot_acc and source_mode == "PDB":
        for acc, payload in sifts.get("UniProt", {}).items():
            if acc != uniprot_acc:
                continue
            for m in payload.get("mappings", []):
                if m.get("chain_id") == chain_id and m.get("unp_start") and m.get("unp_end"):
                    coverage_unp.append((int(m["unp_start"]), int(m["unp_end"])) )

    # Compute offset of UniProt inside canonical (handles local prefixes)
    shifted_domains = interpro_df
    shifted_coverage = coverage_unp
    if uniprot_seq and canonical_seq and canonical_kind in ("pdb_local", "pdb_chain"):
        offset = find_subsequence_offset(canonical_seq, uniprot_seq)
        if offset is not None:
            # UniProt is a substring within canonical: shift +1 for 1-based domain coords
            shifted_domains = shift_interpro_domains_to_canonical(interpro_df, offset, L_can)
            shifted_coverage = shift_coverage_segments_to_canonical(coverage_unp, offset, L_can)
        else:
            # If not found, skip shifting; show nothing to avoid misleading ranges
            shifted_domains = pd.DataFrame(columns=["entry", "name", "db", "start", "end"])
            shifted_coverage = []

    seq_letters_for_tracks = tokens_to_residue_letters(canonical_seq, target_len=L_can)
    act_for_tracks = act_array if (act_array is not None and len(act_array) == L_can) else None
    draw_sequence_tracks(
        L_can,
        shifted_domains if shifted_domains is not None else pd.DataFrame(columns=["entry", "name", "db", "start", "end"]),
        activations=act_for_tracks,
        coverage_segments=shifted_coverage,
        title=("UniProt domains (shifted to canonical)" if canonical_kind != "uniprot" else "UniProt domains (± coverage/activations)"),
        sequence_letters=seq_letters_for_tracks,
        show_bars_when_activations=True,
    )

# -----------------------
# Multi-variant panel (Full / Clean / Corrupted)
# -----------------------
with st.expander("Full vs Clean vs Corrupted (optional)"):
    st.markdown("Provide rough hyperparameters; we’ll compute sequences, contacts, activations.")
    c1, c2 = st.columns(2)
    with c1: #[182, 316]
        ss1_mid = st.number_input("ss1_start (1-based)", min_value=1, value=182, step=1)
        ss2_mid = st.number_input("ss1_end (1-based)",   min_value=1, value=316, step=1)
    # with c2:
    #     ss2_start = st.number_input("ss2_start (1-based)", min_value=1, value=20, step=1)
    #     ss2_end   = st.number_input("ss2_end (1-based)",   min_value=1, value=30, step=1)
    with c2:
        clean_fl  = st.number_input("clean flank length",  min_value=0, value=44, step=1)
        corr_fl   = st.number_input("corrupted flank length", min_value=0, value=43, step=1)

    do_variants = st.button("Compute variants (contacts + activations)")

    if do_variants and canonical_seq:
        try:
            esm_transformer, batch_converter, esm2_alphabet = get_esm()
            device = get_device()
            ss1_start = ss1_mid - 5
            ss1_end = ss1_mid + 5 + 1
            ss2_start = ss2_mid - 5
            ss2_end = ss2_mid + 5 + 1
            # Use canonical sequence for variant generation
            seq = canonical_seq
            L = len(seq)
            s1s = max(0, int(ss1_start))
            s1e = min(L, int(ss1_end))
            s2s = max(0, int(ss2_start))
            s2e = min(L, int(ss2_end))

            # Full
            _, _, full_tokens = batch_converter([(1, seq)])
            full_tokens = full_tokens.to(device)
            full_mask = (full_tokens != esm2_alphabet.padding_idx).to(device)
            with torch.no_grad():
                full_contacts = esm_transformer.predict_contacts(full_tokens, full_mask)[0]

            # Clean mask: unmask flanks around segments
            left_start = max(0, s1s - int(clean_fl))
            left_end   = s1s
            right_start = s2e
            right_end   = min(L, s2e + int(clean_fl))
            unmask_left_idxs  = list(range(left_start, left_end))
            unmask_right_idxs = list(range(right_start, right_end))
            # try:
            clean_seq = mask_flanks_segment(seq, s1s, s1e, s2s, s2e, unmask_left_idxs, unmask_right_idxs)  # predefined
            # except NameError:
            #     st.error("mask_flanks_segment not found. Edit the import block at the top to your module.")
            #     clean_seq = seq
            _, _, clean_tokens = batch_converter([(1, clean_seq)])
            clean_tokens = clean_tokens.to(device)
            clean_mask = (clean_tokens != esm2_alphabet.padding_idx).to(device)
            with torch.no_grad():
                clean_contacts = esm_transformer.predict_contacts(clean_tokens, clean_mask)[0]

            # Corrupted mask: shorter flanks
            left_start = max(0, s1s - int(corr_fl))
            left_end   = s1s
            right_start = s2e
            right_end   = min(L, s2e + int(corr_fl))
            unmask_left_idxs  = list(range(left_start, left_end))
            unmask_right_idxs = list(range(right_start, right_end))
            # try:
            corr_seq = mask_flanks_segment(seq, s1s, s1e, s2s, s2e, unmask_left_idxs, unmask_right_idxs)  # predefined
            # except NameError:
            #     corr_seq = seq
            _, _, corr_tokens = batch_converter([(1, corr_seq)])
            corr_tokens = corr_tokens.to(device)
            corr_mask = (corr_tokens != esm2_alphabet.padding_idx).to(device)
            with torch.no_grad():
                corr_contacts = esm_transformer.predict_contacts(corr_tokens, corr_mask)[0]

            # Patching metric
            try:
                _patching_metric = lambda pred: patching_metric(
                    contact_preds=pred,
                    orig_contact=full_contacts,
                    ss1_start=s1s,
                    ss1_end=s1e,
                    ss2_start=s2s,
                    ss2_end=s2e,
                )
                baseline_recovery = float(_patching_metric(clean_contacts))
                corrupted_recovery = float(_patching_metric(corr_contacts))
                st.write(f"Baseline contact recovery (clean): **{baseline_recovery:.4f}**")
                print(f"full seq: {seq}")
                print(f"clean seq: {clean_seq}")
                print(f"corr seq: {corr_seq}")
                st.write(f"Corrupted contact recovery: **{corrupted_recovery:.4f}**")
            except Exception as e:
                st.info("Patching metric not available or signature mismatch. Skipping metrics.")

            # Activations for chosen layer/latent on each variant
            with st.spinner("Computing variant activations (predefined)..."):
                act_full = compute_latent_activations_on_sequence(seq, selected_layer, int(latent_idx), True, use_error)
                act_clean = compute_latent_activations_on_sequence(clean_seq, selected_layer, int(latent_idx), True, use_error)
                act_corr = compute_latent_activations_on_sequence(corr_seq, selected_layer, int(latent_idx), True, use_error)

            # Trim potential BOS/EOS tokens to match sequence lengths
            def _trim_to_length(arr: np.ndarray, L: int) -> np.ndarray:
                if arr is None:
                    return arr
                n = len(arr)
                if n == L + 2:
                    return arr[1:-1]
                if n == L + 1:
                    return arr[1:]
                if n > L:
                    return arr[:L]
                return arr
            act_full = _trim_to_length(act_full, len(seq))
            act_clean = _trim_to_length(act_clean, len(clean_seq))
            act_corr = _trim_to_length(act_corr, len(corr_seq))

            # Normalize for plots if user asked
            act_full_p = act_full.copy()
            act_clean_p = act_clean.copy()
            act_corr_p = act_corr.copy()
            def norm_inplace(arr):
                if clip_val > 0:
                    np.clip(arr, -clip_val, clip_val, out=arr)
                if norm_mode == "zscore":
                    mu, sd = arr.mean(), arr.std()
                    arr[:] = (arr - mu) / (sd if sd else 1.0)
                elif norm_mode == "minmax [0,1]":
                    mn, mx = float(arr.min()), float(arr.max())
                    arr[:] = (arr - mn) / (mx - mn + 1e-12)
            for a in (act_full_p, act_clean_p, act_corr_p):
                norm_inplace(a)

            # Bar-plot helper with residue letters
            import matplotlib.pyplot as plt
            def _barplot_sequence_with_letters(sequence: str, values: np.ndarray, title: str):
                vals = np.asarray(values, dtype=float)
                N = len(vals)
                xs = np.arange(1, N + 1)
                # map amino acids to consistent colors per figure
                uniq = sorted(set(sequence[:N]))
                color_map = {aa: f"C{i % 10}" for i, aa in enumerate(uniq)}
                colors = [color_map.get(aa, "C0") for aa in sequence[:N]]
                fig, ax = plt.subplots(figsize=(12, 3.2))
                ax.bar(xs, vals, color=colors, width=0.8)
                # annotate each bar with residue letter
                for x, aa, v in zip(xs, sequence[:N], vals):
                    va = "bottom" if v >= 0 else "top"
                    offset = 0.01 if v >= 0 else -0.01
                    ax.text(x, v + offset, aa, ha="center", va=va, fontsize=7)
                ax.set_xlim(0.5, N + 0.5)
                ax.set_xlabel("Residue index")
                ax.set_ylabel("Activation")
                ax.set_title(title)
                ax.grid(True, axis="y", alpha=0.2)
                st.pyplot(fig)

            # Plot each variant as its own protein-style bar plot
            seq_letters_full  = tokens_to_residue_letters(seq,      target_len=len(act_full_p))
            seq_letters_clean = tokens_to_residue_letters(clean_seq, target_len=len(act_clean_p))
            seq_letters_corr  = tokens_to_residue_letters(corr_seq,  target_len=len(act_corr_p))
            _barplot_sequence_with_letters(seq_letters_full,  act_full_p,  "Full sequence activations")
            _barplot_sequence_with_letters(seq_letters_clean, act_clean_p, "Clean sequence activations")
            _barplot_sequence_with_letters(seq_letters_corr,  act_corr_p,  "Corrupted sequence activations")
        except Exception as e:
            st.error(f"Variant compute error: {e}")

# -----------------------
# Debug / JSON
# -----------------------
with st.expander("Debug JSON (UniProt/InterPro)"):
    if uniprot_meta:
        st.json(uniprot_meta)
    if not interpro_df.empty:
        st.json(json.loads(interpro_df.to_json(orient="records")))
