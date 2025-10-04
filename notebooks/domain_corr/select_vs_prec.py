# %%

"""
1. load the sign domain pairs csv
2. load metxa and top2 latent token pairs
3. find all stat sign high selective latent, domain pairs 
4. find high select + high precision x1 
5. find high select + high lift but experiment with various %iles and more

Overlap pruning notes:
- Rank domains per latent by precision, lift, and MWU AUC, then keep them greedily.
- Drop any subsequent domain whose protein mask overlaps (>0 shared proteins) with a kept domain, so both full and partial overlaps are removed.
"""

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
LAYER_LATENT_DIR = REPO_ROOT / "results" / "layer_latent_dicts"
DOMAIN_RESULTS_DIR = BASE_DIR / "results" / "domain_correlation_results"
METADATA_DIR = BASE_DIR / "metadata"


def _latent_in_dict(latent_dict, layer_val, latent_val):
    """Check whether the given (layer, latent) pair exists in the supplied dict."""
    if pd.isna(layer_val) or pd.isna(latent_val):
        return False
    try:
        layer_float = float(layer_val)
    except (TypeError, ValueError):
        return False
    layer_key = str(int(layer_float)) if layer_float.is_integer() else str(layer_val)
    try:
        latent_idx = int(latent_val)
    except (TypeError, ValueError):
        return False
    return latent_idx in latent_dict.get(layer_key, [])


def _safe_int(value):
    try:
        float_val = float(value)
    except (TypeError, ValueError):
        return None
    if not float_val.is_integer():
        return None
    return int(float_val)


def _format_axis_value(value):
    try:
        float_val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if float_val.is_integer():
        return str(int(float_val))
    return str(float_val)


def domain_overlap_report(
    all_properties,
    domain_indices: list[int],
    entry_names: list[str] | None = None,
    protein_ids: list[str] | None = None,
):
    """
    Build overlap summaries for a given list of InterPro domain indices.
    Returns:
      summary_df    : per-domain counts & prevalence
      pairwise_df   : long-form pairwise overlaps (A,B)
      multiway_info : dict with union/intersection masks & sizes
    """
    D = np.asarray(all_properties) != 0
    N, M = D.shape

    idx = np.array(domain_indices, dtype=int)
    if np.any((idx < 0) | (idx >= M)):
        bad = idx[(idx < 0) | (idx >= M)]
        raise IndexError(f"Invalid domain indices: {bad.tolist()} (M={M})")

    masks = {i: D[:, i] for i in idx}
    names = {i: (entry_names[i] if entry_names is not None else str(i)) for i in idx}

    n_in = {i: int(masks[i].sum()) for i in idx}
    prev = {i: n_in[i] / N for i in idx}
    summary_df = pd.DataFrame(
        {
            "domain_idx": idx,
            "entry_name": [names[i] for i in idx],
            "n_in": [n_in[i] for i in idx],
            "prevalence": [prev[i] for i in idx],
        }
    ).sort_values("prevalence", ascending=False, kind="mergesort").reset_index(drop=True)

    rows = []
    for i_pos, i in enumerate(idx):
        Ai = masks[i]
        ni = n_in[i]
        for j in idx[i_pos + 1 :]:
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

            table = [[inter, only_i], [only_j, neither]]
            _, fisher_p = fisher_exact(table, alternative="greater")

            rows.append(
                {
                    "A_idx": i,
                    "A_name": names[i],
                    "A_n": ni,
                    "B_idx": j,
                    "B_name": names[j],
                    "B_n": nj,
                    "intersect": inter,
                    "union": union,
                    "only_A": only_i,
                    "only_B": only_j,
                    "neither": neither,
                    "jaccard": jacc,
                    "P(A|B)": pA_given_B,
                    "P(B|A)": pB_given_A,
                    "fisher_p": fisher_p,
                }
            )
    if rows:
        pairwise_df = (
            pd.DataFrame(rows)
            .sort_values(
                ["jaccard", "intersect"], ascending=[False, False], kind="mergesort"
            )
            .reset_index(drop=True)
        )
    else:
        pairwise_df = pd.DataFrame(
            columns=[
                "A_idx",
                "A_name",
                "A_n",
                "B_idx",
                "B_name",
                "B_n",
                "intersect",
                "union",
                "only_A",
                "only_B",
                "neither",
                "jaccard",
                "P(A|B)",
                "P(B|A)",
                "fisher_p",
            ]
        )

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
        "union_mask": union_mask,
        "intersection_mask": inter_mask,
    }

    if protein_ids is not None:
        ids = np.asarray(protein_ids)
        multiway_info["intersection_protein_ids"] = ids[inter_mask].tolist()
        multiway_info["union_protein_ids"] = ids[union_mask].tolist()

    return summary_df, pairwise_df, multiway_info


def select_non_overlapping_domains(
    df: pd.DataFrame, domain_matrix: np.ndarray, entry_names: list[str]
) -> pd.DataFrame:
    """Remove overlapping domains per latent while keeping the strongest signals."""
    if df.empty:
        return df

    working = df.copy()
    lift_cols = [col for col in working.columns if col.startswith("lift_")]
    if lift_cols:
        working["_priority_lift"] = working[lift_cols].max(axis=1)
    else:
        working["_priority_lift"] = -np.inf

    working["_priority_precision"] = (
        working["precision_1x"].fillna(-np.inf)
        if "precision_1x" in working.columns
        else -np.inf
    )
    working["_priority_auc"] = (
        working["mwu_auc"].fillna(-np.inf)
        if "mwu_auc" in working.columns
        else -np.inf
    )

    domain_upper = domain_matrix.shape[1]
    filtered_groups = []
    for (layer, latent), group in working.groupby(
        ["layer_id", "latent_ind"], sort=False
    ):
        index_to_domain = {}
        candidate_indices = []
        for idx, row in group.iterrows():
            domain_idx = _safe_int(row.get("domain_idx"))
            if domain_idx is None or domain_idx < 0 or domain_idx >= domain_upper:
                index_to_domain[idx] = None
                continue
            index_to_domain[idx] = domain_idx
            candidate_indices.append(domain_idx)

        unique_candidates = sorted(set(candidate_indices))
        overlaps = {di: set() for di in unique_candidates}
        if unique_candidates:
            _, pairwise_df, _ = domain_overlap_report(
                domain_matrix, unique_candidates, entry_names=entry_names
            )
            for overlap_row in pairwise_df.itertuples():
                if overlap_row.intersect > 0:
                    a_idx = int(overlap_row.A_idx)
                    b_idx = int(overlap_row.B_idx)
                    overlaps.setdefault(a_idx, set()).add(b_idx)
                    overlaps.setdefault(b_idx, set()).add(a_idx)

        group_sorted = group.sort_values(
            by=["_priority_precision", "_priority_lift", "_priority_auc"],
            ascending=[False, False, False],
            kind="mergesort",
        )

        kept_indices = []
        kept_domains = set()
        for idx in group_sorted.index:
            domain_idx = index_to_domain.get(idx)
            if domain_idx is None or domain_idx not in overlaps:
                kept_indices.append(idx)
                if domain_idx is not None:
                    kept_domains.add(domain_idx)
                continue
            if any(overlap in kept_domains for overlap in overlaps[domain_idx]):
                continue
            kept_indices.append(idx)
            kept_domains.add(domain_idx)
        filtered_groups.append(group.loc[kept_indices])

    filtered_df = pd.concat(filtered_groups, ignore_index=False)
    filtered_df = filtered_df.drop(
        columns=["_priority_precision", "_priority_lift", "_priority_auc"],
        errors="ignore",
    )
    return filtered_df.reset_index(drop=True)


def build_latent_domain_payload(
    df: pd.DataFrame, entry_accessions: list[str]
) -> dict[str, dict[str, list[dict[str, object]]]]:
    """Serialize domains per latent for JSON export, nested by layer then latent."""
    payload: dict[str, dict[str, list[dict[str, object]]]] = {}
    if df.empty:
        return payload

    for (layer, latent), group in df.groupby(["layer_id", "latent_ind"], sort=False):
        layer_int = _safe_int(layer)
        layer_key = str(
            layer_int if layer_int is not None else _format_axis_value(layer)
        )
        latent_int = _safe_int(latent)
        latent_key = str(
            latent_int if latent_int is not None else _format_axis_value(latent)
        )

        records: list[dict[str, object]] = []
        for _, row in group.iterrows():
            domain_idx = _safe_int(row.get("domain_idx"))
            ipr_id = None
            if domain_idx is not None and 0 <= domain_idx < len(entry_accessions):
                ipr_id = entry_accessions[domain_idx]
            records.append(
                {
                    "domain_idx": domain_idx,
                    "ipr_id": ipr_id,
                    "entry_name": row.get("entry_name"),
                }
            )
        payload.setdefault(layer_key, {})[latent_key] = records
    return payload


def print_layer_domain_summary(frames: dict[str, pd.DataFrame]) -> None:
    """Print unique-domain counts per layer for each export cohort."""
    label_map = {
        "high_auc_high_precision_metx": "MetXA / precision",
        "high_auc_high_precision_top2": "Top2 / precision",
        "high_auc_high_lift_metx": "MetXA / lift",
        "high_auc_high_lift_top2": "Top2 / lift",
    }

    def layer_sort_key(layer_val):
        layer_int = _safe_int(layer_val)
        if layer_int is not None:
            return (0, layer_int)
        return (1, str(layer_val))

    for name, df in frames.items():
        label = label_map.get(name, name)
        print(f"{label}:")
        if df.empty:
            print("  no domains retained")
            print()
            continue

        layer_counts: list[tuple[object, int]] = []
        for layer, group in df.groupby("layer_id", sort=False):
            unique_ids: set[int] = set()
            for value in group["domain_idx"]:
                value_int = _safe_int(value)
                if value_int is not None:
                    unique_ids.add(value_int)
            layer_counts.append((layer, len(unique_ids)))

        for layer, count in sorted(layer_counts, key=lambda item: layer_sort_key(item[0])):
            layer_label = _format_axis_value(layer)
            print(f"  layer {layer_label}: {count} unique domains")
        print()


with open(LAYER_LATENT_DIR / "layer_latent_dict_MetXA_0.70.json", "r") as file:
    metx_latents = json.load(file)

with open(LAYER_LATENT_DIR / "layer_latent_dict_Top2_0.70.json", "r") as file:
    top2_latents = json.load(file)

domain_sign_path = DOMAIN_RESULTS_DIR / "significant_domain_pairs (1).csv"
domain_sign_df = pd.read_csv(domain_sign_path)

# %%
# Order the dataframe so latents always appear grouped together
sort_columns = ["layer_id", "latent_ind"]
domain_sign_df = domain_sign_df.sort_values(by=sort_columns, kind="mergesort")
column_order = sort_columns + [
    col for col in domain_sign_df.columns if col not in sort_columns
]
domain_sign_df = domain_sign_df[column_order]

domain_sign_df["in_metx_latents"] = domain_sign_df.apply(
    lambda row: _latent_in_dict(metx_latents, row["layer_id"], row["latent_ind"]),
    axis=1,
)
domain_sign_df["in_top2_latents"] = domain_sign_df.apply(
    lambda row: _latent_in_dict(top2_latents, row["layer_id"], row["latent_ind"]),
    axis=1,
)

domain_sign_df.head()

# %%
output_path = DOMAIN_RESULTS_DIR / "domain_sign_with_latent_flags.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
domain_sign_df.to_csv(output_path, index=False)

# %%
high_auc_mask = (domain_sign_df["mwu_auc"] > 0.95) & (
    domain_sign_df["mwu_p_adj"] < 1e-10
)
high_auc_df = domain_sign_df.loc[high_auc_mask].copy()
high_auc_df = high_auc_df.sort_values(by=sort_columns, kind="mergesort")

high_auc_high_precision_df = high_auc_df.loc[
    high_auc_df["precision_1x"] > 0.5
].copy()

lift_cols = [col for col in high_auc_df.columns if col.startswith("lift_")]
if lift_cols:
    lift_mask = high_auc_df[lift_cols].gt(3).any(axis=1)
    precision_series = (
        high_auc_df["precision_1x"].fillna(0.0)
        if "precision_1x" in high_auc_df
        else pd.Series(0.0, index=high_auc_df.index)
    )
    high_auc_high_lift_df = high_auc_df.loc[
        lift_mask & (precision_series < 0.5)
    ].copy()
else:
    high_auc_high_lift_df = high_auc_df.iloc[0:0].copy()

high_auc_metx_df = high_auc_df.loc[high_auc_df["in_metx_latents"]].copy()
high_auc_top2_df = high_auc_df.loc[high_auc_df["in_top2_latents"]].copy()

high_auc_high_precision_metx_df = high_auc_high_precision_df.loc[
    high_auc_high_precision_df["in_metx_latents"]
].copy()
high_auc_high_precision_top2_df = high_auc_high_precision_df.loc[
    high_auc_high_precision_df["in_top2_latents"]
].copy()

high_auc_high_lift_metx_df = high_auc_high_lift_df.loc[
    high_auc_high_lift_df["in_metx_latents"]
].copy()
high_auc_high_lift_top2_df = high_auc_high_lift_df.loc[
    high_auc_high_lift_df["in_top2_latents"]
].copy()

# %%
high_auc_df.head()
# %%
high_auc_df[["layer_id", "latent_ind", "entry_name", "mwu_auc", "precision_1x"]].head(30)
# %%
high_auc_high_precision_df[["layer_id", "latent_ind", "entry_name", "mwu_auc", "precision_1x"]].head(30)
# %%
display_columns = ["layer_id", "latent_ind", "entry_name", "mwu_auc"] + lift_cols
high_auc_high_lift_df[display_columns].head(30)
# %%
high_auc_metx_df[["layer_id", "latent_ind", "entry_name", "mwu_auc", "precision_1x"]].head(30)
# %%
high_auc_top2_df[["layer_id", "latent_ind", "entry_name", "mwu_auc", "precision_1x"]].head(30)
# %%
high_auc_high_precision_metx_df[["layer_id", "latent_ind", "entry_name", "mwu_auc", "precision_1x"]].head(30)
# %%
high_auc_high_precision_top2_df[["layer_id", "latent_ind", "entry_name", "mwu_auc", "precision_1x"]].head(30)
# %%
high_auc_high_lift_metx_df[display_columns].head(30)
# %%
high_auc_high_lift_top2_df[display_columns].head(30)

# %%
export_dir = output_path.parent
export_frames = {
    "high_auc_high_precision_metx": high_auc_high_precision_metx_df,
    "high_auc_high_precision_top2": high_auc_high_precision_top2_df,
    "high_auc_high_lift_metx": high_auc_high_lift_metx_df,
    "high_auc_high_lift_top2": high_auc_high_lift_top2_df,
}

# %%
try:
    all_properties = torch.load(
        METADATA_DIR / "ptn_fam_tensor_nonzero.pt",
        map_location="cpu",
        weights_only=True,
    )
except TypeError:
    all_properties = torch.load(
        METADATA_DIR / "ptn_fam_tensor_nonzero.pt",
        map_location="cpu",
    )

if isinstance(all_properties, torch.Tensor):
    domain_matrix = all_properties.detach().cpu().numpy() != 0
else:
    domain_matrix = np.asarray(all_properties) != 0

interpro_annotations_nonzero = pd.read_csv(
    METADATA_DIR / "interpro_entry_list_mapping_nonzero.csv"
).reset_index(drop=True)
entry_names = interpro_annotations_nonzero["ENTRY_NAME"].tolist()
entry_accessions = interpro_annotations_nonzero["ENTRY_AC"].tolist()

minimal_frames = {
    name: select_non_overlapping_domains(df, domain_matrix, entry_names)
    for name, df in export_frames.items()
}
export_frames = minimal_frames

print_layer_domain_summary(export_frames)

for name, df in export_frames.items():
    df.to_csv(export_dir / f"{name}.csv", index=False)

for name, df in export_frames.items():
    payload = build_latent_domain_payload(df, entry_accessions)
    (export_dir / f"{name}.json").write_text(json.dumps(payload, indent=2))
