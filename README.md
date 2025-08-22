# plm_circuits
SAE Circuit Discovery for Protein Language Models

## What this repo does
End-to-end workflow to discover, quantify, and validate mechanistic circuits in protein LMs with sparse autoencoders (SAEs):
- Performance recovery curves per-layer and globally
- Feature cluster “ablation effect” and quantity in the circuit
- Path patching to rank edges between features across layers
- Large-scale activation capture (enrichment) over Swiss‑Prot
- Downstream analyses: motif conservation and domain correlation

This research builds on:
1. Zhang et al. (`https://github.com/zzhangzzhang/pLMs-interpretability`) – task setup, sequence data, SSE positions
2. Adams et al. (`https://github.com/etowahadams/interprot/tree/main`) – SAE inference code and weights
3. Simons et al. (`https://github.com/ElanaPearl/interPLM`) – protein 3D and latent visualization

## Environment and prerequisites
- Python 3.9+ with CUDA‑enabled GPU recommended
- Install Python deps (minimum):
  - torch, numpy, pandas, matplotlib, tqdm, scipy, scikit-learn, biopython, logomaker
- Data and helpers required in this repo:
  - `data/full_seq_dict.json`
  - `data/protein_params.py` providing `sse_dict`, `fl_dict`, `protein_name`, `protein2pdb`
  - `data/feature_clusters_hypotheses.py` providing `feature_clusters_MetXA`, `feature_clusters_Top2`
- ESM-2 and SAE weights are loaded via helpers in `plm_circuits/helpers`. Ensure those weights/paths are available as configured there.

## End-to-end workflow (recommended order)
1) Performance recovery curves → 2) Feature effect → 3) Path patching → 4) Enrichment (activations) → 5) Motif conservation / Domain correlation

Below, each step lists the notebook/script, key knobs to set, and the outputs produced.

### 1) Performance recovery curves
File: `notebooks/performance_recovery_curves.py`

What it does:
- Computes causal effects and per-layer recovery curves as a function of top‑k features
- Computes global recovery curves (SAE features + error terms)
- Saves a per‑protein per‑threshold latent dictionary for downstream steps

Key variables to set inside the script:
- `protein = "MetXA"` or `"Top2"`
- `target_recovery_percent = 0.7` (used for k selection and filenames)

Outputs:
- Figures: `results/performance_recovery_curves/*.png` and `.pdf`
- Latent dict JSON (used later):
  - `results/layer_latent_dicts/layer_latent_dict_{protein}_{target_recovery_percent}.json`
- Flank length analysis figure: `results/flank_jump_showcase/`

Run:
```bash
python notebooks/performance_recovery_curves.py
```

### 2) Feature effect (cluster ablation and quantity)
File: `notebooks/feature_effect.py`

What it does:
- Uses the causal effects to size the per‑layer circuit at the chosen recovery threshold
- For each hypothesized feature cluster, computes: percentage in circuit, recovery with exclusion, absolute/relative drops
- Writes a concise per‑layer summary

Key variables to set:
- `protein = "MetXA"` or `"Top2"`
- `target_recovery_percent = 0.7`
- Uses `data/feature_clusters_hypotheses.py`

Inputs it expects (from step 1):
- `results/layer_latent_dicts/layer_latent_dict_{protein}_{target_recovery_percent}.json`

Outputs:
- Markdown report:
  - `results/feature_cluster_analysis/feature_cluster_analysis_{protein}_{target_recovery_percent}.md`

Run:
```bash
python notebooks/feature_effect.py
```

### 3) Path patching (rank edges between features)
File: `notebooks/path_patching.py`

What it does:
- For layer pairs in causal order, ablates an upstream latent, caches the induced downstream change, patches it in, and measures metric deltas
- Ranks edges by absolute metric change and produces diagnostics and a markdown report of interpretable edges

Key variables to set:
- `protein = "MetXA"` or `"Top2"`
- `target_recovery_percent = 0.7`
- Optionally filter/choose layers (`n_layers` section) to control compute

Inputs it expects (from step 1):
- `results/layer_latent_dicts/layer_latent_dict_{protein}_{target_recovery_percent}.json`

Outputs:
- JSON of ranked edges: `results/path_patching_results/path_patching_results_{protein}_{target_recovery_percent}.json`
- Plot: `results/path_patching_results/edge_strength_distribution_{protein}_{target_recovery_percent}.(png|pdf)`
- Interpretable edges markdown: `results/path_patching_results/interpretable_edges_{protein}_{target_recovery_percent}.md`

Run:
```bash
python notebooks/path_patching.py
```

### 4) Enrichment: large‑scale activation capture
File: `notebooks/enrich_act.py`

What it does:
- Processes up to tens of thousands of Swiss‑Prot proteins and caches masked SAE latents per layer
- Sparse storage by default with per‑protein files and a `metadata.json`

Key variables to set (in `__main__`):
- `FASTA_PATH` (Swiss‑Prot FASTA)
- `SAVE_PATH` (output directory for the activation cache)
- `N_PROTEINS` and `RANDOM_SEED`

Outputs:
- Under `SAVE_PATH/`: per‑protein activation files and `metadata.json`

Run:
```bash
python notebooks/enrich_act.py
```

### 5a) Motif conservation (uses enrichment cache)
File: `notebooks/motif_conservation.py`

What it does:
- Loads the cached activations and finds top‑K activating neighborhoods per latent
- Builds conservation logos (top‑N amino acids per position) for manual figure assembly

Key variables to set:
- Path in `load_cached_activations(<SAVE_PATH>)`
- `FASTA_PATH` for sequences matching your activation cache
- `layer_latent_dict` (JSON) or your custom list of (layer, latent) to analyze

Outputs:
- Clean sequence logos: `results/sequence_logos_clean/*.png`
- Optional intermediate pickles under `intermediate_ops/`

Run:
```bash
python notebooks/motif_conservation.py
```

### 5b) Domain correlation (uses on‑the‑fly activations)
File: `notebooks/domain_corr.py`

What it does:
- For a set of targets `(layer, unit, domain_id, pos_fasta)`, computes per‑protein scores via several aggregators, then AUROC with stratified bootstrap CI and Mann–Whitney U; adjusts p‑values across tests

Key variables to set:
- `FASTA_SPROT` (reviewed Swiss‑Prot for negatives)
- `TARGETS` (list of `(layer, unit, domain_id, pos_fasta)`)
- Optional: aggregators, bootstrap iters, length‑matching strategy, output dir

Outputs:
- CSV: `reconcile_out_main_gpu/results_reconcile.csv`
- Plots per aggregator: `reconcile_out_main_gpu/plot_*.png`
- `meta.json` with run configuration

Run:
```bash
python notebooks/domain_corr.py
```

## Reproduction tips
- GPU memory: all steps benefit from a GPU; reduce layers/latents or batch sizes if memory‑constrained
- Determinism: set seeds where provided (e.g., `set_seed(0)`, `RANDOM_SEED`)
- Paths: several notebooks use absolute paths; update to your environment before running
- Latent dict hand‑off: the file written in Step 1 is consumed by Steps 2–3 and optionally by 5a

## Repository layout (selected)
- `notebooks/` – analysis scripts described above
- `plm_circuits/` – hooks, attribution, helpers
- `data/` – sequences, protein params, cluster hypotheses
- `results/` – figures, analyses, caches written by notebooks

## Citation and prior work

