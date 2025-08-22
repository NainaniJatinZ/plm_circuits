# plm_circuits
SAE Circuit Discovery for Protein Language Models

This repo relies on the following repos:
1. Zhang et al. 
2. Adams et al. 
3. Simons et al.




## Notebooks quickstart

Requirements
- Python 3.10+
- CUDA GPU recommended
- Install deps: `pip install -r requirements.txt`

Data
- Ensure `data/full_seq_dict.json` exists (used by all notebooks).

Results directory
- Notebooks write outputs to `results/` relative to the repo.

Notebooks (script-style)
- `notebooks/performance_recovery_curves.py`
  - Computes IG effects and layer-wise recovery curves.
  - Outputs plots to `results/`.
- `notebooks/path_patching.py`
  - Computes path patching effects using precomputed latent lists in `results/layer_latent_dict_top2.json`.
  - Saves results JSON to `results/`.
- `notebooks/feature_effect.py`
  - Analyzes feature clusters vs circuit composition and performance.
  - Saves figures and pickles to `results/`.

Helpers
- Shared utilities live under `plm_circuits/analysis/` and `plm_circuits/data/`.

Run
- From repo root, run: `python notebooks/performance_recovery_curves.py`
- Or open in Jupyter; cells are delimited with `# %%`.

Cite Zhang et al