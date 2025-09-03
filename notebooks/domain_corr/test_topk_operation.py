import torch
import sys
import time
sys.path.append('../..')
from utils import load_cached_activations

# Load activations
print("Loading activations...")
tmp = load_cached_activations("../../acts")

# Test on first 3 proteins with detailed output
n_test = 3
print(f"\nTesting top-k operation on {n_test} proteins with detailed steps:")

for protein_ix in range(n_test):
    print(f"\n{'='*60}")
    print(f"PROTEIN {protein_ix}")
    print(f"{'='*60}")
    
    cur_ptn = tmp.load_protein_activations_original_format(protein_ix)
    
    # Show initial info
    first_layer = tmp.layers[0]
    print(f"Total layers: {len(tmp.layers)}")
    print(f"Shape of activations for layer {first_layer}: {cur_ptn['activations'][first_layer].shape}")
    print(f"  → {cur_ptn['activations'][first_layer].shape[0]} positions × {cur_ptn['activations'][first_layer].shape[1]} features")
    
    # Process first 2 layers in detail
    results = []
    for i, l in enumerate(tmp.layers[:2]):
        print(f"\n--- Layer {l} ---")
        acts = cur_ptn['activations'][l]
        print(f"Step 1: Input shape: {acts.shape}")
        
        # Calculate k
        k = max(1, int(0.01 * acts.shape[0]))
        print(f"Step 2: Calculate k (top 1%): max(1, int(0.01 * {acts.shape[0]})) = {k}")
        
        # Get top k values
        top_values, top_indices = torch.topk(acts, k=k, dim=0)
        print(f"Step 3: torch.topk(acts, k={k}, dim=0)")
        print(f"         → top_values.shape: {top_values.shape}")
        print(f"         → This gives the {k} highest values for each of {acts.shape[1]} features")
        
        # Show example for first feature
        print(f"         Example for feature 0:")
        print(f"           Top {k} values: {top_values[:, 0].tolist()}")
        
        # Take mean
        mean_top = top_values.mean(dim=0)
        print(f"Step 4: mean(dim=0) of top values")
        print(f"         → shape after mean: {mean_top.shape}")
        print(f"         → Example: mean of feature 0 = {mean_top[0]:.4f}")
        
        results.append(mean_top)
    
    # Concatenate
    print(f"\n--- Concatenation ---")
    print(f"Step 5: Concatenating {len(tmp.layers)} layers, each with shape ({acts.shape[1]},)")
    tmp_store = torch.cat(tuple([torch.topk(cur_ptn['activations'][l], 
                                            k=max(1, int(0.01 * cur_ptn['activations'][l].shape[0])), 
                                            dim=0)[0].mean(dim=0) 
                                 for l in tmp.layers]))
    print(f"         → Final concatenated shape: {tmp_store.shape}")
    print(f"         → Total features: {len(tmp.layers)} layers × {acts.shape[1]} features/layer = {tmp_store.shape[0]}")

print(f"\n{'='*60}")
print("Test complete!")