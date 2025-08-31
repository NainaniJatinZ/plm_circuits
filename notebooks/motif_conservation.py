# %%
import sys
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

ExtendedActivationInfo = namedtuple('ExtendedActivationInfo', 
        ['activation_value', 'protein_idx', 'token_idx', 'protein_id', 'residue_idx', 'layer_idx', 'latent_idx'])
# Import utility functions
from helpers.utils import load_esm, load_sae_prot, cleanup_cuda
from hook_manager import SAEHookProt

from enrich_act import load_cached_activations, EfficientActivationCache

# All helpers here 

# Define a named tuple to store activation information
ActivationInfo = namedtuple('ActivationInfo', ['activation_value', 'protein_idx', 'token_idx', 'protein_id', 'residue_idx'])
# %%

def find_top_k_activations_streamlined(cache_system, layer_idx, latent_idx, k=100):
    """
    Streamlined approach: single pass, one max per protein, top K globally.
    """
    print(f"Finding top {k} activations for layer {layer_idx}, latent {latent_idx}...")
    print("Using streamlined single-pass approach (one max per protein)")
    
    # Use a min-heap to efficiently maintain top-K
    top_k_heap = []
    
    for protein_idx in tqdm(range(len(cache_system.metadata['proteins'])), desc="Processing proteins"):
        try:
            protein_data = cache_system.metadata['proteins'][protein_idx]
            protein_id = protein_data['protein_id']
            sequence_length = protein_data['sequence_length']
            
            # Load the specific layer file
            layer_file = protein_data[f'layer_{layer_idx}_file']
            
            if cache_system.use_sparse:
                sparse_matrix = load_npz(layer_file)
                latent_column = sparse_matrix[:, latent_idx].toarray().flatten()
            else:
                acts_np = np.load(layer_file)
                latent_column = acts_np[:, latent_idx]
            
            # # Find max activation and its position
            # if len(latent_column) > 0:
            #     valid_length = min(len(latent_column), sequence_length)
            #     valid_activations = latent_column[:valid_length]
                
            #     if len(valid_activations) > 0:
            max_idx = np.argmax(latent_column)
            max_activation = float(latent_column[max_idx])
            
            activation_info = ActivationInfo(
                activation_value=max_activation,
                protein_idx=protein_idx,
                token_idx=max_idx + 1,  # +1 for BOS token offset
                protein_id=protein_id,
                residue_idx=max_idx  # Direct residue position
            )
            
            # Maintain top-K using heap
            if len(top_k_heap) < k:
                heapq.heappush(top_k_heap, (-max_activation, protein_idx, activation_info))
            elif max_activation > -top_k_heap[0][0]:
                heapq.heapreplace(top_k_heap, (-max_activation, protein_idx, activation_info))
                
        except Exception as e:
            print(f"Error processing protein {protein_idx}: {e}")
            continue
    
    # Extract results and sort by activation value (highest first)
    top_k_activations = [item[2] for item in top_k_heap]
    top_k_activations.sort(key=lambda x: x.activation_value, reverse=True)
    
    print(f"\nResults:")
    print(f"- Processed {len(cache_system.metadata['proteins'])} proteins")
    print(f"- Found {len(top_k_activations)} top activations")
    if top_k_activations:
        print(f"- Activation range: {top_k_activations[0].activation_value:.4f} to {top_k_activations[-1].activation_value:.4f}")
    
    return top_k_activations

def extract_neighborhoods(top_activations, protein_sequences, window_size=10, pad_token='X'):
    """
    Extract amino acid neighborhoods around top activating positions.
    Handles proper token-to-residue mapping and pads incomplete neighborhoods.
    
    Args:
        top_activations: List of ActivationInfo tuples
        protein_sequences: Dict mapping protein_id to sequence string
        window_size: Number of residues on each side of the central residue
        pad_token: Token to use for padding when neighborhood extends beyond sequence
    
    Returns:
        List of dicts containing neighborhood information
    """
    neighborhoods = []
    
    for i, act_info in enumerate(tqdm(top_activations, desc="Extracting neighborhoods")):
        protein_id = act_info.protein_id
        residue_idx = act_info.residue_idx - 1 # Use residue index, not token index
        
        if protein_id not in protein_sequences:
            print(f"Warning: Protein {protein_id} not found in sequence database")
            continue
        
        sequence = protein_sequences[protein_id]
        
        # Validate that residue_idx is within the sequence
        if residue_idx >= len(sequence) or residue_idx < 0:
            print(f"Warning: Residue index {residue_idx} out of bounds for protein {protein_id} (length {len(sequence)})")
            continue
        
        # Calculate desired neighborhood boundaries
        desired_start = residue_idx - window_size
        desired_end = residue_idx + window_size + 1
        
        # Calculate actual sequence boundaries
        seq_start = max(0, desired_start)
        seq_end = min(len(sequence), desired_end)
        
        # Extract the actual sequence part
        actual_seq = sequence[seq_start:seq_end]
        
        # Calculate padding needed
        left_padding = seq_start - desired_start  # How many positions we're missing on the left
        right_padding = desired_end - seq_end     # How many positions we're missing on the right
        
        # Create the full neighborhood with padding
        neighborhood_seq = (pad_token * left_padding) + actual_seq + (pad_token * right_padding)
        
        # The central residue is always at position window_size in the padded sequence
        central_pos_in_neighborhood = window_size
        
        neighborhood_info = {
            'rank': i + 1,
            'activation_value': act_info.activation_value,
            'protein_idx': act_info.protein_idx,
            'protein_id': protein_id,
            'token_idx': act_info.token_idx,  # Keep original token index for reference
            'residue_idx': residue_idx,       # Actual amino acid position in sequence
            'central_residue': sequence[residue_idx],
            'neighborhood_seq': neighborhood_seq,
            'neighborhood_length': len(neighborhood_seq),
            'central_pos_in_neighborhood': central_pos_in_neighborhood,
            'full_seq_length': len(sequence),
            'left_padding': left_padding,
            'right_padding': right_padding,
            'actual_start_in_sequence': seq_start,
            'actual_end_in_sequence': seq_end
        }
        
        neighborhoods.append(neighborhood_info)
    
    return neighborhoods

def analyze_conservation(aligned_sequences, positions_to_analyze=None):
    """
    Perform basic conservation analysis on aligned sequences.
    
    Args:
        aligned_sequences: List of aligned amino acid sequences
        positions_to_analyze: List of positions to analyze (None for all)
    
    Returns:
        position_frequencies: Dict of position -> amino acid -> count
    """
    if not aligned_sequences:
        return {}
    
    seq_length = len(aligned_sequences[0])
    position_frequencies = {}
    
    # Analyze all positions if none specified
    if positions_to_analyze is None:
        positions_to_analyze = list(range(seq_length))
    
    for pos in positions_to_analyze:
        position_frequencies[pos] = {}
        
        for seq in aligned_sequences:
            if pos < len(seq):
                aa = seq[pos]
                position_frequencies[pos][aa] = position_frequencies[pos].get(aa, 0) + 1
    
    return position_frequencies

def prepare_logomaker_data(conservation_data, aligned_sequences):
    """
    Convert conservation data to format needed for logomaker.
    
    Args:
        conservation_data: Dict from analyze_conservation
        aligned_sequences: List of aligned sequences
        
    Returns:
        pandas DataFrame with amino acids as rows and positions as columns
    """
    
    if not aligned_sequences or not conservation_data:
        return None
    
    seq_length = len(aligned_sequences[0])
    
    # All 20 standard amino acids (excluding X for now)
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Use simple integer positions for logomaker (it doesn't like multi-character column names)
    positions = list(range(seq_length))
    
    # Initialize count matrix with integer positions
    count_matrix = pd.DataFrame(0, index=amino_acids, columns=positions)
    
    # Fill in the counts
    for pos in range(seq_length):
        if pos in conservation_data:
            for aa, count in conservation_data[pos].items():
                if aa in amino_acids:  # Skip X (padding) for cleaner logo
                    count_matrix.loc[aa, pos] = count
    
    return count_matrix

# %%

# tmp = load_cached_activations('/project/pi_annagreen_umass_edu/jatin/plm_circuits/acts')
# test_protein = tmp.load_protein_activations_original_format(0)
# print(test_protein)

# # Load protein sequences from FASTA file to get actual amino acid sequences
# FASTA_PATH = pathlib.Path("../../plm_interp/uniprot_sprot.fasta")
# print("Loading protein sequences...")
# protein_sequences = {}
# for rec in SeqIO.parse(FASTA_PATH, "fasta"):
#     if len(rec.seq) <= 1022:  # Same filter as used in the activation cache
#         protein_sequences[rec.id] = str(rec.seq)

# print(f"Loaded {len(protein_sequences)} protein sequences")

def find_top_k_activations_multi_layer_vectorized(cache_system, layer_latent_dict, layer_subset=None, k=100):
    """
    Vectorized approach: Process multiple layers and their specific latents simultaneously.
    
    Args:
        cache_system: Your activation cache
        layer_latent_dict: Dict mapping layer (as string) to list of latent indices
                          e.g., {'4': [2311, 2443, 1682], '8': [1234, 5678], '12': [9876]}
        layer_subset: Optional list of layers to process (e.g., [4, 8, 12]). 
                     If None, processes all layers in layer_latent_dict
        k: Number of top activations to keep globally
    
    Returns:
        List of ExtendedActivationInfo with layer_idx and latent_idx added
    """
    # Convert string keys to integers and filter by layer_subset if provided
    if layer_subset is not None:
        filtered_dict = {str(layer): latents for layer, latents in layer_latent_dict.items() 
                        if int(layer) in layer_subset}
    else:
        filtered_dict = layer_latent_dict
    
    # Convert to integer keys for processing
    layer_latent_mapping = {int(layer): latents for layer, latents in filtered_dict.items()}
    
    total_combinations = sum(len(latents) for latents in layer_latent_mapping.values())
    print(f"Finding top {k} activations across {len(layer_latent_mapping)} layers, {total_combinations} total layer-latent combinations...")
    
    # Enhanced ActivationInfo to include layer and latent
    from collections import namedtuple
    ExtendedActivationInfo = namedtuple('ExtendedActivationInfo', 
        ['activation_value', 'protein_idx', 'token_idx', 'protein_id', 'residue_idx', 'layer_idx', 'latent_idx'])
    
    # Global heap for top K across all combinations
    global_top_k_heap = []
    
    # Process proteins one by one to manage memory
    for protein_idx in tqdm(range(len(cache_system.metadata['proteins'])), desc="Processing proteins"):
        try:
            protein_data = cache_system.metadata['proteins'][protein_idx]
            protein_id = protein_data['protein_id']
            
            # Load all layers for this protein at once
            protein_layer_data = {}
            for layer_idx, latent_indices in layer_latent_mapping.items():
                layer_file = protein_data[f'layer_{layer_idx}_file']
                
                if cache_system.use_sparse:
                    sparse_matrix = load_npz(layer_file)
                    # Extract only the latents we care about for this layer
                    acts_subset = sparse_matrix[:, latent_indices].toarray()  # [seq_len, num_latents_for_this_layer]
                else:
                    acts_np = np.load(layer_file)
                    acts_subset = acts_np[:, latent_indices]  # [seq_len, num_latents_for_this_layer]
                
                protein_layer_data[layer_idx] = acts_subset
            
            # Vectorized processing: find max across sequence for each (layer, latent) combination
            for layer_idx, latent_indices in layer_latent_mapping.items():
                acts = protein_layer_data[layer_idx]  # [seq_len, num_latents_for_this_layer]
                
                # Find max values and indices for all latents at once
                max_values = np.max(acts, axis=0)  # [num_latents_for_this_layer]
                max_indices = np.argmax(acts, axis=0)  # [num_latents_for_this_layer]
                
                # Process each latent for this layer
                for i, latent_idx in enumerate(latent_indices):
                    max_activation = float(max_values[i])
                    max_residue_idx = int(max_indices[i])
                    
                    activation_info = ExtendedActivationInfo(
                        activation_value=max_activation,
                        protein_idx=protein_idx,
                        token_idx=max_residue_idx + 1,  # +1 for BOS token offset
                        protein_id=protein_id,
                        residue_idx=max_residue_idx,
                        layer_idx=layer_idx,
                        latent_idx=latent_idx
                    )
                    
                    # Maintain global top-K
                    if len(global_top_k_heap) < k:
                        heapq.heappush(global_top_k_heap, (-max_activation, protein_idx, activation_info))
                    elif max_activation > -global_top_k_heap[0][0]:
                        heapq.heapreplace(global_top_k_heap, (-max_activation, protein_idx, activation_info))
                        
        except Exception as e:
            print(f"Error processing protein {protein_idx}: {e}")
            continue
    
    # Extract and sort results
    top_k_activations = [item[2] for item in global_top_k_heap]
    top_k_activations.sort(key=lambda x: x.activation_value, reverse=True)
    
    print(f"\nResults across {len(layer_latent_mapping)} layers, {total_combinations} layer-latent combinations:")
    print(f"- Found {len(top_k_activations)} top activations")
    if top_k_activations:
        print(f"- Activation range: {top_k_activations[0].activation_value:.4f} to {top_k_activations[-1].activation_value:.4f}")
    
    return top_k_activations

def find_top_k_activations_per_latent(cache_system, layer_latent_dict, layer_subset=None, k=100, protein=""):
    """
    Find top K activations FOR EACH latent individually.
    Load each protein once, process all latents for that protein.
    
    Args:
        cache_system: Your activation cache
        layer_latent_dict: Dict mapping layer (as string) to list of latent indices
        layer_subset: Optional list of layers to process
        k: Number of top activations to keep PER latent
    
    Returns:
        Dict mapping (layer_idx, latent_idx) -> list of top K activations for that latent
    """
    # Convert string keys to integers and filter by layer_subset if provided
    if layer_subset is not None:
        filtered_dict = {str(layer): latents for layer, latents in layer_latent_dict.items() 
                        if int(layer) in layer_subset}
    else:
        filtered_dict = layer_latent_dict
    
    # Convert to integer keys for processing
    layer_latent_mapping = {int(layer): latents for layer, latents in filtered_dict.items()}
    
    total_combinations = sum(len(latents) for latents in layer_latent_mapping.values())
    print(f"Finding top {k} activations for EACH of {total_combinations} latents across {len(layer_latent_mapping)} layers...")
    
    # Enhanced ActivationInfo to include layer and latent
    from collections import namedtuple
    ExtendedActivationInfo = namedtuple('ExtendedActivationInfo', 
        ['activation_value', 'protein_idx', 'token_idx', 'protein_id', 'residue_idx', 'layer_idx', 'latent_idx'])
    
    # Create separate top-K heap for each latent
    latent_heaps = {}
    for layer_idx, latent_indices in layer_latent_mapping.items():
        for latent_idx in latent_indices:
            latent_heaps[(layer_idx, latent_idx)] = []
    
    # Process proteins one by one (OUTER LOOP - load each protein once)
    for protein_idx in tqdm(range(len(cache_system.metadata['proteins'])), desc="Processing proteins"):
        try:
            protein_data = cache_system.metadata['proteins'][protein_idx]
            protein_id = protein_data['protein_id']
            
            # Load all layers for this protein at once
            protein_layer_data = {}
            for layer_idx, latent_indices in layer_latent_mapping.items():
                layer_file = protein_data[f'layer_{layer_idx}_file']
                
                if cache_system.use_sparse:
                    sparse_matrix = load_npz(layer_file)
                    # Extract only the latents we care about for this layer
                    acts_subset = sparse_matrix[:, latent_indices].toarray()  # [seq_len, num_latents_for_this_layer]
                else:
                    acts_np = np.load(layer_file)
                    acts_subset = acts_np[:, latent_indices]  # [seq_len, num_latents_for_this_layer]
                
                protein_layer_data[layer_idx] = acts_subset
            
            # Process all latents for this protein (INNER LOOP - process each latent)
            for layer_idx, latent_indices in layer_latent_mapping.items():
                acts = protein_layer_data[layer_idx]  # [seq_len, num_latents_for_this_layer]
                
                # Find max values and indices for all latents at once
                max_values = np.max(acts, axis=0)  # [num_latents_for_this_layer]
                max_indices = np.argmax(acts, axis=0)  # [num_latents_for_this_layer]
                
                # Process each latent for this layer
                for i, latent_idx in enumerate(latent_indices):
                    max_activation = float(max_values[i])
                    max_residue_idx = int(max_indices[i])
                    
                    activation_info = ExtendedActivationInfo(
                        activation_value=max_activation,
                        protein_idx=protein_idx,
                        token_idx=max_residue_idx + 1,  # +1 for BOS token offset
                        protein_id=protein_id,
                        residue_idx=max_residue_idx,
                        layer_idx=layer_idx,
                        latent_idx=latent_idx
                    )
                    
                    # Maintain top-K for THIS SPECIFIC latent
                    latent_key = (layer_idx, latent_idx)
                    heap = latent_heaps[latent_key]
                    
                    if len(heap) < k:
                        heapq.heappush(heap, (-max_activation, protein_idx, activation_info))
                    elif max_activation > -heap[0][0]:
                        heapq.heapreplace(heap, (-max_activation, protein_idx, activation_info))
                        
        except Exception as e:
            print(f"Error processing protein {protein_idx}: {e}")
            continue
    
    # Extract and sort results for each latent
    results_per_latent = {}
    
    for latent_key, heap in latent_heaps.items():
        layer_idx, latent_idx = latent_key
        
        # Extract results and sort by activation value (highest first)
        latent_results = [item[2] for item in heap]
        latent_results.sort(key=lambda x: x.activation_value, reverse=True)
        results_per_latent[latent_key] = latent_results
        
        print(f"Layer {layer_idx}, Latent {latent_idx}: {len(latent_results)} activations, "
              f"range {latent_results[0].activation_value:.4f} to {latent_results[-1].activation_value:.4f}")
    
    print(f"\nCompleted! Found top {k} activations for each of {len(results_per_latent)} latents")
    
    # Save results automatically
    import pickle
    save_path = f'../intermediate_ops/top_activations_per_latent_k{k}_{protein}.pkl'
    with open(save_path, 'wb') as f:
        # Convert to dictionaries to avoid pickle issues
        save_dict = {}
        for (layer_idx, latent_idx), activations in results_per_latent.items():
            key_str = f"layer_{layer_idx}_latent_{latent_idx}"
            save_dict[key_str] = [
                {
                    'activation_value': act.activation_value,
                    'protein_idx': act.protein_idx,
                    'token_idx': act.token_idx,
                    'protein_id': act.protein_id,
                    'residue_idx': act.residue_idx,
                    'layer_idx': act.layer_idx,
                    'latent_idx': act.latent_idx
                }
                for act in activations
            ]
        pickle.dump(save_dict, f)
    
    print(f"Results automatically saved to: {save_path}")
    
    return results_per_latent

# %%

# protein = "MetXA" # "MetXA" or "Top2"
# target_recovery_percent = 0.70

# with open(f'../results/layer_latent_dicts/layer_latent_dict_{protein}_{target_recovery_percent:.2f}.json', 'r') as f:
#     layer_latent_dict = json.load(f)
# print(layer_latent_dict)

# # %%
# tempo = {"4": [59, 117, 136, 430, 441, 830, 1024, 1052, 1072, 1273, 1297, 1395, 1505, 1533, 1744, 1763, 1794, 1799, 1800, 1931, 1960, 2222, 2322, 2324, 2473, 2495, 2721, 2967, 3044, 3431, 3616, 3639, 3677, 3728, 3732, 3775, 3924, 4058]}
# top_activations = find_top_k_activations_per_latent(
#     cache_system=tmp, 
#     layer_latent_dict=tempo, #layer_latent_dict,
#     layer_subset=[4],  # Only process these layers
#     k=100,
#     protein=protein
# )

# # Convert namedtuples to dictionaries before saving
# top_activations_dict = []
# for act in top_activations:
#     act_dict = {
#         'activation_value': act.activation_value,
#         'protein_idx': act.protein_idx,
#         'token_idx': act.token_idx,
#         'protein_id': act.protein_id,
#         'residue_idx': act.residue_idx,
#         'layer_idx': act.layer_idx,
#         'latent_idx': act.latent_idx
#     }
#     top_activations_dict.append(act_dict)

# # Now save the dictionaries
# with open(f'../intermediate_ops/top_activations_layers_4_latentdict_top100_{protein}_{target_recovery_percent}_single_contact_global.pkl', 'wb') as f:
#     pickle.dump(top_activations_dict, f)

# %%

# with open(f'../intermediate_ops/top_activations_layers_4_latentdict_top100_{protein}_{target_recovery_percent}_single_contact_global.pkl', 'rb') as f:
#     top_activations = pickle.load(f)

# %%

# # Load the stored activation data
# import pickle
# import os

# # Load the stored activations 
# stored_activations_path = f'../intermediate_ops/top_activations_per_latent_k100_{protein}.pkl'

# if os.path.exists(stored_activations_path):
#     with open(stored_activations_path, 'rb') as f:
#         stored_results = pickle.load(f)
#     print(f"Loaded stored activations from {stored_activations_path}")
#     print(f"Number of layer-latent combinations: {len(stored_results)}")
    
#     # Show available keys
#     print("\nAvailable layer-latent combinations:")
#     for key in list(stored_results.keys())[:10]:  # Show first 10
#         print(f"  {key}")
#     print(f"... and {len(stored_results) - 10} more" if len(stored_results) > 10 else "")
# else:
#     print(f"File {stored_activations_path} not found. Need to run the activation extraction first.")

# %%

# Create modified count matrix - only keep top 3 amino acids per position
def prepare_logomaker_data_top3(conservation_data, aligned_sequences, top_n=3):
    """
    Convert conservation data to format needed for logomaker, but only keep top N amino acids per position.
    
    Args:
        conservation_data: Dict from analyze_conservation
        aligned_sequences: List of aligned sequences
        top_n: Number of top amino acids to keep per position (default=3)
        
    Returns:
        pandas DataFrame with amino acids as rows and positions as columns, 
        with only top N amino acids per position (others set to 0)
    """
    
    if not aligned_sequences or not conservation_data:
        return None
    
    seq_length = len(aligned_sequences[0])
    
    # All 20 standard amino acids (excluding X for now)
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    # Use simple integer positions for logomaker
    positions = list(range(seq_length))
    
    # Initialize count matrix with integer positions
    count_matrix = pd.DataFrame(0, index=amino_acids, columns=positions)
    
    # Fill in the counts and then filter to top N per position
    for pos in range(seq_length):
        if pos in conservation_data:
            # Get all amino acid counts for this position
            position_counts = {}
            for aa, count in conservation_data[pos].items():
                if aa in amino_acids:  # Skip X (padding) for cleaner logo
                    position_counts[aa] = count
            
            # Sort amino acids by count (descending) and keep only top N
            sorted_aas = sorted(position_counts.items(), key=lambda x: x[1], reverse=True)
            top_aas = sorted_aas[:top_n]
            
            # Only set counts for top N amino acids, others remain 0
            for aa, count in top_aas:
                count_matrix.loc[aa, pos] = count
                
            print(f"Position {pos}: Top {top_n} AAs = {[f'{aa}({count})' for aa, count in top_aas]}")
    
    return count_matrix

# Batch generate clean sequence logos for all latents
# import os
# import matplotlib.pyplot as plt
# import logomaker
# from tqdm import tqdm

def generate_clean_logo_for_latent(activations_dict, protein_sequences, layer_idx, latent_idx, 
                                   window_size=10, activation_threshold=0.05, save_dir="../results/sequence_logos_clean/"):
    """
    Generate a clean sequence logo with minimal labeling for manual arrangement.
    
    Returns True if successful, False if failed
    """
    
    # Get the data
    target_key = f"layer_{layer_idx}_latent_{latent_idx}"
    if target_key not in activations_dict:
        print(f"❌ No data for {target_key}")
        return False
    
    # Convert to ActivationInfo format and filter
    target_activations = []
    for act_dict in activations_dict[target_key]:
        if act_dict['activation_value'] >= activation_threshold:
            act_info = ActivationInfo(
                activation_value=act_dict['activation_value'],
                protein_idx=act_dict['protein_idx'], 
                token_idx=act_dict['token_idx'],
                protein_id=act_dict['protein_id'],
                residue_idx=act_dict['residue_idx']
            )
            target_activations.append(act_info)
    
    if len(target_activations) < 5:  # Need at least some activations
        print(f"❌ Too few activations for {target_key}: {len(target_activations)}")
        return False
    
    # Extract neighborhoods and analyze conservation
    neighborhoods = extract_neighborhoods(target_activations, protein_sequences, window_size=window_size)
    if not neighborhoods:
        print(f"❌ No neighborhoods for {target_key}")
        return False
        
    aligned_sequences = [n['neighborhood_seq'] for n in neighborhoods]
    conservation_data = analyze_conservation(aligned_sequences)
    count_matrix = prepare_logomaker_data_top3(conservation_data, aligned_sequences, top_n=3)
    
    if count_matrix is None or count_matrix.empty:
        print(f"❌ No count matrix for {target_key}")
        return False
    
    # Prepare logomaker data
    logo_df = count_matrix.T
    logo_df.index.name = 'pos'
    aa_cols = list('ACDEFGHIKLMNPQRSTVWY')
    logo_df = logo_df.reindex(columns=aa_cols, fill_value=0)
    seq_len = len(logo_df)
    
    # Generate CLEAN plot
    plt.rcParams['figure.dpi'] = 300  # High DPI for publication
    plt.rcParams['font.size'] = 10
    
    fig, ax = plt.subplots(figsize=(9, 3))  # Your preferred ratio
    
    logomaker.Logo(
        logo_df,
        ax=ax,
        fade_below=0.5,
        stack_order='big_on_top',
        color_scheme='NajafabadiEtAl2017'
    )
    
    # Position labels (keep the numbers)
    rel_labels = [f'{i - window_size:+d}' if i != window_size else '0'
                  for i in range(seq_len)]
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(rel_labels, fontsize=9)
    
    # REMOVE axis labels (you'll add in Canva)
    ax.set_xlabel('')  # Remove "Position relative to center"  
    ax.set_ylabel('')  # Remove "Info (bits)"
    
    # Short title format
    ax.set_title(f'L{layer_idx}-{latent_idx}', pad=10, weight='bold', fontsize=11)
    
    # Subtle center line
    ax.axvline(window_size, color='red', ls='--', lw=1, alpha=.5)
    
    # Clean appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=.2)
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.1)  # Minimal padding
    
    # Save the file
    os.makedirs(save_dir, exist_ok=True)
    filename = f"logo_L{layer_idx}_latent{latent_idx}_clean.png"
    filepath = os.path.join(save_dir, filename)
    
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close(fig)  # Important: close to free memory
    
    print(f"✅ Saved {filename}")
    return True
# %%

# # Batch process ALL latents to generate clean sequence logos
# print(f"Starting batch processing of {len(stored_results)} latents...")
# print("This may take a while - generating high-quality logos...")

# # Parse all layer-latent combinations from stored data
# all_combinations = []
# for key in stored_results.keys():
#     # Parse keys like "layer_4_latent_237"
#     parts = key.split('_')
#     if len(parts) == 4 and parts[0] == 'layer' and parts[2] == 'latent':
#         layer_idx = int(parts[1])
#         latent_idx = int(parts[3])
#         all_combinations.append((layer_idx, latent_idx))

# print(f"Found {len(all_combinations)} layer-latent combinations to process")

# # Sort combinations for organized processing
# all_combinations.sort()

# # Process all combinations
# successful = 0
# failed = 0
# failed_list = []

# for layer_idx, latent_idx in tqdm(all_combinations, desc="Generating logos"):
#     try:
#         success = generate_clean_logo_for_latent(
#             stored_results, protein_sequences,
#             layer_idx, latent_idx,
#             window_size=10,
#             activation_threshold=0.05,
#             save_dir="../results/sequence_logos_clean/"
#         )
        
#         if success:
#             successful += 1
#         else:
#             failed += 1
#             failed_list.append(f"L{layer_idx}-{latent_idx}")
            
#     except Exception as e:
#         print(f"❌ Error processing L{layer_idx}-{latent_idx}: {e}")
#         failed += 1
#         failed_list.append(f"L{layer_idx}-{latent_idx}")

# print(f"\n🎉 Batch processing complete!")
# print(f"✅ Successful: {successful}")
# print(f"❌ Failed: {failed}")

# if failed_list:
#     print(f"\nFailed latents: {failed_list[:10]}")  # Show first 10 failures
#     if len(failed_list) > 10:
#         print(f"... and {len(failed_list) - 10} more")
        
# print(f"\n📁 All successful logos saved to: ../results/sequence_logos_clean/")
# print(f"💡 Ready for manual arrangement in Canva!")

# %%

def find_top_k_activations_efficient_filtered(cache_system, layer_latent_dict, k=100, save_results=False, protein_name="", verbose=True):
    """
    EFFICIENT version: Load each protein once, process all needed latents for that protein.
    Adapted from the original find_top_k_activations_per_latent but optimized for filtering.
    
    Args:
        cache_system: Your activation cache
        layer_latent_dict: Dict mapping layer (as string) to list of latent indices
                          e.g., {"4": [59, 117], "8": [1234, 5678]}
        k: Number of top activations to keep PER latent
        save_results: Whether to save results to pickle file
        protein_name: Name for saving (only used if save_results=True)
        verbose: Whether to show progress bars
    
    Returns:
        Dict mapping f"layer_{layer_idx}_latent_{latent_idx}" -> list of activation dicts
    """
    
    # Convert to integer keys for processing
    layer_latent_mapping = {int(layer): latents for layer, latents in layer_latent_dict.items()}
    
    total_combinations = sum(len(latents) for latents in layer_latent_mapping.values())
    if verbose:
        print(f"Finding top {k} activations for EACH of {total_combinations} latents across {len(layer_latent_mapping)} layers...")
    
    # Create separate top-K heap for each latent
    latent_heaps = {}
    for layer_idx, latent_indices in layer_latent_mapping.items():
        for latent_idx in latent_indices:
            latent_heaps[(layer_idx, latent_idx)] = []
    
    # Process proteins one by one (OUTER LOOP - load each protein once)
    protein_iterator = range(len(cache_system.metadata['proteins']))
    if verbose:
        protein_iterator = tqdm(protein_iterator, desc="Processing proteins")
        
    for protein_idx in protein_iterator:
        try:
            protein_data = cache_system.metadata['proteins'][protein_idx]
            protein_id = protein_data['protein_id']
            
            # Load all layers for this protein at once
            protein_layer_data = {}
            for layer_idx, latent_indices in layer_latent_mapping.items():
                layer_file = protein_data[f'layer_{layer_idx}_file']
                
                if cache_system.use_sparse:
                    sparse_matrix = load_npz(layer_file)
                    # Extract only the latents we care about for this layer
                    acts_subset = sparse_matrix[:, latent_indices].toarray()  # [seq_len, num_latents_for_this_layer]
                else:
                    acts_np = np.load(layer_file)
                    acts_subset = acts_np[:, latent_indices]  # [seq_len, num_latents_for_this_layer]
                
                protein_layer_data[layer_idx] = acts_subset
            
            # Process all latents for this protein (INNER LOOP - process each latent)
            for layer_idx, latent_indices in layer_latent_mapping.items():
                acts = protein_layer_data[layer_idx]  # [seq_len, num_latents_for_this_layer]
                
                # Find max values and indices for all latents at once
                max_values = np.max(acts, axis=0)  # [num_latents_for_this_layer]
                max_indices = np.argmax(acts, axis=0)  # [num_latents_for_this_layer]
                
                # Process each latent for this layer
                for i, latent_idx in enumerate(latent_indices):
                    max_activation = float(max_values[i])
                    max_residue_idx = int(max_indices[i])
                    
                    activation_info = {
                        'activation_value': max_activation,
                        'protein_idx': protein_idx,
                        'token_idx': max_residue_idx + 1,  # +1 for BOS token offset
                        'protein_id': protein_id,
                        'residue_idx': max_residue_idx,
                        'layer_idx': layer_idx,
                        'latent_idx': latent_idx
                    }
                    
                    # Maintain top-K for THIS SPECIFIC latent
                    latent_key = (layer_idx, latent_idx)
                    heap = latent_heaps[latent_key]
                    
                    if len(heap) < k:
                        heapq.heappush(heap, (-max_activation, protein_idx, activation_info))
                    elif max_activation > -heap[0][0]:
                        heapq.heapreplace(heap, (-max_activation, protein_idx, activation_info))
                        
        except Exception as e:
            if verbose:
                print(f"Error processing protein {protein_idx}: {e}")
            continue
    
    # Extract and sort results for each latent
    results_per_latent = {}
    
    for latent_key, heap in latent_heaps.items():
        layer_idx, latent_idx = latent_key
        
        # Extract results and sort by activation value (highest first)
        latent_results = [item[2] for item in heap]
        latent_results.sort(key=lambda x: x['activation_value'], reverse=True)
        
        # Format as expected by downstream functions
        result_key = f"layer_{layer_idx}_latent_{latent_idx}"
        results_per_latent[result_key] = latent_results
        
        if verbose:
            print(f"Layer {layer_idx}, Latent {latent_idx}: {len(latent_results)} activations, "
                  f"range {latent_results[0]['activation_value']:.4f} to {latent_results[-1]['activation_value']:.4f}")
    
    if verbose:
        print(f"Completed! Found top {k} activations for each of {len(results_per_latent)} latents")
    
    # Optional saving
    if save_results and protein_name:
        os.makedirs('../intermediate_ops', exist_ok=True)
        save_path = f'../intermediate_ops/top_activations_per_latent_k{k}_{protein_name}.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(results_per_latent, f)
        if verbose:
            print(f"Results saved to: {save_path}")
    
    return results_per_latent


def generate_clean_logo_for_latent_quiet(activations_dict, protein_sequences, layer_idx, latent_idx, 
                                         window_size=10, activation_threshold=0.05, save_dir="../results/sequence_logos_clean/"):
    """
    Generate a clean sequence logo with minimal output.
    
    Returns: (success: bool, conservation_dict: dict or None)
    """
    
    # Get the data
    target_key = f"layer_{layer_idx}_latent_{latent_idx}"
    if target_key not in activations_dict:
        return False, None
    
    # Convert to ActivationInfo format and filter
    target_activations = []
    for act_dict in activations_dict[target_key]:
        if act_dict['activation_value'] >= activation_threshold:
            act_info = ActivationInfo(
                activation_value=act_dict['activation_value'],
                protein_idx=act_dict['protein_idx'], 
                token_idx=act_dict['token_idx'],
                protein_id=act_dict['protein_id'],
                residue_idx=act_dict['residue_idx']
            )
            target_activations.append(act_info)
    
    if len(target_activations) < 5:  # Need at least some activations
        return False, None
    
    # Extract neighborhoods and analyze conservation
    neighborhoods = extract_neighborhoods(target_activations, protein_sequences, window_size=window_size)
    if not neighborhoods:
        return False, None
        
    aligned_sequences = [n['neighborhood_seq'] for n in neighborhoods]
    conservation_data = analyze_conservation(aligned_sequences)
    count_matrix = prepare_logomaker_data_top3(conservation_data, aligned_sequences, top_n=3)
    
    if count_matrix is None or count_matrix.empty:
        return False, None
    
    # Generate conservation motif (>50% conservation gets actual residue, else 'X')
    conservation_motif = []
    total_sequences = len(aligned_sequences)
    for pos in range(len(aligned_sequences[0])):
        if pos in conservation_data:
            # Find most common residue at this position
            max_count = 0
            most_common_residue = 'X'
            for aa, count in conservation_data[pos].items():
                if aa != 'X' and count > max_count:  # Ignore padding
                    max_count = count
                    most_common_residue = aa
            
            # If >50% conservation, use the residue, else 'X'
            if max_count / total_sequences > 0.5:
                conservation_motif.append(most_common_residue)
            else:
                conservation_motif.append('X')
        else:
            conservation_motif.append('X')
    
    conservation_dict = {
        'layer': layer_idx,
        'latent': latent_idx,
        'motif': ''.join(conservation_motif)
    }
    
    # Prepare logomaker data
    logo_df = count_matrix.T
    logo_df.index.name = 'pos'
    aa_cols = list('ACDEFGHIKLMNPQRSTVWY')
    logo_df = logo_df.reindex(columns=aa_cols, fill_value=0)
    seq_len = len(logo_df)
    
    # Generate CLEAN plot
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 10
    
    fig, ax = plt.subplots(figsize=(9, 3))
    
    logomaker.Logo(
        logo_df,
        ax=ax,
        fade_below=0.5,
        stack_order='big_on_top',
        color_scheme='NajafabadiEtAl2017'
    )
    
    # Position labels
    rel_labels = [f'{i - window_size:+d}' if i != window_size else '0'
                  for i in range(seq_len)]
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(rel_labels, fontsize=9)
    
    # Clean appearance
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(f'L{layer_idx}-{latent_idx}', pad=10, weight='bold', fontsize=11)
    ax.axvline(window_size, color='red', ls='--', lw=1, alpha=.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=.2)
    
    fig.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.1)
    
    # Save the file
    os.makedirs(save_dir, exist_ok=True)
    filename = f"logo_L{layer_idx}_latent{latent_idx}_clean.png"
    filepath = os.path.join(save_dir, filename)
    
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    plt.close(fig)
    
    return True, conservation_dict


def generateLogos(layer_latent_dict, 
                  cache_system_path="/project/pi_annagreen_umass_edu/jatin/plm_circuits/acts",
                  protein_sequences_path="../../plm_interp/uniprot_sprot.fasta",
                  window_size=10, 
                  k=100, 
                  activation_threshold=0.05, 
                  save_dir="../results/sequence_logos_clean/",
                  verbose=True):
    """
    Operationalized function to generate sequence logos for latents.
    EFFICIENT: Filters existing logos first, then processes remaining latents in batch.
    
    Args:
        layer_latent_dict: Dict mapping layer (as string) to list of latent indices
                          e.g., {"4": [59, 117, 136], "8": [1234, 5678]}
        cache_system_path: Path to the activation cache directory
        protein_sequences_path: Path to the FASTA file with protein sequences
        window_size: Number of residues on each side of the central residue (default: 10)
        k: Number of top activations to keep per latent (default: 100)
        activation_threshold: Minimum activation value to include (default: 0.05)
        save_dir: Directory to save the logo PNG files (default: "../results/sequence_logos_clean/")
        verbose: Whether to print progress information (default: True)
    
    Returns:
        Dict containing conservation motifs for each latent:
        {
            "L4_59": {"layer": 4, "latent": 59, "motif": "XXGXXXXXXXPXXXXXX"},
            "L4_117": {"layer": 4, "latent": 117, "motif": "XXXXXAXXXXXXX"},
            ...
        }
    """
    
    if verbose:
        print("Starting EFFICIENT operationalized logo generation...")
        print(f"Cache system: {cache_system_path}")
        print(f"Protein sequences: {protein_sequences_path}")
        print(f"Save directory: {save_dir}")
    
    # 1. FIRST: Filter out latents that already have logos
    if verbose:
        print("Filtering latents that already have logos...")
        
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists
    
    filtered_layer_latent_dict = {}
    skipped_count = 0
    total_original = sum(len(latents) for latents in layer_latent_dict.values())
    
    for layer_str, latents in layer_latent_dict.items():
        layer_idx = int(layer_str)
        remaining_latents = []
        
        for latent_idx in latents:
            logo_path = os.path.join(save_dir, f"logo_L{layer_idx}_latent{latent_idx}_clean.png")
            if os.path.exists(logo_path):
                skipped_count += 1
            else:
                remaining_latents.append(latent_idx)
        
        if remaining_latents:  # Only include layers that have latents to process
            filtered_layer_latent_dict[layer_str] = remaining_latents
    
    remaining_total = sum(len(latents) for latents in filtered_layer_latent_dict.values())
    
    if verbose:
        print(f"Original latents: {total_original}")
        print(f"Skipped existing: {skipped_count}")
        print(f"Remaining to process: {remaining_total}")
    
    # If nothing to process, return early
    if remaining_total == 0:
        if verbose:
            print("All logos already exist! Nothing to process.")
        return {}
    
    # 2. Load cache system and protein sequences
    if verbose:
        print("Loading activation cache...")
    from enrich_act import load_cached_activations
    cache_system = load_cached_activations(cache_system_path)
    
    if verbose:
        print("Loading protein sequences...")
    from Bio import SeqIO
    protein_sequences = {}
    for rec in SeqIO.parse(protein_sequences_path, "fasta"):
        if len(rec.seq) <= 1022:  # Same filter as used in the activation cache
            protein_sequences[rec.id] = str(rec.seq)
    
    if verbose:
        print(f"Loaded {len(protein_sequences)} protein sequences")
    
    # 3. EFFICIENT: Process all remaining latents in one batch
    if verbose:
        print(f"Processing {remaining_total} latents across {len(filtered_layer_latent_dict)} layers...")
    
    activations_dict = find_top_k_activations_efficient_filtered(
        cache_system=cache_system,
        layer_latent_dict=filtered_layer_latent_dict,
        k=k,
        save_results=False,
        verbose=verbose
    )
    
    # 4. Generate logos for all latents
    conservation_results = {}
    failed = 0
    
    if verbose:
        print(f"\nGenerating logos for {len(activations_dict)} latents...")
        
    # Use tqdm for logo generation too
    logo_iterator = activations_dict.items()
    if verbose:
        logo_iterator = tqdm(logo_iterator, desc="Generating logos")
    
    for target_key, _ in logo_iterator:
        # Parse the key to get layer and latent
        parts = target_key.split('_')
        if len(parts) == 4 and parts[0] == 'layer' and parts[2] == 'latent':
            layer_idx = int(parts[1])
            latent_idx = int(parts[3])
            
            try:
                # Generate logo and get conservation motif
                success, conservation_dict = generate_clean_logo_for_latent_quiet(
                    activations_dict=activations_dict,
                    protein_sequences=protein_sequences,
                    layer_idx=layer_idx,
                    latent_idx=latent_idx,
                    window_size=window_size,
                    activation_threshold=activation_threshold,
                    save_dir=save_dir
                )
                
                if success and conservation_dict:
                    key = f"L{layer_idx}_{latent_idx}"
                    conservation_results[key] = conservation_dict
                else:
                    failed += 1
                    if verbose:
                        print(f"Failed: L{layer_idx}-{latent_idx}")
                        
            except Exception as e:
                failed += 1
                if verbose:
                    print(f"Error L{layer_idx}-{latent_idx}: {str(e)[:50]}")
    
    # 5. Summary
    if verbose:
        print(f"\nLogo generation complete!")
        print(f"Generated: {len(conservation_results)} new logos")
        print(f"Skipped existing: {skipped_count} logos") 
        print(f"Failed: {failed} logos")
        print(f"Saved to: {save_dir}")
        
        # Show some example motifs
        if conservation_results:
            print(f"\nExample conservation motifs:")
            for i, (key, data) in enumerate(list(conservation_results.items())[:5]):
                print(f"   {key}: {data['motif']}")
            if len(conservation_results) > 5:
                print(f"   ... and {len(conservation_results) - 5} more")
    
    return conservation_results


# %%
# TEST CASE - Example usage of the operationalized function

if __name__ == "__main__":
    print("Testing the operationalized generateLogos function...")
    
    # Small test case with just a few latents
    test_layer_latent_dict = {
        "4": [897, 1474, 3204, 1509, 2119, 104, 3576, 2154, 2797, 3501, 1525, 2487, 1080, 794]  # Just 3 latents from layer 4 for testing
    }
    
    print("Test configuration:")
    print(f"   Layer-latent dict: {test_layer_latent_dict}")
    print(f"   Total latents to process: {sum(len(latents) for latents in test_layer_latent_dict.values())}")
    
    try:
        # Call the operationalized function
        conservation_results = generateLogos(
            layer_latent_dict=test_layer_latent_dict,
            cache_system_path="/project/pi_annagreen_umass_edu/jatin/plm_circuits/acts",
            protein_sequences_path="../../plm_interp/uniprot_sprot.fasta",
            window_size=10,
            k=100,
            activation_threshold=0.05,
            save_dir="../results/sequence_logos_clean/",  # Use separate test directory
            verbose=True
        )
        
        print(f"\nTest completed successfully!")
        print(f"Generated {len(conservation_results)} conservation motifs")
        
        if conservation_results:
            print(f"\nGenerated motifs:")
            for key, data in conservation_results.items():
                print(f"   {key}: {data['motif']}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()