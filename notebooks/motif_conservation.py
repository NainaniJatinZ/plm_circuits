# %%
"""
Motif Conservation Analysis Pipeline

1. Pick a latent you like to do an initial experiment on (eventually do all)
2. Find the top 100 activations across all proteins. So even if there's multiple highly activated residues in one protein, take all of them. Note down the position of each of the top activations and note down which protein
3. For each top activating residue, get the neighborhood of residues (say +- 10 just for example)
4. It is already aligned because the central residue in each neighborhood is a max activated residues
5. Make the sequence logo as per the wiki link I sent, with the central residue of the logo being the residue that "seeded" the neighborhood due to its being in the top-100 most highly activated residues

Important Notes:
- Protein language models add special tokens (BOS/EOS), so token positions != residue positions
- Token 0: <cls>/<bos>, Tokens 1 to N: amino acids, Token N+1: <eos>
- We properly map token indices to residue indices and handle padding for incomplete neighborhoods
"""

# %%
import sys
sys.path.append('../plm_circuits')

import torch
import numpy as np
import json
import time
import os
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz, load_npz
from Bio import SeqIO
import pathlib
import heapq
from collections import namedtuple

# Import utility functions
from helpers.utils import load_esm, load_sae_prot, cleanup_cuda
from hook_manager import SAEHookProt

from enrich_act import load_cached_activations, EfficientActivationCache

# %%

tmp = load_cached_activations('/project/pi_annagreen_umass_edu/jatin/plm_circuits/acts')

# %%

test_protein = tmp.load_protein_activations_original_format(0)
print(test_protein)

# %%

test_protein['activations'][12][:, 2112].shape

# %%

# Load protein sequences from FASTA file to get actual amino acid sequences
FASTA_PATH = pathlib.Path("../../plm_interp/uniprot_sprot.fasta")
print("Loading protein sequences...")
protein_sequences = {}
for rec in SeqIO.parse(FASTA_PATH, "fasta"):
    if len(rec.seq) <= 1022:  # Same filter as used in the activation cache
        protein_sequences[rec.id] = str(rec.seq)

print(f"Loaded {len(protein_sequences)} protein sequences")

# %%

# Define a named tuple to store activation information
ActivationInfo = namedtuple('ActivationInfo', ['activation_value', 'protein_idx', 'token_idx', 'protein_id', 'residue_idx'])


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

# %% 1. Finding top (protein, residue) pairs
layer_idx = 12
latent_idx = 2112
top_k = 100
window_size = 10

# Find top K activations across all proteins (using streamlined approach)
top_activations = find_top_k_activations_streamlined(
    tmp, layer_idx, latent_idx, k=top_k
)
# %%
import pickle
with open(f'top_activations_layer{layer_idx}_latent{latent_idx}_top{top_k}_streamlined.pkl', 'wb') as f:
    pickle.dump(top_activations, f)

# %%

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
        residue_idx = act_info.residue_idx  # Use residue index, not token index
        
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

# %% 2. Extracting neighborhoods around top activating positions
 
# Extract neighborhoods around top activating positions
neighborhoods = extract_neighborhoods(top_activations, protein_sequences, window_size=window_size)

print(f"\nExtracted {len(neighborhoods)} neighborhoods")

# Display first few neighborhoods
print("\nTop 10 neighborhoods:")
for i, neighborhood in enumerate(neighborhoods[:10]):
    central_res = neighborhood['central_residue']
    seq = neighborhood['neighborhood_seq']
    central_pos = neighborhood['central_pos_in_neighborhood']
    act_val = neighborhood['activation_value']
    residue_pos = neighborhood['residue_idx']
    token_pos = neighborhood['token_idx']
    
    # Create a visual representation with the central residue highlighted
    seq_display = seq[:central_pos] + f"[{central_res}]" + seq[central_pos+1:]
    
    # Show both residue position (in sequence) and token position (in model)
    print(f"{i+1:2d}. Act={act_val:.4f}, {neighborhood['protein_id']}, residue_pos={residue_pos} (token={token_pos}): {seq_display}")
    
    # Show padding info if any
    if neighborhood['left_padding'] > 0 or neighborhood['right_padding'] > 0:
        print(f"    Padding: {neighborhood['left_padding']} left, {neighborhood['right_padding']} right")

# %%

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

# Prepare data for conservation analysis
def prepare_conservation_data(neighborhoods):
    """
    Prepare the neighborhood data for conservation analysis and sequence logo generation.
    
    Returns:
        aligned_sequences: List of sequences all aligned by their central residue
        central_residues: List of central amino acids
        activation_values: List of corresponding activation values
    """
    aligned_sequences = []
    central_residues = []
    activation_values = []
    
    for neighborhood in neighborhoods:
        aligned_sequences.append(neighborhood['neighborhood_seq'])
        central_residues.append(neighborhood['central_residue'])
        activation_values.append(neighborhood['activation_value'])
    
    return aligned_sequences, central_residues, activation_values

def generate_sequence_logo_working(count_matrix, layer_idx, latent_idx, window_size, aligned_sequences):
    """
    The actual working version - simple and reliable.
    """
    import logomaker
    import matplotlib.pyplot as plt
    
    # Sanity check
    if count_matrix is None or count_matrix.empty:
        raise ValueError("count_matrix is empty – build it first!")
    
    # transpose so rows = positions, columns = amino acids
    logo_df = count_matrix.T            # shape: (positions, 20 AA's)
    logo_df.index.name = 'pos'          # nice index name
    
    # make sure only standard AA columns remain & in canonical order
    aa_cols = list('ACDEFGHIKLMNPQRSTVWY')
    logo_df = logo_df.reindex(columns=aa_cols, fill_value=0)
    
    seq_len = len(logo_df)
    
    # Plot
    plt.rcParams['figure.dpi']  = 300
    plt.rcParams['font.size']   = 12
    
    fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.5), 6))
    
    logomaker.Logo(
        logo_df,
        ax=ax,
        fade_below=0.5,
        stack_order='big_on_top',
        color_scheme='NajafabadiEtAl2017'
    )
    
    # x-axis labels = relative positions
    rel_labels = [f'{i - window_size:+d}' if i != window_size else '0'
                  for i in range(seq_len)]
    
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(rel_labels)
    ax.set_xlabel('Position relative to center')
    ax.set_ylabel('Information Content (bits)')
    ax.set_title(
        f'Sequence Logo – Layer {layer_idx}, Latent {latent_idx}\n'
        f'Top {len(aligned_sequences)} Activating Neighborhoods (±{window_size} residues)',
        pad=20, weight='bold'
    )
    
    # highlight centre residue
    ax.axvline(window_size, color='red', ls='--', lw=2, alpha=.7)
    ax.text(window_size, ax.get_ylim()[1]*0.95, 'Activating\nResidue',
            ha='center', va='top', color='red', weight='bold', fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=.3)
    fig.tight_layout()
    
    # Save & show
    png_name = f'sequence_logo_layer{layer_idx}_latent{latent_idx}_top{len(aligned_sequences)}.png'
    pdf_name = f'sequence_logo_layer{layer_idx}_latent{latent_idx}_top{len(aligned_sequences)}.pdf'
    fig.savefig(png_name, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_name, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved logo as {png_name} and {pdf_name}")
    plt.show()
    
    return fig

def prepare_logomaker_data(conservation_data, aligned_sequences):
    """
    Convert conservation data to format needed for logomaker.
    
    Args:
        conservation_data: Dict from analyze_conservation
        aligned_sequences: List of aligned sequences
        
    Returns:
        pandas DataFrame with amino acids as rows and positions as columns
    """
    import pandas as pd
    import numpy as np
    
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

aligned_sequences = [n['neighborhood_seq'] for n in neighborhoods]
conservation_data = analyze_conservation(aligned_sequences)
count_matrix = prepare_logomaker_data(conservation_data, aligned_sequences)

# Step 4: Generate logo (working version)
if count_matrix is not None:
    fig = generate_sequence_logo_fixed(count_matrix, layer_idx, latent_idx, window_size, aligned_sequences)
    

# %% OLD CODE 

def find_top_k_activations_super_efficient(cache_system, layer_idx, latent_idx, k=100, 
                                           min_activation_threshold=0.1, max_proteins_to_analyze=500):
    """
    Super efficiently find top K activations by directly accessing sparse files.
    Avoids loading full protein data when we only need one layer/latent.
    
    Args:
        cache_system: EfficientActivationCache object
        layer_idx: Layer index 
        latent_idx: Latent/feature index
        k: Number of top activations to return
        min_activation_threshold: Minimum max activation for a protein to be considered
        max_proteins_to_analyze: Maximum number of top proteins to analyze in detail
    
    Returns:
        List of ActivationInfo tuples sorted by activation value (highest first)
    """
    print(f"Finding top {k} activations for layer {layer_idx}, latent {latent_idx}...")
    print(f"Using SUPER efficient direct file access approach")
    
    # Stage 1: Lightning-fast scan using direct sparse file access
    print("\nStage 1: Lightning scan with direct file access...")
    protein_max_activations = []
    
    for protein_idx in tqdm(range(len(cache_system.metadata['proteins'])), desc="Lightning scan"):
        try:
            protein_data = cache_system.metadata['proteins'][protein_idx]
            protein_id = protein_data['protein_id']
            
            # Direct access to the specific layer file (much faster!)
            layer_file = protein_data[f'layer_{layer_idx}_file']
            
            if cache_system.use_sparse:
                # Load only the sparse matrix for this layer
                sparse_matrix = load_npz(layer_file)
                # Extract only the column we need (latent_idx)
                latent_column = sparse_matrix[:, latent_idx].toarray().flatten()
            else:
                # Load dense matrix for this layer only
                acts_np = np.load(layer_file)
                latent_column = acts_np[:, latent_idx]
            
            # Simple max - forget about special tokens for speed
            if len(latent_column) > 0:
                max_activation = float(latent_column.max())
                protein_max_activations.append((max_activation, protein_idx, protein_id))
                
        except Exception as e:
            print(f"Error processing protein {protein_idx}: {e}")
            continue
    
    # Use heap for efficient top-K selection instead of full sorting
    print(f"Stage 1 complete: Found {len(protein_max_activations)} proteins")
    
    # Get top proteins efficiently
    top_proteins = heapq.nlargest(max_proteins_to_analyze, protein_max_activations, key=lambda x: x[0])
    
    # Filter by threshold
    promising_proteins = [(max_act, protein_idx, protein_id) for max_act, protein_idx, protein_id in top_proteins 
                         if max_act >= min_activation_threshold]
    
    print(f"Found {len(promising_proteins)} promising proteins (max_act >= {min_activation_threshold})")
    if promising_proteins:
        print(f"Max activation range: {promising_proteins[0][0]:.4f} to {promising_proteins[-1][0]:.4f}")
    
    # Stage 2: Detailed analysis on promising proteins only
    print(f"\nStage 2: Detailed analysis of {len(promising_proteins)} promising proteins...")
    all_activations = []
    
    for max_act, protein_idx, protein_id in tqdm(promising_proteins, desc="Detailed analysis"):
        try:
            protein_data = cache_system.metadata['proteins'][protein_idx]
            sequence_length = protein_data['sequence_length']
            
            # Direct file access again
            layer_file = protein_data[f'layer_{layer_idx}_file']
            
            if cache_system.use_sparse:
                sparse_matrix = load_npz(layer_file)
                latent_column = sparse_matrix[:, latent_idx].toarray().flatten()
            else:
                acts_np = np.load(layer_file)
                latent_column = acts_np[:, latent_idx]
            
            # Go through all positions - simplified without special token handling
            for pos_idx in range(min(len(latent_column), sequence_length)):
                activation_value = float(latent_column[pos_idx])
                
                all_activations.append(ActivationInfo(
                    activation_value=activation_value,
                    protein_idx=protein_idx,
                    token_idx=pos_idx + 1,  # Add 1 for token position (assuming BOS at 0)
                    protein_id=protein_id,
                    residue_idx=pos_idx  # Direct position in sequence
                ))
                
        except Exception as e:
            print(f"Error in detailed analysis of protein {protein_idx}: {e}")
            continue
    
    # Use heap for top-K instead of full sort
    top_k_activations = heapq.nlargest(k, all_activations, key=lambda x: x.activation_value)
    
    print(f"\nResults:")
    print(f"- Analyzed {len(promising_proteins)} proteins in detail")
    print(f"- Found {len(all_activations)} total activations from promising proteins") 
    print(f"- Top {k} activation values: {[f'{act.activation_value:.4f}' for act in top_k_activations[:10]]}")
    
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
        residue_idx = act_info.residue_idx  # Use residue index, not token index
        
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

# %%

# Configuration
layer_idx = 12
latent_idx = 2112
top_k = 100
window_size = 10

print(f"Analyzing layer {layer_idx}, latent {latent_idx}")
print(f"Looking for top {top_k} activations with ±{window_size} residue neighborhoods")
print(f"Total proteins in dataset: {len(tmp.metadata['proteins'])}")

# Quick estimate of speedup
total_proteins = len(tmp.metadata['proteins'])
avg_seq_length = np.mean([p['sequence_length'] for p in tmp.metadata['proteins']])
naive_operations = total_proteins * avg_seq_length
efficient_operations = total_proteins + (500 * avg_seq_length)  # Stage 1 + Stage 2
speedup_estimate = naive_operations / efficient_operations

print(f"Average sequence length: {avg_seq_length:.1f}")
print(f"Estimated speedup: {speedup_estimate:.1f}x (from {naive_operations:,.0f} to {efficient_operations:,.0f} operations)")

# %%

# Find top K activations across all proteins (using SUPER efficient direct file access)
top_activations = find_top_k_activations_super_efficient(
    tmp, layer_idx, latent_idx, k=top_k,
    min_activation_threshold=0.1,  # Only consider proteins with max activation >= 0.1
    max_proteins_to_analyze=500    # Analyze at most 500 promising proteins in detail
)


# %% save top activations
import pickle
with open(f'top_activations_layer{layer_idx}_latent{latent_idx}_top{top_k}.pkl', 'wb') as f:
    pickle.dump(top_activations, f)

# %%

# Extract neighborhoods around top activating positions
neighborhoods = extract_neighborhoods(top_activations, protein_sequences, window_size=window_size)

print(f"\nExtracted {len(neighborhoods)} neighborhoods")

# Display first few neighborhoods
print("\nTop 10 neighborhoods:")
for i, neighborhood in enumerate(neighborhoods[:10]):
    central_res = neighborhood['central_residue']
    seq = neighborhood['neighborhood_seq']
    central_pos = neighborhood['central_pos_in_neighborhood']
    act_val = neighborhood['activation_value']
    residue_pos = neighborhood['residue_idx']
    token_pos = neighborhood['token_idx']
    
    # Create a visual representation with the central residue highlighted
    seq_display = seq[:central_pos] + f"[{central_res}]" + seq[central_pos+1:]
    
    # Show both residue position (in sequence) and token position (in model)
    print(f"{i+1:2d}. Act={act_val:.4f}, {neighborhood['protein_id']}, residue_pos={residue_pos} (token={token_pos}): {seq_display}")
    
    # Show padding info if any
    if neighborhood['left_padding'] > 0 or neighborhood['right_padding'] > 0:
        print(f"    Padding: {neighborhood['left_padding']} left, {neighborhood['right_padding']} right")

# %%

# Prepare data for conservation analysis
def prepare_conservation_data(neighborhoods):
    """
    Prepare the neighborhood data for conservation analysis and sequence logo generation.
    
    Returns:
        aligned_sequences: List of sequences all aligned by their central residue
        central_residues: List of central amino acids
        activation_values: List of corresponding activation values
    """
    aligned_sequences = []
    central_residues = []
    activation_values = []
    
    for neighborhood in neighborhoods:
        aligned_sequences.append(neighborhood['neighborhood_seq'])
        central_residues.append(neighborhood['central_residue'])
        activation_values.append(neighborhood['activation_value'])
    
    return aligned_sequences, central_residues, activation_values

aligned_sequences, central_residues, activation_values = prepare_conservation_data(neighborhoods)

print(f"\nPrepared {len(aligned_sequences)} aligned sequences for conservation analysis")
print(f"Central residue distribution: {dict(sorted([(res, central_residues.count(res)) for res in set(central_residues)], key=lambda x: x[1], reverse=True))}")

# %%

# Save results for further analysis
results = {
    'layer_idx': layer_idx,
    'latent_idx': latent_idx,
    'top_k': top_k,
    'window_size': window_size,
    'neighborhoods': neighborhoods,
    'aligned_sequences': aligned_sequences,
    'central_residues': central_residues,
    'activation_values': activation_values
}

# Save to file
import pickle
output_file = f"motif_conservation_layer{layer_idx}_latent{latent_idx}_top{top_k}.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print(f"\nResults saved to: {output_file}")

# %%

# Basic conservation analysis
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

# Analyze conservation patterns
conservation_data = analyze_conservation(aligned_sequences)

print("\nConservation analysis for top 5 positions around center:")
central_pos = window_size  # The central position in our aligned sequences

for offset in [-2, -1, 0, 1, 2]:
    pos = central_pos + offset
    if pos in conservation_data:
        print(f"Position {offset:+d} (absolute pos {pos}):")
        # Sort amino acids by frequency
        sorted_aas = sorted(conservation_data[pos].items(), key=lambda x: x[1], reverse=True)
        total_count = sum(conservation_data[pos].values())
        
        for aa, count in sorted_aas[:5]:  # Show top 5 amino acids
            freq = count / total_count
            print(f"  {aa}: {count:3d} ({freq:.2%})")
        print()

# %%

!pip install logomaker



# %%

# =======================================================================
# 100 % reliable sequence‑logo cell (transpose‑and‑plot)
# =======================================================================

import logomaker
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# 0. sanity checks
# -----------------------------------------------------------------------
if count_matrix is None or count_matrix.empty:
    raise ValueError("count_matrix is empty – build it first!")

# transpose so rows = positions, columns = amino acids
logo_df = count_matrix.T            # shape: (positions, 20 AA’s)
logo_df.index.name = 'pos'          # nice index name

# make sure only standard AA columns remain & in canonical order
aa_cols = list('ACDEFGHIKLMNPQRSTVWY')
logo_df = logo_df.reindex(columns=aa_cols, fill_value=0)

seq_len = len(logo_df)

# -----------------------------------------------------------------------
# 1. plot
# -----------------------------------------------------------------------
plt.rcParams['figure.dpi']  = 300
plt.rcParams['font.size']   = 12

fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.5), 6))

logomaker.Logo(
    logo_df,
    ax=ax,
    fade_below=0.5,
    stack_order='big_on_top',
    color_scheme='NajafabadiEtAl2017'
)

# x‑axis labels = relative positions
rel_labels = [f'{i - window_size:+d}' if i != window_size else '0'
              for i in range(seq_len)]

ax.set_xticks(range(seq_len))
ax.set_xticklabels(rel_labels)
ax.set_xlabel('Position relative to center')
ax.set_ylabel('Information Content (bits)')
ax.set_title(
    f'Sequence Logo – Layer {layer_idx}, Latent {latent_idx}\n'
    f'Top {len(aligned_sequences)} Activating Neighborhoods (±{window_size} residues)',
    pad=20, weight='bold'
)

# highlight centre residue
ax.axvline(window_size, color='red', ls='--', lw=2, alpha=.7)
ax.text(window_size, ax.get_ylim()[1]*0.95, 'Activating\nResidue',
        ha='center', va='top', color='red', weight='bold', fontsize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', alpha=.3)
fig.tight_layout()

# -----------------------------------------------------------------------
# 2. save & show
# -----------------------------------------------------------------------
png_name = f'sequence_logo_layer{layer_idx}_latent{latent_idx}_top{len(aligned_sequences)}.png'
pdf_name = f'sequence_logo_layer{layer_idx}_latent{latent_idx}_top{len(aligned_sequences)}.pdf'
fig.savefig(png_name, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_name, format='pdf', bbox_inches='tight', facecolor='white')
print(f"Saved logo as {png_name} and {pdf_name}")
plt.show()




# %%
# Generate Sequence Logo
print("Generating sequence logo...")

def prepare_logomaker_data(conservation_data, aligned_sequences):
    """
    Convert conservation data to format needed for logomaker.
    
    Args:
        conservation_data: Dict from analyze_conservation
        aligned_sequences: List of aligned sequences
        
    Returns:
        pandas DataFrame with amino acids as rows and positions as columns
    """
    import pandas as pd
    import numpy as np
    
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

# Prepare data for logomaker
count_matrix = prepare_logomaker_data(conservation_data, aligned_sequences)

if count_matrix is not None and not count_matrix.empty:
    try:
        import logomaker
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Create sequence logo
        print(f"Creating sequence logo for {len(aligned_sequences)} aligned sequences...")
        
        # Set high DPI for publication quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 12
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(max(8, len(original_positions) * 0.5), 6))
        
        # Keep the original count matrix with integer columns for statistics
        original_count_matrix = count_matrix.astype(int)
        original_positions = list(original_count_matrix.columns)  # Keep track of original positions
        
        print(f"Count matrix shape: {original_count_matrix.shape}")
        print(f"Original positions: {original_positions}")
        
        # Create single character column names for logomaker (it's very picky about this)
        import string
        single_char_cols = []
        for i in range(len(original_count_matrix.columns)):
            if i < 26:
                single_char_cols.append(string.ascii_uppercase[i])
            else:
                # For sequences longer than 26, use double chars
                first_char = string.ascii_uppercase[i // 26 - 1]
                second_char = string.ascii_uppercase[i % 26]
                single_char_cols.append(first_char + second_char)
        
        # Create new matrix with single-character column names ONLY for logomaker
        logo_matrix = original_count_matrix.copy()
        logo_matrix.columns = single_char_cols
        
        print(f"Using single-char columns for logo: {single_char_cols}")
        
        # Create the logo
        logo = logomaker.Logo(logo_matrix, 
                            ax=ax,
                            fade_below=0.5,
                            stack_order='big_on_top',
                            color_scheme='NajafabadiEtAl2017',
                            font_name='Arial Rounded MT Bold')
        print("Successfully created logo with single-char column names")
        
        # Customize the plot
        ax.set_title(f'Sequence Logo - Layer {layer_idx}, Latent {latent_idx}\n'
                    f'Top {len(aligned_sequences)} Activating Neighborhoods (±{window_size} residues)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Position relative to center', fontsize=12)
        ax.set_ylabel('Information Content (bits)', fontsize=12)
        
        # Create relative position labels for the x-axis using original positions
        position_labels = []
        for i, original_pos in enumerate(original_positions):
            relative_pos = original_pos - window_size
            if relative_pos == 0:
                position_labels.append('0')
            else:
                position_labels.append(f'{relative_pos:+d}')
        
        ax.set_xticks(range(len(logo_matrix.columns)))
        ax.set_xticklabels(position_labels)
        
        # Highlight the central position (should be at index window_size)
        center_pos = window_size  # This is the index in the original positions
        ax.axvline(x=center_pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(center_pos, ax.get_ylim()[1] * 0.95, 'Activating\nResidue', 
               ha='center', va='top', color='red', fontweight='bold', fontsize=10)
        
        # Add some styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        
        # Save the figure
        logo_filename = f'sequence_logo_layer{layer_idx}_latent{latent_idx}_top{len(aligned_sequences)}.png'
        fig.savefig(logo_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Sequence logo saved as: {logo_filename}")
        
        # Also save as PDF for vector graphics
        pdf_filename = f'sequence_logo_layer{layer_idx}_latent{latent_idx}_top{len(aligned_sequences)}.pdf'
        fig.savefig(pdf_filename, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"Vector version saved as: {pdf_filename}")
        
        plt.show()
        
        # Print some statistics about the logo
        print(f"\nSequence Logo Statistics:")
        print(f"- Total sequences: {len(aligned_sequences)}")
        print(f"- Alignment length: {len(original_positions)} positions")
        
        # Calculate statistics using the original count matrix (with integer positions)
        try:
            # Convert to information content for better statistics
            stats_info_matrix = logomaker.transform_matrix(original_count_matrix, 
                                                         from_type='counts', 
                                                         to_type='information')
            print(f"- Information content range: {stats_info_matrix.values.min():.2f} to {stats_info_matrix.values.max():.2f} bits")
            
            # Find most informative positions using original positions
            position_info_values = []
            for i, original_pos in enumerate(original_positions):
                info_sum = stats_info_matrix.iloc[:, i].sum()
                position_info_values.append((original_pos, info_sum))
            
            # Sort by information content
            position_info_values.sort(key=lambda x: x[1], reverse=True)
            
            print(f"- Most informative positions:")
            for i, (pos, info) in enumerate(position_info_values[:5]):
                relative_pos = f'{pos - window_size:+d}' if pos != window_size else '0'
                print(f"  {i+1}. Position {relative_pos}: {info:.2f} bits")
                
        except Exception as e:
            print(f"- Could not calculate information content statistics: {e}")
            # Fallback: show most conserved positions by frequency
            print("- Most conserved positions (by frequency):")
            position_max_values = []
            for i, original_pos in enumerate(original_positions):
                max_count = original_count_matrix.iloc[:, i].max()
                total_at_pos = original_count_matrix.iloc[:, i].sum()
                freq = max_count / total_at_pos if total_at_pos > 0 else 0
                position_max_values.append((original_pos, freq, max_count))
            
            # Sort by frequency
            position_max_values.sort(key=lambda x: x[1], reverse=True)
            
            for i, (pos, freq, max_count) in enumerate(position_max_values[:5]):
                relative_pos = f'{pos - window_size:+d}' if pos != window_size else '0'
                print(f"  {i+1}. Position {relative_pos}: {freq:.1%} conservation")
        
    except ImportError:
        print("Error: logomaker not installed. Install with: pip install logomaker")
    except Exception as e:
        print(f"Error creating sequence logo: {e}")
        print("Falling back to simple text-based visualization...")
        
        # Simple text-based logo as fallback
        print("\nSimple text-based sequence logo:")
        print("=" * 80)
        
        # Use the original_count_matrix with proper position mapping
        for i, original_pos in enumerate(original_positions):
            pos_data = original_count_matrix.iloc[:, i]
            if pos_data.sum() > 0:
                # Get top amino acid
                top_aa = pos_data.idxmax()
                top_count = pos_data.max()
                freq = top_count / pos_data.sum()
                
                # Create relative position label
                relative_pos = original_pos - window_size
                if relative_pos == 0:
                    pos_label = '0'
                else:
                    pos_label = f'{relative_pos:+d}'
                
                bar_length = int(freq * 20)  # Scale for display
                bar = '█' * bar_length
                
                print(f"Pos {pos_label:>3}: {top_aa} {bar:<20} ({freq:.1%})")
        
        print("=" * 80)

else:
    print("No logo data available - check conservation analysis")

# %%

# Additional conservation statistics
def print_conservation_stats(conservation_data, aligned_sequences):
    """Print detailed conservation statistics."""
    print(f"\n{'='*60}")
    print(f"CONSERVATION ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total aligned sequences: {len(aligned_sequences)}")
    print(f"Alignment length: {len(aligned_sequences[0]) if aligned_sequences else 0}")
    print(f"Central position: {window_size} (0-indexed)")
    
    if conservation_data:
        print(f"\nMost conserved positions:")
        
        # Calculate conservation score for each position (simple Shannon entropy-based)
        position_conservation = {}
        total_seqs = len(aligned_sequences)
        
        for pos, aa_counts in conservation_data.items():
            total_at_pos = sum(aa_counts.values())
            if total_at_pos > 0:
                # Calculate Shannon entropy (lower = more conserved)
                entropy = 0
                for count in aa_counts.values():
                    if count > 0:
                        freq = count / total_at_pos
                        entropy -= freq * np.log2(freq)
                
                # Conservation score (higher = more conserved)
                max_entropy = np.log2(len(aa_counts))  # Maximum possible entropy
                conservation_score = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
                position_conservation[pos] = conservation_score
        
        # Show top 5 most conserved positions
        sorted_positions = sorted(position_conservation.items(), key=lambda x: x[1], reverse=True)
        
        for i, (pos, score) in enumerate(sorted_positions[:5]):
            offset = pos - window_size
            most_common_aa = max(conservation_data[pos].items(), key=lambda x: x[1])
            aa, count = most_common_aa
            freq = count / sum(conservation_data[pos].values())
            
            print(f"  {i+1}. Position {offset:+2d}: {aa} ({freq:.1%}, conservation={score:.3f})")
        
        # Central residue analysis
        if window_size in conservation_data:
            central_stats = conservation_data[window_size]
            total_central = sum(central_stats.values())
            print(f"\nCentral residue (position 0) composition:")
            sorted_central = sorted(central_stats.items(), key=lambda x: x[1], reverse=True)
            for aa, count in sorted_central[:5]:
                freq = count / total_central
                print(f"  {aa}: {count:3d} sequences ({freq:.1%})")

print_conservation_stats(conservation_data, aligned_sequences)

print(f"\nAnalysis complete! You now have:")
print(f"- Top {top_k} highest activating positions across all proteins")
print(f"- ±{window_size} residue neighborhoods around each position")
print(f"- Aligned sequences ready for sequence logo generation")
print(f"- Conservation analysis data")
print(f"- Publication-quality sequence logo")
print(f"- All data saved to {output_file}")

# %%
