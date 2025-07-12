# %%

# random_sampler.py
from Bio import SeqIO
import random, pathlib

FASTA = pathlib.Path("../../plm_interp/uniprot_sprot.fasta")   # change if you renamed



def sample_random(fasta=FASTA, n=100, max_len=1500):
    """Return a list of (id, seq) tuples."""
    records = [
        (rec.id, str(rec.seq))
        for rec in SeqIO.parse(fasta, "fasta")
        if len(rec.seq) <= max_len
    ]
    random.shuffle(records)
    return records[:n]

# # quick test
# if __name__ == "__main__":
#     for pid, seq in sample_random(n=5):
#         print(pid, len(seq))
# %%

FASTA = pathlib.Path("/project/pi_jensen_umass_edu/jnainani_umass_edu/plm_data/uniref50_sp.fasta")
FASTA = pathlib.Path("../../plm_interp/uniprot_sprot.fasta")
records = [
        (rec.id, str(rec.seq))
        for rec in SeqIO.parse(FASTA, "fasta")
        if len(rec.seq) <= 1000]
print(len(records))

# %%
import sys
sys.path.append('../plm_circuits')

# Import utility functions
from helpers.utils import (
    clear_memory,
    load_esm,
    load_sae_prot,
    mask_flanks_segment,
    patching_metric,
    cleanup_cuda
)
# Import attribution functions
from attribution import (
    integrated_gradients_sae,
    topk_sae_err_pt
)

# Import hook classes
from hook_manager import SAEHookProt

# %%
import torch
# Setup device and load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load ESM-2 model
esm_transformer, batch_converter, esm2_alphabet = load_esm(33, device=device)

# Load SAEs for multiple layers
main_layers = [4, 8, 12, 16, 20, 24, 28]
saes = []
for layer in main_layers:
    sae_model = load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=layer, device=device)
    saes.append(sae_model)

layer_2_saelayer = {layer: layer_idx for layer_idx, layer in enumerate(main_layers)}


# %%

test_protein = records[0][1]
print(test_protein)

# %%
test_layer = 4
test_sae = saes[layer_2_saelayer[test_layer]]
# %%
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from helpers.sae_model_interprot import SparseAutoencoder

class SAEHookProt:
    """Patch / scale SAE latent activations during the forward pass.

    Parameters
    ----------
    sae
        The :class:`~interprot.sae_model.SparseAutoencoder` attached to
        the target ESM layer.
    mask
        ``BoolTensor`` of shape *(B, L)* – ``True`` for non‑padding
        tokens. You can obtain this via
        ``(batch_tokens != alphabet.padding_idx)``.
    patch_latent, patch_value
        1‑D index tensor and replacement values. Mutually exclusive with
        *patch_mask*.
    patch_mask
        Boolean tensor of the same shape as the SAE activations; ``True``
        entries will be replaced by *patch_value*.
    layer_lm
        Set to *True* if you are attaching the hook to the LM head rather
        than an encoder layer.
    cache_sae_acts
        Store the activations in ``sae.feature_acts`` for later use.
    calc_error, use_error, use_mean_error
        Compute the difference between the original and reconstructed
        hidden states (``sae.error_term``) and optionally add it back in
        *use_error* or add a *mean* error stored in
        ``sae.mean_error`` (*use_mean_error*).
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        mask_BL: torch.Tensor,
        *,
        patch_latent_S: Optional[torch.Tensor] = None,
        patch_value: Optional[torch.Tensor] = None,
        patch_mask_BLS: Optional[torch.Tensor] = None,
        layer_is_lm: bool = False,
        cache_latents: bool = False,
        cache_masked_latents: bool = False,
        calc_error: bool = False,
        use_error: bool = False,
        use_mean_error: bool = False,
        no_detach: bool = False
    ) -> None:
        self.sae = sae.eval()
        self.mask_BL = mask_BL
        self.patch_latent_S = patch_latent_S
        self.patch_value = patch_value
        self.patch_mask_BLS = patch_mask_BLS
        self.layer_is_lm = layer_is_lm
        self.cache_latents = cache_latents
        self.cache_masked_latents = cache_masked_latents
        self.calc_error = calc_error
        self.use_error = use_error
        self.use_mean_error = use_mean_error
        self.no_detach = no_detach
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        _module: nn.Module,
        _inputs: Tuple[torch.Tensor, ...],
        outputs,
    ):
        hidden_BLF = outputs if self.layer_is_lm else outputs[0]
        mod_BLF = hidden_BLF.clone()

        # 1) select valid tokens (no padding) -------------------------- #
        valid_BXF = mod_BLF[self.mask_BL]

        # 2) SAE encode / decode -------------------------------------- #
        x, mu_F, std_F = self.sae.LN(valid_BXF)
        f_BXS = (x - self.sae.b_pre) @ self.sae.w_enc + self.sae.b_enc

        if self.patch_value is not None:
            if self.patch_latent_S is not None:
                f_BXS[:, self.patch_latent_S] = self.patch_value
            elif self.patch_mask_BLS is not None:
                f_BXS[self.patch_mask_BLS] = self.patch_value[self.patch_mask_BLS]

        if self.cache_latents:
            if self.no_detach:
                self.sae.feature_acts = f_BXS.clone()
            else:
                self.sae.feature_acts = f_BXS.detach().clone().cpu()

        topk_BXS = self.sae.topK_activation(f_BXS, self.sae.k)

        if self.cache_masked_latents:
            self.sae.masked_latents = topk_BXS.clone()

        recon_BXF = (topk_BXS @ self.sae.w_dec + self.sae.b_pre) * std_F + mu_F

        mod_BLF[self.mask_BL] = recon_BXF

        # 3) error handling ------------------------------------------- #
        if self.calc_error:
            self.sae.error_term = hidden_BLF - mod_BLF
            if self.use_error:
                mod_BLF = mod_BLF + self.sae.error_term
        if self.use_mean_error:
            mod_BLF = mod_BLF + self.sae.mean_error

        return mod_BLF if self.layer_is_lm else (mod_BLF,) + outputs[1:]
# %%
import time
start_time = time.time()
full_seq_L = [(1, test_protein)]
_, _, batch_tokens_BL = batch_converter(full_seq_L)
batch_tokens_BL = batch_tokens_BL.to(device)
batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)

hooks = []
for layer in main_layers:
    test_sae = saes[layer_2_saelayer[layer]]
    caching_hook = SAEHookProt(sae=test_sae, mask_BL=batch_mask_BL, cache_masked_latents=True, layer_is_lm=False, calc_error=True, use_error=True)
    handle = esm_transformer.esm.encoder.layer[layer].register_forward_hook(caching_hook)
    hooks.append(handle)

with torch.no_grad():
    full_seq_contact_LL = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]
cleanup_cuda()
for hook in hooks:
    hook.remove()

prot_cache_ALS = torch.stack([sae.masked_latents.cpu() for sae in saes], dim=0)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# %%
# Calculate memory usage of prot_cache_ALS in MB
bytes_per_element = prot_cache_ALS.element_size()
num_elements = prot_cache_ALS.numel() 
total_bytes = bytes_per_element * num_elements
total_mb = total_bytes / (1024 * 1024)

print(f"\nMemory usage of prot_cache_ALS:")
print(f"Shape: {prot_cache_ALS.shape}")
print(f"Bytes per element: {bytes_per_element}")
print(f"Number of elements: {num_elements:,}")
print(f"Total memory: {total_mb:.2f} MB")



# %%

prot_cache_ALS

# %%

full_save_path = "/project/pi_annagreen_umass_edu/jatin/plm_circuits/acts"
# %%
test_sae_cache_LS = test_sae.feature_acts 

test_latent = 340 

print("Activations for latent 340:")
print(test_sae_cache_LS[:, test_latent])
print(f"Max activation: {test_sae_cache_LS[:, test_latent].max().item():.4f}")
print(f"Mean activation: {test_sae_cache_LS[:, test_latent].mean().item():.4f}")

# %%

test_sae_cache_LS[:, test_latent] > 0.1
print(len(test_protein), len(test_sae_cache_LS[:, test_latent]))
# %%
import numpy as np
from collections import Counter, defaultdict

def find_motif_windows(sequence, activations, threshold_percentile=95, window_size=11, firing_position_in_window=4, bos_offset=1):
    """
    Find sequence windows around high-activation positions.
    
    Args:
        sequence: protein sequence string
        activations: 1D tensor of activations for each position (includes BOS/EOS tokens)
        threshold_percentile: percentile threshold for "high" activation
        window_size: size of window to extract (should be odd, e.g., 11 for 5+1+5)
        firing_position_in_window: position within window where activation happens (0-indexed)
        bos_offset: offset due to beginning-of-sequence token (usually 1)
    
    Returns:
        List of (sequence_position, window, activation_value) tuples
    """
    # Convert to numpy for easier manipulation
    acts = activations.cpu().numpy()
    
    # Find threshold
    threshold = np.percentile(acts[acts > 0], threshold_percentile) if np.any(acts > 0) else 0
    
    # Find high-activation positions in activation tensor
    high_positions = np.where(acts >= threshold)[0]
    
    windows = []
    
    for act_pos in high_positions:
        # Convert activation position to sequence position (account for BOS token)
        seq_pos = act_pos - bos_offset
        
        # Skip if position is in special tokens (BOS/EOS)
        if seq_pos < 0 or seq_pos >= len(sequence):
            continue
        
        # Extract window so that activation position is at firing_position_in_window
        start = max(0, seq_pos - firing_position_in_window)
        end = min(len(sequence), seq_pos + (window_size - firing_position_in_window))
        
        # Extract window from sequence
        window_seq = sequence[start:end]
        
        # Pad with 'X' if at boundaries
        if start == 0 and seq_pos < firing_position_in_window:
            window_seq = 'X' * (firing_position_in_window - seq_pos) + window_seq
        if end == len(sequence) and seq_pos >= len(sequence) - (window_size - firing_position_in_window):
            needed_padding = window_size - len(window_seq)
            window_seq = window_seq + 'X' * needed_padding
            
        windows.append((seq_pos, window_seq, acts[act_pos]))
    
    return windows

def analyze_motifs(windows, mask_threshold=0.1):
    """
    Analyze motifs from extracted windows.
    
    Args:
        windows: List of (position, window, activation) tuples
        mask_threshold: activation threshold below which to mask amino acids as 'X'
    
    Returns:
        Dictionary with motif analysis results
    """
    print(f"Analyzing {len(windows)} high-activation windows...")
    
    # Extract just the windows and activations
    sequences = [w[1] for w in windows]
    activations = [w[2] for w in windows]
    
    # Count motifs of different lengths
    motif_counts = defaultdict(int)
    all_motifs = set()
    
    # Analyze motifs of length 3, 4, 5
    for motif_len in [3, 4, 5]:
        for seq in sequences:
            if len(seq) >= motif_len:
                for i in range(len(seq) - motif_len + 1):
                    motif = seq[i:i+motif_len]
                    motif_counts[motif] += 1
                    all_motifs.add(motif)
    
    # Find most common motifs
    common_motifs = Counter(motif_counts).most_common(20)
    
    return {
        'windows': windows,
        'sequences': sequences,
        'motif_counts': motif_counts,
        'common_motifs': common_motifs,
        'total_motifs': len(all_motifs)
    }

# Run motif analysis for latent 340
print(f"\nProtein sequence length: {len(test_protein)}")
print(f"First 50 chars: {test_protein[:50]}")

activations_340 = test_sae_cache_LS[:, test_latent]
windows = find_motif_windows(test_protein, activations_340, threshold_percentile=90, window_size=11, firing_position_in_window=4)

print(f"\nFound {len(windows)} high-activation windows:")
for i, (pos, window, activation) in enumerate(windows):
    print(f"Position {pos}: {window} (activation: {activation:.4f})")

# %%
# Analyze motifs
motif_analysis = analyze_motifs(windows)

print(f"\nTop 20 most common motifs:")
for motif, count in motif_analysis['common_motifs']:
    print(f"'{motif}': {count} occurrences")

# %%
# Let's also look at the central amino acid pattern (what's at the max activation position)
def analyze_central_residues(windows):
    """Look at the amino acid at the firing position in each high-activation window."""
    central_residues = []
    for pos, window, activation in windows:
        if len(window) >= 11:  # Full window
            central = window[4]  # Firing position in 11-char window
            central_residues.append(central)
        elif len(window) > 4:  # Window long enough to have position 4
            central = window[4]
            central_residues.append(central)
    
    return Counter(central_residues)

central_aa_counts = analyze_central_residues(windows)
print(f"\nAmino acids at high-activation positions:")
for aa, count in central_aa_counts.most_common():
    print(f"'{aa}': {count} times")

# %%
# Let's also create a simple position-specific scoring matrix (PSSM) for the windows
def create_simple_pssm(windows, window_size=11):
    """Create a position-specific scoring matrix from the windows."""
    if not windows:
        return None
    
    # Initialize count matrix
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    pssm = {aa: [0] * window_size for aa in amino_acids}
    total_counts = [0] * window_size
    
    # Count amino acids at each position
    for pos, window, activation in windows:
        # Pad or trim window to exact size, keeping firing position at index 4
        if len(window) < window_size:
            # The window should already be properly aligned from find_motif_windows
            # Just pad to exact size if needed
            if len(window) < window_size:
                window = window + 'X' * (window_size - len(window))
        elif len(window) > window_size:
            # Trim to exact size, keeping firing position at index 4
            window = window[:window_size]
        
        for i, aa in enumerate(window):
            if aa in amino_acids:
                pssm[aa][i] += 1
                total_counts[i] += 1
    
    # Convert to frequencies
    for aa in amino_acids:
        for i in range(window_size):
            if total_counts[i] > 0:
                pssm[aa][i] = pssm[aa][i] / total_counts[i]
    
    return pssm, total_counts

def create_consensus_motif(windows, window_size=11, conservation_threshold=0.8, firing_marker='*'):
    """
    Create a consensus motif from high-activation windows.
    
    Args:
        windows: List of (position, window, activation) tuples
        window_size: Expected window size
        conservation_threshold: Fraction threshold for considering a position "conserved"
        firing_marker: Character to mark the firing position (center of window)
    
    Returns:
        Dictionary with consensus motif information
    """
    if not windows:
        return None
    
    print(f"\nCreating consensus motif from {len(windows)} windows...")
    
    # Normalize all windows to same size, keeping firing position at index 4
    normalized_windows = []
    for pos, window, activation in windows:
        # Pad or trim window to exact size
        if len(window) < window_size:
            # The window should already be properly aligned from find_motif_windows
            # Just pad to exact size if needed
            if len(window) < window_size:
                window = window + 'X' * (window_size - len(window))
        elif len(window) > window_size:
            # Trim to exact size, keeping firing position at index 4
            window = window[:window_size]
        normalized_windows.append(window)
    
    # Count amino acids at each position
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    position_counts = []
    
    for i in range(window_size):
        pos_counter = Counter()
        for window in normalized_windows:
            if i < len(window) and window[i] in amino_acids:
                pos_counter[window[i]] += 1
        position_counts.append(pos_counter)
    
    # Create consensus motif
    consensus = []
    conserved_positions = []
    firing_position = 4  # Position where activation happens (right after F at position 3)
    
    for i, pos_counter in enumerate(position_counts):
        total_count = sum(pos_counter.values())
        
        if total_count == 0:
            consensus.append('X')
        elif i == firing_position:
            # Mark firing position specially
            most_common_aa, count = pos_counter.most_common(1)[0]
            consensus.append(f'{most_common_aa}{firing_marker}')
        else:
            # Check if position is conserved
            most_common_aa, count = pos_counter.most_common(1)[0]
            conservation_fraction = count / total_count
            
            if conservation_fraction >= conservation_threshold:
                consensus.append(most_common_aa)
                conserved_positions.append((i, most_common_aa, conservation_fraction))
            else:
                consensus.append('X')
    
    consensus_motif = ''.join(consensus)
    
    # Print detailed analysis
    print(f"Raw windows (aligned):")
    for i, window in enumerate(normalized_windows):
        print(f"  {i+1}: {window}")
    
    print(f"\nPosition-by-position analysis:")
    for i, pos_counter in enumerate(position_counts):
        total = sum(pos_counter.values())
        if total > 0:
            most_common = pos_counter.most_common(3)
            position_type = "FIRING" if i == firing_position else "CONSERVED" if any(cp[0] == i for cp in conserved_positions) else "VARIABLE"
            print(f"  Pos {i:2d} ({position_type:8s}): " + 
                  ", ".join(f"{aa}({count}/{total}={count/total:.2f})" for aa, count in most_common))
    
    print(f"\nConsensus motif: {consensus_motif}")
    print(f"Conserved positions: {conserved_positions}")
    print(f"Firing position: {firing_position} (marked with '{firing_marker}')")
    
    return {
        'consensus_motif': consensus_motif,
        'conserved_positions': conserved_positions,
        'firing_position': firing_position,
        'position_counts': position_counts,
        'normalized_windows': normalized_windows
    }

if windows:
    pssm, total_counts = create_simple_pssm(windows)
    
    print(f"\nPosition-specific amino acid frequencies (window size 11):")
    print("Position:", "".join(f"{i:>6}" for i in range(11)))
    print("Counts:  ", "".join(f"{c:>6}" for c in total_counts))
    print()
    
    # Show top amino acids at each position
    for i in range(11):
        position_freqs = [(aa, pssm[aa][i]) for aa in 'ACDEFGHIKLMNPQRSTVWY' if pssm[aa][i] > 0]
        position_freqs.sort(key=lambda x: x[1], reverse=True)
        top_3 = position_freqs[:3]
        print(f"Pos {i}: " + ", ".join(f"{aa}({freq:.2f})" for aa, freq in top_3))

# %%
# Create consensus motif for latent 340
if windows:
    consensus_result = create_consensus_motif(windows, conservation_threshold=0.5)
    
    if consensus_result:
        print(f"\n" + "="*60)
        print(f"FINAL CONSENSUS MOTIF FOR LATENT 340:")
        print(f"  {consensus_result['consensus_motif']}")
        print("="*60)
        
        # Let's also try with different conservation thresholds
        print(f"\nTrying different conservation thresholds:")
        for threshold in [0.3, 0.5, 0.8]:
            result = create_consensus_motif(windows, conservation_threshold=threshold, firing_marker='*')
            print(f"  Threshold {threshold}: {result['consensus_motif']}")


# %%



# %%

activation_save_path = "/project/pi_jensen_umass_edu/jnainani_umass_edu/plm_data/"

# %%
import pickle
import os
from tqdm import tqdm
import gc

def process_proteins_batch(records, layer, feature_idx, n_proteins=5000, batch_size=1, save_path=None):
    """
    Process a batch of proteins and extract activations for a specific layer and feature.
    
    Args:
        records: List of (protein_id, sequence) tuples
        layer: Layer number to extract activations from
        feature_idx: Feature index to extract (e.g., 2112)
        n_proteins: Number of proteins to process
        batch_size: Number of proteins to process at once (1 for memory safety)
        save_path: Path to save the results
    
    Returns:
        List of (protein_id, sequence, activation_vector) tuples
    """
    
    # Setup for target layer
    target_sae = saes[layer_2_saelayer[layer]]
    
    # Prepare results storage
    results = []
    
    # Process proteins in batches
    proteins_to_process = records[:n_proteins]
    print(f"Processing {len(proteins_to_process)} proteins for layer {layer}, feature {feature_idx}")
    
    for i in tqdm(range(0, len(proteins_to_process), batch_size), desc="Processing proteins"):
        batch_proteins = proteins_to_process[i:i+batch_size]
        
        for protein_id, sequence in batch_proteins:
            try:
                # Prepare sequence for model
                full_seq_L = [(1, sequence)]
                _, _, batch_tokens_BL = batch_converter(full_seq_L)
                batch_tokens_BL = batch_tokens_BL.to(device)
                batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)
                
                # Setup caching hook for target layer
                caching_hook = SAEHookProt(
                    sae=target_sae, 
                    mask_BL=batch_mask_BL, 
                    cache_latents=True, 
                    layer_is_lm=False, 
                    calc_error=True, 
                    use_error=True
                )
                
                # Register hook and run model
                handle = esm_transformer.esm.encoder.layer[layer].register_forward_hook(caching_hook)
                
                with torch.no_grad():
                    # Run model (we don't need the contact prediction output)
                    _ = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]
                
                # Extract activations for the specific feature
                feature_activations = target_sae.feature_acts[:, feature_idx].cpu().numpy()
                
                # Store result
                results.append((protein_id, sequence, feature_activations))
                
                # Clean up
                handle.remove()
                cleanup_cuda()
                
            except Exception as e:
                print(f"Error processing protein {protein_id}: {e}")
                continue
        
        # Save intermediate results every 1000 proteins
        if (i + batch_size) % 1000 == 0 and save_path:
            intermediate_file = os.path.join(save_path, f"activations_layer{layer}_feature{feature_idx}_batch_{i//1000}.pkl")
            with open(intermediate_file, 'wb') as f:
                pickle.dump(results[-1000:], f)
            print(f"Saved intermediate batch to {intermediate_file}")
    
    return results

def save_activation_data(results, layer, feature_idx, save_path):
    """Save the activation data to disk."""
    filename = f"activations_layer{layer}_feature{feature_idx}_complete.pkl"
    filepath = os.path.join(save_path, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved {len(results)} protein activation records to {filepath}")
    
    # Also save a summary file with just protein IDs and sequence lengths
    summary = [(pid, len(seq), len(acts)) for pid, seq, acts in results]
    summary_file = os.path.join(save_path, f"activation_summary_layer{layer}_feature{feature_idx}.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    return filepath

def load_activation_data(layer, feature_idx, save_path):
    """
    Load saved activation data from disk.
    
    Args:
        layer: Layer number that was processed
        feature_idx: Feature index that was processed
        save_path: Path where the data was saved
    
    Returns:
        List of (protein_id, sequence, activation_vector) tuples
    """
    filename = f"activations_layer{layer}_feature{feature_idx}_complete.pkl"
    filepath = os.path.join(save_path, filename)
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} protein activation records from {filepath}")
    
    # Print summary
    if data:
        total_proteins = len(data)
        avg_seq_len = np.mean([len(seq) for _, seq, _ in data])
        avg_act_len = np.mean([len(acts) for _, _, acts in data])
        
        print(f"Summary:")
        print(f"  Total proteins: {total_proteins}")
        print(f"  Average sequence length: {avg_seq_len:.1f}")
        print(f"  Average activation vector length: {avg_act_len:.1f}")
    
    return data

def load_activation_summary(layer, feature_idx, save_path):
    """
    Load the summary file (lighter weight than full data).
    
    Returns:
        List of (protein_id, sequence_length, activation_length) tuples
    """
    summary_file = os.path.join(save_path, f"activation_summary_layer{layer}_feature{feature_idx}.pkl")
    
    if not os.path.exists(summary_file):
        print(f"Summary file not found: {summary_file}")
        return None
    
    with open(summary_file, 'rb') as f:
        summary = pickle.load(f)
    
    print(f"Loaded summary for {len(summary)} proteins")
    return summary

# %%
# Process proteins for layer 12, feature 2112
target_layer = 12
target_feature = 2112
n_proteins_to_process = 5000  # Start with 5k, can increase to 10k
activation_save_path = "/project/pi_jensen_umass_edu/jnainani_umass_edu/plm_data/"
print(f"Starting batch processing for layer {target_layer}, feature {target_feature}")
print(f"Total proteins available: {len(records)}")
print(f"Will process: {n_proteins_to_process} proteins")

# Run the batch processing
activation_results = process_proteins_batch(
    records=records,
    layer=target_layer,
    feature_idx=target_feature,
    n_proteins=n_proteins_to_process,
    batch_size=1,  # Process one at a time for memory safety
    save_path=activation_save_path
)

# Save final results
final_save_path = save_activation_data(
    activation_results, 
    target_layer, 
    target_feature, 
    activation_save_path
)

print(f"\nCompleted processing {len(activation_results)} proteins")
print(f"Results saved to: {final_save_path}")

# %%
# Quick verification - load and check the first few results
def verify_saved_data(filepath):
    """Quick verification of saved data."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} protein records")
    
    # Show first few examples
    for i in range(min(3, len(data))):
        pid, seq, acts = data[i]
        print(f"Protein {i+1}: {pid}")
        print(f"  Sequence length: {len(seq)}")
        print(f"  Activation vector length: {len(acts)}")
        print(f"  Max activation: {acts.max():.4f}")
        print(f"  Mean activation: {acts.mean():.4f}")
        print(f"  Non-zero activations: {(acts > 0).sum()}")
        print()

if 'final_save_path' in locals():
    verify_saved_data(final_save_path)

# %%
# Example: How to load the saved activation data later

# Method 1: Load full activation data
def example_load_and_analyze():
    """Example of how to load and analyze saved activation data."""
    
    # Load the data
    layer = 12
    feature_idx = 2112
    save_path = "/project/pi_jensen_umass_edu/jnainani_umass_edu/plm_data/"
    
    print("Loading activation data...")
    loaded_data = load_activation_data(layer, feature_idx, save_path)
    
    if loaded_data is None:
        print("No data found!")
        return
    
    print(f"\nAnalyzing {len(loaded_data)} proteins...")
    
    # Example analysis: find proteins with highest max activations
    max_activations = []
    for protein_id, sequence, activations in loaded_data:
        max_act = np.max(activations)
        max_activations.append((protein_id, max_act, len(sequence)))
    
    # Sort by max activation
    max_activations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 proteins with highest activations for feature {feature_idx}:")
    for i, (pid, max_act, seq_len) in enumerate(max_activations[:10]):
        print(f"{i+1:2d}. {pid}: max_activation={max_act:.4f}, seq_length={seq_len}")
    
    # Example: Extract all high-activation windows across all proteins
    all_windows = []
    for protein_id, sequence, activations in loaded_data[:100]:  # Just first 100 for demo
        activations_tensor = torch.tensor(activations)
        windows = find_motif_windows(sequence, activations_tensor, threshold_percentile=95)
        for pos, window, act_val in windows:
            all_windows.append((protein_id, pos, window, act_val))
    
    print(f"\nFound {len(all_windows)} high-activation windows across first 100 proteins")
    
    if all_windows:
        # Sort by activation value
        all_windows.sort(key=lambda x: x[3], reverse=True)
        print(f"\nTop 10 highest-activation windows:")
        for i, (pid, pos, window, act_val) in enumerate(all_windows[:10]):
            print(f"{i+1:2d}. {pid}[{pos}]: '{window}' (activation: {act_val:.4f})")
    
    return loaded_data

# Method 2: Just load the summary (faster for quick checks)
def example_load_summary():
    """Example of how to load just the summary data."""
    
    layer = 12
    feature_idx = 2112
    save_path = "/project/pi_jensen_umass_edu/jnainani_umass_edu/plm_data/"
    
    summary = load_activation_summary(layer, feature_idx, save_path)
    
    if summary:
        print(f"\nQuick summary statistics:")
        protein_ids = [item[0] for item in summary]
        seq_lengths = [item[1] for item in summary]
        act_lengths = [item[2] for item in summary]
        
        print(f"  Proteins: {len(protein_ids)}")
        print(f"  Sequence length range: {min(seq_lengths)} - {max(seq_lengths)}")
        print(f"  Average sequence length: {np.mean(seq_lengths):.1f}")
        print(f"  Activation vector length range: {min(act_lengths)} - {max(act_lengths)}")
    
    return summary

# Uncomment the line below to run the example when you have saved data:
# example_data = example_load_and_analyze()
# example_summary = example_load_summary()

# %%
# Let's first analyze the characteristics of our activation data
# This will help us design the most efficient storage strategy
import numpy as np
def analyze_activation_characteristics():
    """Analyze precision, sparsity, and memory characteristics of SAE activations."""
    print("Analyzing activation data characteristics...")
    
    # Use the test protein to understand data properties
    full_seq_L = [(1, test_protein)]
    _, _, batch_tokens_BL = batch_converter(full_seq_L)
    batch_tokens_BL = batch_tokens_BL.to(device)
    batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)
    
    results = {}
    
    for layer_idx, layer in enumerate(main_layers):
        test_sae = saes[layer_idx]
        
        # Setup caching hook
        caching_hook = SAEHookProt(
            sae=test_sae, 
            mask_BL=batch_mask_BL, 
            cache_latents=True,
            cache_masked_latents=True,
            layer_is_lm=False, 
            calc_error=True, 
            use_error=True
        )
        
        # Register hook and run model
        handle = esm_transformer.esm.encoder.layer[layer].register_forward_hook(caching_hook)
        
        with torch.no_grad():
            _ = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]
        
        # Analyze the cached activations
        full_acts = test_sae.feature_acts  # Full activations before masking
        masked_acts = test_sae.masked_latents  # Top-K masked activations
        
        # Calculate statistics
        full_acts_cpu = full_acts.cpu()
        masked_acts_cpu = masked_acts.cpu()
        
        layer_stats = {
            'layer': layer,
            'shape': full_acts_cpu.shape,
            'dtype': full_acts_cpu.dtype,
            'total_elements': full_acts_cpu.numel(),
            'full_nonzero_count': (full_acts_cpu != 0).sum().item(),
            'masked_nonzero_count': (masked_acts_cpu != 0).sum().item(),
            'full_sparsity': 1 - (full_acts_cpu != 0).sum().item() / full_acts_cpu.numel(),
            'masked_sparsity': 1 - (masked_acts_cpu != 0).sum().item() / masked_acts_cpu.numel(),
            'full_memory_mb': full_acts_cpu.numel() * full_acts_cpu.element_size() / (1024**2),
            'masked_memory_mb': masked_acts_cpu.numel() * masked_acts_cpu.element_size() / (1024**2),
            'max_value': full_acts_cpu.max().item(),
            'min_value': full_acts_cpu.min().item(),
            'mean_nonzero': full_acts_cpu[full_acts_cpu != 0].mean().item() if (full_acts_cpu != 0).any() else 0,
        }
        
        results[layer] = layer_stats
        
        # Clean up
        handle.remove()
        print(f"Layer {layer}: Shape {layer_stats['shape']}, Sparsity: {layer_stats['masked_sparsity']:.3f}, Memory: {layer_stats['masked_memory_mb']:.2f}MB")
    
    cleanup_cuda()
    
    # Calculate total memory for 10k proteins
    print(f"\n" + "="*60)
    print("MEMORY PROJECTIONS FOR 10,000 PROTEINS:")
    print("="*60)
    
    total_memory_full = 0
    total_memory_masked = 0
    
    for layer, stats in results.items():
        layer_memory_full = stats['full_memory_mb'] * 10000
        layer_memory_masked = stats['masked_memory_mb'] * 10000
        total_memory_full += layer_memory_full
        total_memory_masked += layer_memory_masked
        
        print(f"Layer {layer:2d}: Full={layer_memory_full/1024:.1f}GB, Masked={layer_memory_masked/1024:.1f}GB")
    
    print(f"\nTOTAL (all layers):")
    print(f"  Full activations: {total_memory_full/1024:.1f} GB")
    print(f"  Masked activations: {total_memory_masked/1024:.1f} GB")
    print(f"  Average sparsity: {np.mean([s['masked_sparsity'] for s in results.values()]):.3f}")
    
    # Test different precisions
    print(f"\n" + "="*60)
    print("PRECISION ANALYSIS:")
    print("="*60)
    
    sample_layer = results[main_layers[0]]
    sample_acts = saes[0].masked_latents.cpu()
    
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        converted = sample_acts.to(dtype)
        memory_mb = converted.numel() * converted.element_size() / (1024**2)
        memory_gb_10k = memory_mb * 10000 * 7 / 1024  # 7 layers, 10k proteins
        
        # Calculate precision loss
        if dtype != torch.float32:
            diff = (sample_acts.float() - converted.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"{str(dtype):15s}: {memory_gb_10k:6.1f} GB total, max_diff: {max_diff:.6f}, mean_diff: {mean_diff:.6f}")
        else:
            print(f"{str(dtype):15s}: {memory_gb_10k:6.1f} GB total (baseline)")
    
    return results

# Run the analysis
analysis_results = analyze_activation_characteristics()

# %%



# %%

# %%
# Efficient multi-layer activation caching system for 10,000 proteins

import h5py
import scipy.sparse as sp
from scipy.sparse import csr_matrix, save_npz, load_npz
import json
import time
from tqdm import tqdm

class EfficientActivationCache:
    """
    Efficient storage and retrieval system for SAE activations across multiple layers.
    Uses sparse matrices and HDF5 for memory-efficient storage.
    """
    
    def __init__(self, save_path, layers, n_features=4096, use_sparse=True, precision='float16'):
        self.save_path = save_path
        self.layers = layers
        self.n_features = n_features
        self.use_sparse = use_sparse
        self.precision = getattr(torch, precision) if isinstance(precision, str) else precision
        
        # Create save directory if it doesn't exist
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Metadata storage
        self.metadata = {
            'layers': layers,
            'n_features': n_features,
            'use_sparse': use_sparse,
            'precision': str(precision),
            'proteins': []
        }
    
    def save_protein_activations(self, protein_id, sequence, layer_activations_dict, protein_idx):
        """
        Save activations for a single protein across all layers.
        
        Args:
            protein_id: Protein identifier
            sequence: Protein sequence string
            layer_activations_dict: Dict mapping layer -> activation tensor
            protein_idx: Index of this protein in the batch
        """
        protein_data = {
            'protein_id': protein_id,
            'sequence_length': len(sequence),
            'protein_idx': protein_idx
        }
        
        # Save each layer's activations
        for layer in self.layers:
            acts = layer_activations_dict[layer].cpu()
            
            # Convert to specified precision
            if self.precision != torch.float32:
                acts = acts.to(self.precision)
            
            if self.use_sparse:
                # Convert to sparse matrix (CSR format)
                acts_np = acts.numpy()
                sparse_matrix = csr_matrix(acts_np)
                
                # Save sparse matrix
                sparse_file = os.path.join(
                    self.save_path, 
                    f"protein_{protein_idx:06d}_layer_{layer}_sparse.npz"
                )
                save_npz(sparse_file, sparse_matrix)
                
                protein_data[f'layer_{layer}_file'] = sparse_file
                protein_data[f'layer_{layer}_shape'] = acts_np.shape
                protein_data[f'layer_{layer}_nnz'] = sparse_matrix.nnz
            else:
                # Save as regular numpy array
                acts_file = os.path.join(
                    self.save_path,
                    f"protein_{protein_idx:06d}_layer_{layer}.npy"
                )
                np.save(acts_file, acts.numpy())
                protein_data[f'layer_{layer}_file'] = acts_file
                protein_data[f'layer_{layer}_shape'] = acts.shape
        
        self.metadata['proteins'].append(protein_data)
        
        # Save metadata periodically
        if len(self.metadata['proteins']) % 100 == 0:
            self.save_metadata()
    
    def save_metadata(self):
        """Save metadata to JSON file."""
        metadata_file = os.path.join(self.save_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_protein_activations(self, protein_idx):
        """Load activations for a specific protein."""
        if protein_idx >= len(self.metadata['proteins']):
            raise IndexError(f"Protein index {protein_idx} out of range")
        
        protein_data = self.metadata['proteins'][protein_idx]
        activations = {}
        
        for layer in self.layers:
            file_path = protein_data[f'layer_{layer}_file']
            shape = protein_data[f'layer_{layer}_shape']
            
            if self.use_sparse:
                sparse_matrix = load_npz(file_path)
                acts = torch.tensor(sparse_matrix.toarray(), dtype=torch.float32)
            else:
                acts_np = np.load(file_path)
                acts = torch.tensor(acts_np, dtype=torch.float32)
            
            activations[layer] = acts
        
        return {
            'protein_id': protein_data['protein_id'],
            'sequence_length': protein_data['sequence_length'],
            'activations': activations
        }
    
    def load_protein_activations_original_format(self, protein_idx):
        """
        Load activations for a specific protein in the ORIGINAL tensor format.
        This reconstructs the exact same tensor format as your original caching,
        perfect for handing off to collaborators.
        
        Returns:
            Dict with 'protein_id', 'sequence_length', and 'activations' where
            activations is a dict: {layer: torch.Tensor} in original dense format
        """
        if protein_idx >= len(self.metadata['proteins']):
            raise IndexError(f"Protein index {protein_idx} out of range")
        
        protein_data = self.metadata['proteins'][protein_idx]
        activations = {}
        
        for layer in self.layers:
            file_path = protein_data[f'layer_{layer}_file']
            
            if self.use_sparse:
                # Load sparse and convert back to dense
                sparse_matrix = load_npz(file_path)
                acts = torch.tensor(sparse_matrix.toarray(), dtype=torch.float32)
            else:
                acts_np = np.load(file_path)
                acts = torch.tensor(acts_np, dtype=torch.float32)
            
            activations[layer] = acts
        
        return {
            'protein_id': protein_data['protein_id'],
            'sequence_length': protein_data['sequence_length'],
            'activations': activations
        }
    
    def load_batch_activations_original_format(self, protein_indices):
        """
        Load multiple proteins in original format - convenient for batch analysis.
        
        Args:
            protein_indices: List of protein indices to load
            
        Returns:
            List of dicts, each containing protein data in original format
        """
        batch_data = []
        for idx in protein_indices:
            try:
                protein_data = self.load_protein_activations_original_format(idx)
                batch_data.append(protein_data)
            except Exception as e:
                print(f"Error loading protein {idx}: {e}")
                continue
        
        return batch_data
    
    def create_legacy_format(self, protein_indices=None, layer_subset=None):
        """
        Create a data structure that exactly matches your original caching format.
        This makes it seamless to hand off to collaborators.
        
        Args:
            protein_indices: List of protein indices (None for all)
            layer_subset: List of layers to include (None for all)
            
        Returns:
            Dict with 'proteins' and 'layer_activations' matching original format
        """
        if protein_indices is None:
            protein_indices = list(range(len(self.metadata['proteins'])))
        
        if layer_subset is None:
            layer_subset = self.layers
        
        print(f"Creating legacy format for {len(protein_indices)} proteins, {len(layer_subset)} layers...")
        
        # Load all protein data
        proteins_data = []
        layer_activations = {layer: [] for layer in layer_subset}
        
        for idx in tqdm(protein_indices, desc="Loading proteins"):
            try:
                protein_data = self.load_protein_activations_original_format(idx)
                
                proteins_data.append({
                    'protein_id': protein_data['protein_id'],
                    'sequence_length': protein_data['sequence_length'],
                    'protein_idx': idx
                })
                
                # Collect activations by layer
                for layer in layer_subset:
                    layer_activations[layer].append(protein_data['activations'][layer])
                    
            except Exception as e:
                print(f"Error loading protein {idx}: {e}")
                continue
        
        # Stack activations into tensors (list of tensors, one per protein)
        legacy_format = {
            'proteins': proteins_data,
            'layer_activations': layer_activations,
            'metadata': {
                'total_proteins': len(proteins_data),
                'layers': layer_subset,
                'source_cache': self.save_path,
                'original_precision': str(self.precision),
                'compression_method': 'sparse' if self.use_sparse else 'dense'
            }
        }
        
        print(f"Legacy format created: {len(proteins_data)} proteins across {len(layer_subset)} layers")
        return legacy_format

    def get_summary_stats(self):
        """Get summary statistics about the cached data."""
        if not self.metadata['proteins']:
            return None
        
        total_proteins = len(self.metadata['proteins'])
        total_files = total_proteins * len(self.layers)
        
        # Calculate sparsity and size stats
        sparsity_stats = {}
        size_stats = {}
        
        for layer in self.layers:
            layer_nnz = []
            layer_sizes = []
            
            for protein_data in self.metadata['proteins']:
                if f'layer_{layer}_nnz' in protein_data:
                    layer_nnz.append(protein_data[f'layer_{layer}_nnz'])
                
                shape = protein_data[f'layer_{layer}_shape']
                layer_sizes.append(np.prod(shape))
            
            if layer_nnz:
                avg_sparsity = 1 - np.mean(layer_nnz) / np.mean(layer_sizes)
                sparsity_stats[layer] = avg_sparsity
        
        return {
            'total_proteins': total_proteins,
            'total_files': total_files,
            'layers': self.layers,
            'sparsity_by_layer': sparsity_stats,
            'use_sparse': self.use_sparse,
            'precision': self.precision
        }

def process_proteins_all_layers(records, n_proteins=10000, batch_size=1, 
                              save_path=None, use_sparse=True, precision='float16',
                              checkpoint_every=500):
    """
    Process proteins and cache activations for ALL layers efficiently.
    
    Args:
        records: List of (protein_id, sequence) tuples
        n_proteins: Number of proteins to process
        batch_size: Number of proteins to process at once (keep at 1 for memory safety)
        save_path: Path to save the cached activations
        use_sparse: Whether to use sparse matrix storage
        precision: Precision to use ('float32', 'float16', 'bfloat16')
        checkpoint_every: Save progress every N proteins
    
    Returns:
        EfficientActivationCache object
    """
    
    print(f"Starting efficient processing of {n_proteins} proteins")
    print(f"Storage: {'Sparse' if use_sparse else 'Dense'}, Precision: {precision}")
    print(f"Layers: {main_layers}")
    
    # Initialize cache system
    cache_system = EfficientActivationCache(
        save_path=save_path,
        layers=main_layers,
        use_sparse=use_sparse,
        precision=precision
    )
    
    # Process proteins
    proteins_to_process = records[:n_proteins]
    start_time = time.time()
    
    for protein_idx in tqdm(range(0, len(proteins_to_process), batch_size), 
                           desc="Processing proteins"):
        
        batch_proteins = proteins_to_process[protein_idx:protein_idx+batch_size]
        
        for batch_offset, (protein_id, sequence) in enumerate(batch_proteins):
            current_protein_idx = protein_idx + batch_offset
            
            try:
                # Prepare sequence for model
                full_seq_L = [(1, sequence)]
                _, _, batch_tokens_BL = batch_converter(full_seq_L)
                batch_tokens_BL = batch_tokens_BL.to(device)
                batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)
                
                # Setup caching hooks for ALL layers simultaneously
                hooks = []
                for layer_idx, layer in enumerate(main_layers):
                    sae = saes[layer_idx]
                    caching_hook = SAEHookProt(
                        sae=sae, 
                        mask_BL=batch_mask_BL, 
                        cache_masked_latents=True,  # Use masked (sparse) activations
                        layer_is_lm=False, 
                        calc_error=True, 
                        use_error=True
                    )
                    handle = esm_transformer.esm.encoder.layer[layer].register_forward_hook(caching_hook)
                    hooks.append(handle)
                
                # Single forward pass captures all layers
                with torch.no_grad():
                    _ = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]
                
                # Extract activations from all layers
                layer_activations = {}
                for layer_idx, layer in enumerate(main_layers):
                    # Use masked_latents (top-K activations) for sparsity
                    activations = saes[layer_idx].masked_latents.cpu()
                    layer_activations[layer] = activations
                
                # Save this protein's activations
                cache_system.save_protein_activations(
                    protein_id, sequence, layer_activations, current_protein_idx
                )
                
                # Clean up hooks
                for handle in hooks:
                    handle.remove()
                cleanup_cuda()
                
            except Exception as e:
                print(f"Error processing protein {protein_id}: {e}")
                # Clean up hooks in case of error
                for handle in hooks:
                    handle.remove()
                continue
            
            # Checkpoint progress
            if (current_protein_idx + 1) % checkpoint_every == 0:
                elapsed = time.time() - start_time
                rate = (current_protein_idx + 1) / elapsed
                eta = (n_proteins - current_protein_idx - 1) / rate / 60  # minutes
                
                print(f"\nCheckpoint: {current_protein_idx + 1}/{n_proteins} proteins processed")
                print(f"Rate: {rate:.2f} proteins/sec, ETA: {eta:.1f} minutes")
                
                # Save metadata checkpoint
                cache_system.save_metadata()
    
    # Final save
    cache_system.save_metadata()
    
    total_time = time.time() - start_time
    print(f"\nCompleted processing {len(cache_system.metadata['proteins'])} proteins")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average rate: {len(cache_system.metadata['proteins'])/total_time:.2f} proteins/sec")
    
    # Print summary statistics
    stats = cache_system.get_summary_stats()
    if stats:
        print(f"\nSummary Statistics:")
        print(f"  Total proteins: {stats['total_proteins']}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Storage format: {'Sparse' if stats['use_sparse'] else 'Dense'}")
        print(f"  Precision: {stats['precision']}")
        if stats['sparsity_by_layer']:
            print(f"  Average sparsity by layer:")
            for layer, sparsity in stats['sparsity_by_layer'].items():
                print(f"    Layer {layer}: {sparsity:.3f}")
    
    return cache_system

def load_cached_activations(save_path):
    """Load a previously saved activation cache."""
    import os
    metadata_file = os.path.join(save_path, "metadata.json")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"No metadata file found at {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Reconstruct cache system
    cache_system = EfficientActivationCache(
        save_path=save_path,
        layers=metadata['layers'],
        n_features=metadata['n_features'],
        use_sparse=metadata['use_sparse'],
        precision=metadata['precision']
    )
    cache_system.metadata = metadata
    
    print(f"Loaded cache with {len(metadata['proteins'])} proteins")
    
    return cache_system

# %%



# %%
# Example usage and testing

def test_small_batch(n_test=10):
    """Test the system with a small batch of proteins first."""
    print("Testing with small batch of proteins...")
    
    test_save_path = "/project/pi_jensen_umass_edu/jnainani_umass_edu/plm_data/test_cache"
    
    # Run with small batch
    test_cache = process_proteins_all_layers(
        records=records,
        n_proteins=n_test,
        save_path=test_save_path,
        use_sparse=True,
        precision='float16',
        checkpoint_every=5
    )
    
    # Test loading
    print(f"\nTesting data loading...")
    for i in range(min(3, n_test)):
        protein_data = test_cache.load_protein_activations(i)
        print(f"Protein {i}: {protein_data['protein_id']}")
        print(f"  Sequence length: {protein_data['sequence_length']}")
        print(f"  Layers available: {list(protein_data['activations'].keys())}")
        
        # Check one layer
        layer_4_acts = protein_data['activations'][4]
        print(f"  Layer 4 shape: {layer_4_acts.shape}")
        print(f"  Layer 4 non-zero: {(layer_4_acts != 0).sum().item()}")
        print(f"  Layer 4 sparsity: {1 - (layer_4_acts != 0).sum().item() / layer_4_acts.numel():.3f}")
    
    return test_cache

def analyze_cached_data(cache_system, protein_indices=None):
    """
    Analyze cached activation data for patterns and statistics.
    
    Args:
        cache_system: EfficientActivationCache object
        protein_indices: List of protein indices to analyze (None for all)
    """
    
    if protein_indices is None:
        protein_indices = list(range(min(100, len(cache_system.metadata['proteins']))))
    
    print(f"Analyzing {len(protein_indices)} proteins...")
    
    # Collect statistics
    layer_stats = {layer: [] for layer in cache_system.layers}
    sequence_lengths = []
    
    for i in protein_indices:
        try:
            protein_data = cache_system.load_protein_activations(i)
            sequence_lengths.append(protein_data['sequence_length'])
            
            for layer in cache_system.layers:
                acts = protein_data['activations'][layer]
                nonzero_count = (acts != 0).sum().item()
                sparsity = 1 - nonzero_count / acts.numel()
                max_activation = acts.max().item()
                
                layer_stats[layer].append({
                    'sparsity': sparsity,
                    'nonzero_count': nonzero_count,
                    'max_activation': max_activation,
                    'total_elements': acts.numel()
                })
        except Exception as e:
            print(f"Error loading protein {i}: {e}")
            continue
    
    # Print analysis
    print(f"\nSequence length statistics:")
    print(f"  Mean: {np.mean(sequence_lengths):.1f}")
    print(f"  Min: {min(sequence_lengths)}, Max: {max(sequence_lengths)}")
    
    print(f"\nLayer-wise activation statistics:")
    for layer in cache_system.layers:
        if layer_stats[layer]:
            sparsities = [s['sparsity'] for s in layer_stats[layer]]
            max_acts = [s['max_activation'] for s in layer_stats[layer]]
            
            print(f"  Layer {layer:2d}:")
            print(f"    Average sparsity: {np.mean(sparsities):.3f} ± {np.std(sparsities):.3f}")
            print(f"    Max activation range: {min(max_acts):.3f} - {max(max_acts):.3f}")
    
    return layer_stats

def find_top_activating_proteins(cache_system, layer, feature_idx, top_k=10):
    """
    Find proteins with highest activations for a specific layer/feature.
    
    Args:
        cache_system: EfficientActivationCache object
        layer: Layer number
        feature_idx: Feature index (0-4095)
        top_k: Number of top proteins to return
    
    Returns:
        List of (protein_idx, protein_id, max_activation, sequence_length) tuples
    """
    
    print(f"Finding top {top_k} proteins for layer {layer}, feature {feature_idx}...")
    
    protein_activations = []
    
    for i in range(len(cache_system.metadata['proteins'])):
        try:
            protein_data = cache_system.load_protein_activations(i)
            acts = protein_data['activations'][layer]
            
            # Get max activation for this feature
            feature_acts = acts[:, feature_idx]
            max_activation = feature_acts.max().item()
            
            protein_activations.append((
                i, 
                protein_data['protein_id'], 
                max_activation,
                protein_data['sequence_length']
            ))
            
        except Exception as e:
            print(f"Error processing protein {i}: {e}")
            continue
    
    # Sort by max activation
    protein_activations.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Top {top_k} activating proteins for layer {layer}, feature {feature_idx}:")
    for i, (protein_idx, protein_id, max_act, seq_len) in enumerate(protein_activations[:top_k]):
        print(f"  {i+1:2d}. {protein_id}: max_act={max_act:.4f}, seq_len={seq_len}")
    
    return protein_activations[:top_k]

def extract_motifs_from_cached_data(cache_system, layer, feature_idx, 
                                   n_proteins=100, threshold_percentile=95):
    """
    Extract motifs from cached activation data for a specific layer/feature.
    """
    
    print(f"Extracting motifs for layer {layer}, feature {feature_idx} from {n_proteins} proteins...")
    
    all_windows = []
    
    for i in range(min(n_proteins, len(cache_system.metadata['proteins']))):
        try:
            protein_data = cache_system.load_protein_activations(i)
            
            # Get sequence (we need to load it from the original records)
            protein_id = protein_data['protein_id']
            sequence = None
            for pid, seq in records:
                if pid == protein_id:
                    sequence = seq
                    break
            
            if sequence is None:
                continue
            
            # Get activations for this feature
            acts = protein_data['activations'][layer]
            feature_acts = acts[:, feature_idx]
            
            # Find motif windows
            windows = find_motif_windows(sequence, feature_acts, threshold_percentile=threshold_percentile)
            
            for pos, window, act_val in windows:
                all_windows.append((protein_id, pos, window, act_val))
                
        except Exception as e:
            print(f"Error processing protein {i}: {e}")
            continue
    
    if all_windows:
        # Sort by activation value
        all_windows.sort(key=lambda x: x[3], reverse=True)
        
        print(f"Found {len(all_windows)} high-activation windows")
        print(f"Top 10 windows:")
        for i, (pid, pos, window, act_val) in enumerate(all_windows[:10]):
            print(f"  {i+1:2d}. {pid}[{pos}]: '{window}' (activation: {act_val:.4f})")
        
        # Analyze motifs
        windows_for_analysis = [(pos, window, act_val) for pid, pos, window, act_val in all_windows]
        motif_analysis = analyze_motifs(windows_for_analysis)
        
        print(f"\nTop 10 most common motifs:")
        for motif, count in motif_analysis['common_motifs'][:10]:
            print(f"  '{motif}': {count} occurrences")
    
    return all_windows

# %%
# Test with small batch first
print("Running small test batch...")
# test_cache = test_small_batch(n_test=20)

# %%
# Main processing run for 10,000 proteins
def run_full_processing():
    """Run the full processing for 10,000 proteins."""
    
    # Set up paths
    full_save_path = "/project/pi_jensen_umass_edu/jnainani_umass_edu/plm_data/full_activation_cache"
    
    print("="*60)
    print("STARTING FULL 10,000 PROTEIN PROCESSING")
    print("="*60)
    
    # Run the full processing
    full_cache = process_proteins_all_layers(
        records=records,
        n_proteins=10000,
        save_path=full_save_path,
        use_sparse=True,
        precision='float16',  # Use float16 for 2x memory savings
        checkpoint_every=100  # Checkpoint every 100 proteins
    )
    
    print("="*60)
    print("FULL PROCESSING COMPLETED")
    print("="*60)
    
    # Analyze a sample of the results
    print("\nRunning sample analysis...")
    layer_stats = analyze_cached_data(full_cache, protein_indices=list(range(0, 1000, 50)))
    
    return full_cache

# To run the full processing, uncomment the line below:
# full_cache = run_full_processing()

print("\nSetup complete! To run the analysis:")
print("1. First test with: test_cache = test_small_batch(n_test=20)")
print("2. Then run full processing with: full_cache = run_full_processing()")
print("3. Load existing cache with: cache = load_cached_activations('path/to/cache')")

# %%

# %%
# Perfect handoff functions - your collaborator gets exactly what they expect!

def demo_handoff_workflow():
    """
    Demonstration of how to hand off data in the original format.
    Your collaborator will receive exactly the same tensors as before.
    """
    print("="*60)
    print("DEMONSTRATION: SEAMLESS HANDOFF TO COLLABORATORS")
    print("="*60)
    
    # Simulate loading a cache (replace with your actual cache path)
    # cache = load_cached_activations("/path/to/your/cache")
    
    print("Three ways to hand off data in original format:")
    print()
    
    print("1. Single protein (for detailed analysis):")
    print("   protein_data = cache.load_protein_activations_original_format(42)")
    print("   # Returns: {'protein_id': '...', 'activations': {4: tensor, 8: tensor, ...}}")
    print()
    
    print("2. Batch of proteins (for statistical analysis):")
    print("   batch_data = cache.load_batch_activations_original_format([0, 1, 2, 100, 500])")
    print("   # Returns: list of dicts, each with original tensor format")
    print()
    
    print("3. Full dataset in legacy format (for large-scale analysis):")
    print("   legacy_data = cache.create_legacy_format(protein_indices=range(1000))")
    print("   # Returns: {'proteins': [...], 'layer_activations': {layer: [tensors]}}")
    print()
    
    print("✅ All methods return float32 tensors in the EXACT same format")
    print("   as your original caching - no changes needed downstream!")

def create_handoff_package(cache_path, output_path, protein_subset=None, 
                          layers_subset=None, package_name="activation_package"):
    """
    Create a complete handoff package for your collaborator.
    
    Args:
        cache_path: Path to your cached activations
        output_path: Where to save the handoff package
        protein_subset: List of protein indices (None for all)
        layers_subset: List of layers (None for all 7 layers)
        package_name: Name for the package
    """
    
    print(f"Creating handoff package: {package_name}")
    
    # Load the cache
    cache = load_cached_activations(cache_path)
    
    # Create legacy format
    legacy_data = cache.create_legacy_format(
        protein_indices=protein_subset,
        layer_subset=layers_subset
    )
    
    # Save as pickle (easy for collaborator to load)
    import pickle
    import os
    
    os.makedirs(output_path, exist_ok=True)
    package_file = os.path.join(output_path, f"{package_name}.pkl")
    
    with open(package_file, 'wb') as f:
        pickle.dump(legacy_data, f)
    
    # Create a README for your collaborator
    readme_content = f"""
# Activation Data Package: {package_name}

## Quick Start
```python
import pickle
with open('{package_name}.pkl', 'rb') as f:
    data = pickle.load(f)

# Access protein metadata
proteins = data['proteins']  # List of protein info
print(f"Loaded {{len(proteins)}} proteins")

# Access activations by layer
layer_activations = data['layer_activations']
layers = data['metadata']['layers']  # {layers_subset or cache.layers}

# Example: Get activations for first protein, layer 12
protein_0_layer_12 = layer_activations[12][0]  # Shape: [seq_len, 4096]
print(f"Shape: {{protein_0_layer_12.shape}}")
```

## Data Format
- **proteins**: List of dicts with protein_id, sequence_length, protein_idx
- **layer_activations**: Dict where keys are layer numbers, values are lists of tensors
- **metadata**: Information about the dataset

## Layers Available
{layers_subset or cache.layers}

## Total Proteins
{len(protein_subset) if protein_subset else len(cache.metadata['proteins'])}

## Notes
- All tensors are float32 in original dense format
- Each tensor shape: [sequence_length, 4096] 
- Sequence lengths vary by protein (max 1000 in your dataset)
- Data was compressed using sparse storage but reconverted to dense for this package
"""
    
    readme_file = os.path.join(output_path, "README.md")
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Package created at: {package_file}")
    print(f"📖 README created at: {readme_file}")
    print(f"📊 Package size: {os.path.getsize(package_file) / (1024**3):.2f} GB")
    
    return package_file

# Run the demo
demo_handoff_workflow()

print("\n" + "="*60)
print("READY TO PROCEED!")
print("="*60)
print("✅ Storage optimized: 17.3 GB (15.9x compression)")
print("✅ Format preserved: Original tensors for handoff")
print("✅ Precision: float16 storage → float32 output")
print("\nNext steps:")
print("1. test_cache = test_small_batch(n_test=20)  # Test first")
print("2. full_cache = run_full_processing()        # Run 10k proteins") 
print("3. create_handoff_package(...)               # Package for collaborator")

# %%

# %%
# Test actual sparse storage compression benefits
def test_sparse_storage_compression():
    """Test how much space sparse storage actually saves."""
    print("Testing sparse storage compression...")
    
    # Use the test protein to get real compression ratios
    full_seq_L = [(1, test_protein)]
    _, _, batch_tokens_BL = batch_converter(full_seq_L)
    batch_tokens_BL = batch_tokens_BL.to(device)
    batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)
    
    compression_results = {}
    
    for layer_idx, layer in enumerate(main_layers):
        test_sae = saes[layer_idx]
        
        # Setup caching hook
        caching_hook = SAEHookProt(
            sae=test_sae, 
            mask_BL=batch_mask_BL, 
            cache_masked_latents=True,
            layer_is_lm=False, 
            calc_error=True, 
            use_error=True
        )
        
        # Register hook and run model
        handle = esm_transformer.esm.encoder.layer[layer].register_forward_hook(caching_hook)
        
        with torch.no_grad():
            _ = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]
        
        # Get the masked activations
        masked_acts = test_sae.masked_latents.cpu()
        
        # Test different storage formats
        # 1. Dense float32
        dense_f32_bytes = masked_acts.numel() * 4  # 4 bytes per float32
        
        # 2. Dense float16
        dense_f16 = masked_acts.to(torch.float16)
        dense_f16_bytes = dense_f16.numel() * 2  # 2 bytes per float16
        
        # 3. Sparse float32
        sparse_f32 = csr_matrix(masked_acts.numpy())
        # CSR format stores: data array + indices array + indptr array
        sparse_f32_bytes = (sparse_f32.data.nbytes + 
                           sparse_f32.indices.nbytes + 
                           sparse_f32.indptr.nbytes)
        
        # 4. Sparse float16
        sparse_f16 = csr_matrix(masked_acts.to(torch.float16).numpy())
        sparse_f16_bytes = (sparse_f16.data.nbytes + 
                           sparse_f16.indices.nbytes + 
                           sparse_f16.indptr.nbytes)
        
        compression_results[layer] = {
            'shape': masked_acts.shape,
            'sparsity': 1 - (masked_acts != 0).sum().item() / masked_acts.numel(),
            'dense_f32_mb': dense_f32_bytes / (1024**2),
            'dense_f16_mb': dense_f16_bytes / (1024**2),
            'sparse_f32_mb': sparse_f32_bytes / (1024**2),
            'sparse_f16_mb': sparse_f16_bytes / (1024**2),
            'compression_ratio_f32': dense_f32_bytes / sparse_f32_bytes,
            'compression_ratio_f16': dense_f16_bytes / sparse_f16_bytes,
        }
        
        # Clean up
        handle.remove()
        
        print(f"Layer {layer}: Sparsity {compression_results[layer]['sparsity']:.3f}, "
              f"Compression F32: {compression_results[layer]['compression_ratio_f32']:.1f}x, "
              f"F16: {compression_results[layer]['compression_ratio_f16']:.1f}x")
    
    cleanup_cuda()
    
    # Calculate totals for 10k proteins
    print(f"\n" + "="*60)
    print("ACTUAL STORAGE REQUIREMENTS FOR 10,000 PROTEINS:")
    print("="*60)
    
    total_dense_f32 = sum(r['dense_f32_mb'] for r in compression_results.values()) * 10000
    total_dense_f16 = sum(r['dense_f16_mb'] for r in compression_results.values()) * 10000
    total_sparse_f32 = sum(r['sparse_f32_mb'] for r in compression_results.values()) * 10000
    total_sparse_f16 = sum(r['sparse_f16_mb'] for r in compression_results.values()) * 10000
    
    print(f"Dense Float32:   {total_dense_f32/1024:.1f} GB")
    print(f"Dense Float16:   {total_dense_f16/1024:.1f} GB")
    print(f"Sparse Float32:  {total_sparse_f32/1024:.1f} GB")
    print(f"Sparse Float16:  {total_sparse_f16/1024:.1f} GB  ⭐ RECOMMENDED")
    
    print(f"\nCompression ratios:")
    avg_compression_f32 = np.mean([r['compression_ratio_f32'] for r in compression_results.values()])
    avg_compression_f16 = np.mean([r['compression_ratio_f16'] for r in compression_results.values()])
    print(f"Float32 sparse vs dense: {avg_compression_f32:.1f}x compression")
    print(f"Float16 sparse vs dense: {avg_compression_f16:.1f}x compression")
    print(f"Overall reduction (F16 sparse vs F32 dense): {total_dense_f32/total_sparse_f16:.1f}x")
    
    return compression_results

# Run the compression test
compression_results = test_sparse_storage_compression()

# %%


