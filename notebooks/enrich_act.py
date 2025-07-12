#!/usr/bin/env python3
"""
Efficient SAE Activation Capture for 10,000 Proteins
Minimal production script for capturing and loading activations.
"""

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

# Import utility functions
from helpers.utils import load_esm, load_sae_prot, cleanup_cuda
from hook_manager import SAEHookProt

class EfficientActivationCache:
    """Efficient storage system for SAE activations across multiple layers."""
    
    def __init__(self, save_path, layers, n_features=4096, use_sparse=True, precision='float16'):
        self.save_path = save_path
        self.layers = layers
        self.n_features = n_features
        self.use_sparse = use_sparse
        self.precision = getattr(torch, precision) if isinstance(precision, str) else precision
        
        os.makedirs(save_path, exist_ok=True)
        
        self.metadata = {
            'layers': layers,
            'n_features': n_features,
            'use_sparse': use_sparse,
            'precision': str(precision),
            'proteins': []
        }
    
    def save_protein_activations(self, protein_id, sequence, layer_activations_dict, protein_idx):
        """Save activations for a single protein across all layers."""
        protein_data = {
            'protein_id': protein_id,
            'sequence_length': len(sequence),
            'protein_idx': protein_idx
        }
        
        for layer in self.layers:
            acts = layer_activations_dict[layer].cpu()
            
            if self.precision != torch.float32:
                acts = acts.to(self.precision)
            
            if self.use_sparse:
                acts_np = acts.numpy()
                sparse_matrix = csr_matrix(acts_np)
                sparse_file = os.path.join(self.save_path, f"protein_{protein_idx:06d}_layer_{layer}_sparse.npz")
                save_npz(sparse_file, sparse_matrix)
                
                protein_data[f'layer_{layer}_file'] = sparse_file
                protein_data[f'layer_{layer}_shape'] = acts_np.shape
                protein_data[f'layer_{layer}_nnz'] = sparse_matrix.nnz
            else:
                acts_file = os.path.join(self.save_path, f"protein_{protein_idx:06d}_layer_{layer}.npy")
                np.save(acts_file, acts.numpy())
                protein_data[f'layer_{layer}_file'] = acts_file
                protein_data[f'layer_{layer}_shape'] = acts.shape
        
        self.metadata['proteins'].append(protein_data)
        
        if len(self.metadata['proteins']) % 100 == 0:
            self.save_metadata()
    
    def save_metadata(self):
        """Save metadata to JSON file."""
        metadata_file = os.path.join(self.save_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_protein_activations_original_format(self, protein_idx):
        """Load activations in original dense tensor format for handoff."""
        if protein_idx >= len(self.metadata['proteins']):
            raise IndexError(f"Protein index {protein_idx} out of range")
        
        protein_data = self.metadata['proteins'][protein_idx]
        activations = {}
        
        for layer in self.layers:
            file_path = protein_data[f'layer_{layer}_file']
            
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

def load_cached_activations(save_path):
    """Load a previously saved activation cache."""
    metadata_file = os.path.join(save_path, "metadata.json")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"No metadata file found at {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
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

def process_proteins_all_layers(records, n_proteins, save_path, device, esm_transformer, 
                              batch_converter, esm2_alphabet, saes, main_layers, layer_2_saelayer,
                              use_sparse=True, precision='float16', checkpoint_every=100):
    """Process proteins and cache activations for ALL layers efficiently."""
    
    print(f"Processing {n_proteins} proteins - Storage: {'Sparse' if use_sparse else 'Dense'}, Precision: {precision}")
    
    cache_system = EfficientActivationCache(
        save_path=save_path,
        layers=main_layers,
        use_sparse=use_sparse,
        precision=precision
    )
    
    proteins_to_process = random.sample(records, min(n_proteins, len(records)))
    start_time = time.time()
    
    for protein_idx in tqdm(range(len(proteins_to_process)), desc="Processing proteins"):
        protein_id, sequence = proteins_to_process[protein_idx]
        
        try:
            # Prepare sequence
            full_seq_L = [(1, sequence)]
            _, _, batch_tokens_BL = batch_converter(full_seq_L)
            batch_tokens_BL = batch_tokens_BL.to(device)
            batch_mask_BL = (batch_tokens_BL != esm2_alphabet.padding_idx).to(device)
            
            # Setup hooks for all layers
            hooks = []
            for layer_idx, layer in enumerate(main_layers):
                sae = saes[layer_idx]
                caching_hook = SAEHookProt(
                    sae=sae, 
                    mask_BL=batch_mask_BL, 
                    cache_masked_latents=True,
                    layer_is_lm=False, 
                    calc_error=True, 
                    use_error=True
                )
                handle = esm_transformer.esm.encoder.layer[layer].register_forward_hook(caching_hook)
                hooks.append(handle)
            
            # Single forward pass
            with torch.no_grad():
                _ = esm_transformer.predict_contacts(batch_tokens_BL, batch_mask_BL)[0]
            
            # Extract activations
            layer_activations = {}
            for layer_idx, layer in enumerate(main_layers):
                activations = saes[layer_idx].masked_latents.cpu()
                layer_activations[layer] = activations
            
            # Save activations
            cache_system.save_protein_activations(protein_id, sequence, layer_activations, protein_idx)
            
            # Cleanup
            for handle in hooks:
                handle.remove()
            cleanup_cuda()
            
        except Exception as e:
            print(f"Error processing protein {protein_id}: {e}")
            for handle in hooks:
                handle.remove()
            continue
        
        # Checkpoint
        if (protein_idx + 1) % checkpoint_every == 0:
            elapsed = time.time() - start_time
            rate = (protein_idx + 1) / elapsed
            eta = (n_proteins - protein_idx - 1) / rate / 60
            print(f"Checkpoint: {protein_idx + 1}/{n_proteins}, Rate: {rate:.2f} proteins/sec, ETA: {eta:.1f} min")
            cache_system.save_metadata()
    
    cache_system.save_metadata()
    total_time = time.time() - start_time
    print(f"Completed {len(cache_system.metadata['proteins'])} proteins in {total_time/60:.1f} minutes")
    
    return cache_system

# Main execution
if __name__ == "__main__":
    # CONFIGURATION - MODIFY THESE PATHS
    FASTA_PATH = pathlib.Path("../../plm_interp/uniprot_sprot.fasta")
    SAVE_PATH = "/project/pi_annagreen_umass_edu/jatin/plm_circuits/acts" #"/project/pi_jensen_umass_edu/jnainani_umass_edu/plm_data/full_activation_cache"
    N_PROTEINS = 10000
    RANDOM_SEED = 42  # Set to None for no seed, or any integer for reproducibility
    
    # Set random seed for reproducibility
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        print(f"Random seed set to: {RANDOM_SEED}")
    
    # Load data
    print("Loading protein records...")
    records = [
        (rec.id, str(rec.seq))
        for rec in SeqIO.parse(FASTA_PATH, "fasta")
        if len(rec.seq) <= 1022
    ]
    print(f"Loaded {len(records)} protein records")
    
    # Setup device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load ESM-2 model
    print("Loading ESM-2 model...")
    esm_transformer, batch_converter, esm2_alphabet = load_esm(33, device=device)
    
    # Load SAEs
    print("Loading SAE models...")
    main_layers = [4, 8, 12, 16, 20, 24, 28]
    saes = []
    for layer in main_layers:
        sae_model = load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=layer, device=device)
        saes.append(sae_model)
    
    layer_2_saelayer = {layer: layer_idx for layer_idx, layer in enumerate(main_layers)}
    
    # Process proteins
    print("Starting activation capture...")
    cache = process_proteins_all_layers(
        records=records,
        n_proteins=N_PROTEINS,
        save_path=SAVE_PATH,
        device=device,
        esm_transformer=esm_transformer,
        batch_converter=batch_converter,
        esm2_alphabet=esm2_alphabet,
        saes=saes,
        main_layers=main_layers,
        layer_2_saelayer=layer_2_saelayer,
        use_sparse=True,
        precision='float16',
        checkpoint_every=100
    )
    
    print(f"✅ Activation capture complete! Saved to: {SAVE_PATH}")
    print(f"📊 Total proteins processed: {len(cache.metadata['proteins'])}")
    print(f"🔄 To load: cache = load_cached_activations('{SAVE_PATH}')")
