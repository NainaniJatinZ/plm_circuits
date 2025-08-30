import os
import json
import torch

from scipy.sparse import csr_matrix, save_npz, load_npz

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

    def load_layer(self, layer_idx):
        if layer_idx not in self.layers:
            raise IndexError(f"Layer index {layer_idx} out of range")


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
