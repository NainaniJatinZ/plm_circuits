import gc
import os
import re
import torch
import random
import numpy as np
from IPython import get_ipython
from huggingface_hub import hf_hub_download
from .sae_model_interprot import SparseAutoencoder
from safetensors.torch import load_file
from transformers import AutoTokenizer, EsmForMaskedLM
from esm import FastaBatchedDataset, pretrained

VERBOSE = False

def set_verbose(flag: bool) -> None:
    global VERBOSE
    VERBOSE = bool(flag)


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def activate_autoreload():
    try:
        ipython = get_ipython()
        if ipython is not None:
            ipython.magic("load_ext autoreload")
            ipython.magic("autoreload 2")
            print("In IPython")
            print("Set autoreload")
        else:
            print("Not in IPython")
    except NameError:
        print("`get_ipython` not available. This script is not running in IPython.")

# Call the function during script initialization
activate_autoreload()

# ---------------------------------------------------------------------------
# GPU house‑keeping
# ---------------------------------------------------------------------------

def cleanup_cuda() -> None:
    """Run a GC sweep and empty the PyTorch CUDA cache – prevents OOM."""
    import gc

    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def norm_sum_mult(t1_LL: torch.Tensor, t2_LL: torch.Tensor) -> torch.Tensor:  # noqa: N802
    """∑(a·b) / ∑(a·a) – used for contact‑map recovery metrics."""
    return (t1_LL * t2_LL).sum() / (t1_LL * t1_LL).sum()


def contact_recovery(
    preds_LL: torch.Tensor,
    target_LL: torch.Tensor,
    ss1_start: int,
    ss1_end: int,
    ss2_start: int,
    ss2_end: int,
) -> torch.Tensor:
    seg_pred = preds_LL[ss1_start:ss1_end, ss2_start:ss2_end]
    seg_true = target_LL[ss1_start:ss1_end, ss2_start:ss2_end]
    return norm_sum_mult(seg_true, seg_pred)

# ---------------------------------------------------------------------------
# Model loaders (ESM + SAE)
# ---------------------------------------------------------------------------

def load_esm(model_size, WEIGHTS_DIR='/work/pi_jensen_umass_edu/jnainani_umass_edu/ESM_Interp/weights/',device: torch.device = torch.device("cuda")):
    """Return (model, alphabet, batch_converter) for the requested ESM‑2 size."""
    sizes = {6: "esm2_t6_8M_UR50D", 33: "esm2_t33_650M_UR50D", 36: "esm2_t36_3B_UR50D"}
    if model_size not in sizes:
        raise ValueError(f"Unsupported ESM‑2 size: {model_size}")
    
    esm_model_name = sizes[model_size]
    os.environ["TORCH_HOME"] = WEIGHTS_DIR  # Force PyTorch cache directory
    os.environ["HF_HOME"] = WEIGHTS_DIR
     
    _, esm2_alphabet = pretrained.load_model_and_alphabet(esm_model_name)
    esm_transformer = EsmForMaskedLM.from_pretrained(f"facebook/{esm_model_name}", cache_dir=WEIGHTS_DIR).to(device)
    batch_converter = esm2_alphabet.get_batch_converter()
    return esm_transformer, batch_converter, esm2_alphabet


def load_sae_prot(ESM_DIM=1280, SAE_DIM=4096, LAYER=24, device="cuda"):
    """Load a Sparse Autoencoder trained for a specific ESM layer."""
    checkpoint_path = hf_hub_download(
    repo_id="liambai/InterProt-ESM2-SAEs",
    filename=f"esm2_plm{ESM_DIM}_l{LAYER}_sae{SAE_DIM}.safetensors"
    )
    sae_model = SparseAutoencoder(ESM_DIM, SAE_DIM)
    sae_model.load_state_dict(load_file(checkpoint_path))
    sae_model.to(device)
    return sae_model

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
unusual_AA ="OU" #Pyrrolysine O and selenocysteine U
indeterminate_AA = "BJXZ" #B = Asparagine or Aspartic acid; J = leucine or isoleucine; X = Any/Unknown ; Z = Glutamine or glutamic acid

# Adjust this path as needed
PDB_DIR = "pdbs"

# 3-letter to 1-letter lookup
aa_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

def clear_memory(saes, model, mask_bool=False):
    """
    Clears out the gradients from the SAEs and the main model
    to avoid accumulation or weird artifacts.

    Args:
        saes (List[SAE]): A list of SAE objects (or similarly structured modules).
        model (nn.Module): The main model whose gradients we want to clear.
        mask_bool (bool): Whether to also clear mask gradients if they exist.
    """
    for sae in saes:
        for param in sae.parameters():
            if param.grad is not None:
                param.grad = None
        if mask_bool and hasattr(sae, 'mask'):
            for param in sae.mask.parameters():
                if param.grad is not None:
                    param.grad = None
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    
    return saes, model


def patch_sum(matrix, top_left, size=22):
    i, j = top_left
    return sum(matrix[x][y] for x in range(i, i + size) for y in range(j, j + size))

def find_diagonal_patches(matrix, size=22, threshold=15, sample_numb=3):
    m = len(matrix)
    patches = []

    for i in range(m - size + 1):
        j = i  
        # select those with enough contacts + enough space (10) for explore flanking values 
        if (patch_sum(matrix, (i, j), size) > threshold) and (10 < i < (matrix.shape[0] - size - 10)):
            patches.append((i, j))
            
    sampled_patches = random.sample(patches, min(sample_numb, len(patches))) 

    return sampled_patches

def find_diagonal_patches_all(matrix, size=22, threshold=15, sample_numb=3):
    m = len(matrix)
    patches = []

    for i in range(m - size + 1):
        j = i  
        # select those with enough contacts + enough space (10) for explore flanking values 
        if (patch_sum(matrix, (i, j), size) > threshold) and (10 < i < (matrix.shape[0] - size - 10)):
            patches.append((i, j))
            
    # sampled_patches = random.sample(patches, min(sample_numb, len(patches))) 

    return patches

def get_contact(seq, esm_transformer, esm2_alphabet, batch_converter, device='cuda'): 
    seq_tuple = [(1, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_tuple)
    #batch_tokens = torch.cat((torch.full((batch_tokens.shape[0], 1), 32), batch_tokens[:, 1:-1], torch.full((batch_tokens.shape[0], 1), 32)), dim=1)
    batch_tokens = batch_tokens.to(device)
    batch_mask = (batch_tokens != esm2_alphabet.padding_idx).to(device)
    with torch.no_grad():
        esm2_predictions = esm_transformer.predict_contacts(batch_tokens, batch_mask)[0].cpu()
    return esm2_predictions.numpy()

def norm_sum_mult(ori_contact_seg, seg_cross_contact):
    ori_mult_new = np.multiply(ori_contact_seg, seg_cross_contact)
    ori_mult_ori = np.multiply(ori_contact_seg, ori_contact_seg)
    return (np.sum(ori_mult_new)/np.sum(ori_mult_ori))


dist_range_min = 50
dist_range_max = 100 
expand_type = 'outward' 

# get all SSEs
def get_segments(input_str):
    segments = []
    for match in re.finditer('E+|H+', input_str):
        if (match.group()[0] == 'E' and len(match.group()) > 3) or \
           (match.group()[0] == 'H' and len(match.group()) > 7):
            segments.append((match.start(), match.end()))
    return segments

# get centers of SSEs 
def get_ss_cents(segments): 
    ss_cents = []
    for seg in segments: 
        ss_cents.append((seg[1] + seg[0])//2) 
    return ss_cents 

# select pairs of SSEs separated by certain distance 
def get_pairs(arr):
    pairs = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if dist_range_min < abs(arr[i] - arr[j]) <= dist_range_max: # look at long separation ones 
                pairs.append((arr[i], arr[j]))
    return pairs

# take 5 res on both sides 
# select segment pairs with enough contacts + leaves enough distances for exploring different flanking lengths 
def select_pairs(cent_pairs, matrix, cutoff, seq_len):
    selected_pairs = []
    for pair in cent_pairs: 
        ss1_start = pair[0] - 5 
        ss2_end = pair[1] + 5 + 1 
        patch_sum = np.sum(matrix[(pair[0] - 5): (pair[0] + 6), (pair[1] - 5): (pair[1] + 6)])
        # check there is enough contact between the two SSE 
        # check there is enough region for expanding to check recovery 
        if (patch_sum > cutoff) and (min(ss1_start, seq_len - ss2_end - 1) > 10):
            selected_pairs.append(pair) 
    n = min(len(selected_pairs), 3)
    selected_pairs = random.sample(selected_pairs, n)
    return selected_pairs 

# get the masked sequence and then the contact map 
def get_seg_contact(sequence, frag1_start, frag1_end, frag2_start, frag2_end, esm_transformer, esm2_alphabet, batch_converter, device): # flank_len is the amount of residues to add at sides of the segments  
    seg_seq_i = sequence[frag1_start: frag1_end] 
    seg_seq_j = sequence[frag2_start: frag2_end] 
    mask_length = frag2_start - frag1_end 
    full_seq = frag1_start * '<mask>' + seg_seq_i + mask_length * '<mask>' + seg_seq_j + (len(sequence) -  frag2_end) * '<mask>'
    contact_map = get_contact(full_seq, esm_transformer, esm2_alphabet, batch_converter, device) 
    return contact_map

def mask_flanks_segment(seq, ss1_start, ss1_end, ss2_start, ss2_end, unmask_left_idxs, unmask_right_idxs):
    """
    seq: original protein sequence (string)
    patch_start, patch_end: the region [patch_start:patch_end] is fully unmasked
    unmask_left_idxs, unmask_right_idxs: sets or lists of indices that we keep unmasked on the flanks
    Returns: masked_seq (string) with <mask> everywhere except patch + chosen flank indices
    """
    L = len(seq)
    seq_list = ["<mask>"] * L
    
    # Always unmask patch
    # Unmask the patch region
    seq_list[ss1_start: ss1_end] = list(seq[ss1_start: ss1_end] )
    seq_list[ss2_start: ss2_end] = list(seq[ss2_start: ss2_end] )
    # Unmask chosen flank residues
    for i in unmask_left_idxs:
        seq_list[i] = seq[i]
    for i in unmask_right_idxs:
        seq_list[i] = seq[i]
    
    return "".join(seq_list)

def norm_sum_mult(ori_contact_seg, seg_cross_contact):
    """
    Compute the normalized sum of element-wise multiplication between two tensors.
    Args:
        ori_contact_seg (torch.Tensor): Original contact segment (requires gradients).
        seg_cross_contact (torch.Tensor): Cross contact segment (requires gradients).
    Returns:
        torch.Tensor: Normalized sum of element-wise multiplication.
    """
    ori_mult_new = ori_contact_seg * seg_cross_contact
    ori_mult_ori = ori_contact_seg * ori_contact_seg
    return torch.sum(ori_mult_new) / torch.sum(ori_mult_ori)

def patching_metric(contact_preds, orig_contact, ss1_start, ss1_end, ss2_start, ss2_end):

    seg_cross_contact = contact_preds[ss1_start:ss1_end, ss2_start:ss2_end]
    orig_contact_seg = orig_contact[ss1_start:ss1_end, ss2_start:ss2_end]
    return torch.sum(seg_cross_contact * orig_contact_seg) / torch.sum(orig_contact_seg * orig_contact_seg)

def seq_to_tokens(seq: str, batch_converter, alphabet, device: torch.device):
    """Convenience wrapper: str‑>tokens_BL & mask_BL."""
    _, _, toks_BL = batch_converter([(1, seq)])
    toks_BL = toks_BL.to(device)
    return toks_BL, (toks_BL != alphabet.padding_idx).to(device)