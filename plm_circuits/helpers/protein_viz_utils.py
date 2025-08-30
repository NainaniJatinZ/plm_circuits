"""
Part of this code is adapted from:
Simon et al. https://github.com/ElanaPearl/interPLM
"""

import os
import requests
import tempfile
from Bio.PDB import PDBIO, PDBList, PDBParser
import py3Dmol
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
import gc
import torch

##############################################################################
# Functions for loading and processing protein structures for visualization
##############################################################################

from IPython.display import HTML, display

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

def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()

def get_single_chain_afdb_structure(uniprot_id: str, PDB_DIR="pdbs"):
    """
    Downloads/loads an AlphaFold PDB from the EBI site for the given UniProt ID,
    returning the Bio.PDB structure object for the first model.
    """
    os.makedirs(PDB_DIR, exist_ok=True)
    pdb_file_path = os.path.join(PDB_DIR, f"AF-{uniprot_id}-F1-model_v4.pdb")
    if not os.path.exists(pdb_file_path):
        afdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        try:
            response = requests.get(afdb_url)
            response.raise_for_status()
            with open(pdb_file_path, "w") as pdb_file:
                pdb_file.write(response.text)
        except requests.RequestException:
            print("Error downloading PDB file from AlphaFold")
            return None
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("af", pdb_file_path)
    return structure[0]  # Return the first model

def get_single_chain_pdb_structure(pdb_id: str, chain_id: str, PDB_DIR="pdbs"):
    """
    Downloads a PDB from RCSB (if needed) and returns the specified chain
    as a Bio.PDB structure object.
    """
    os.makedirs(PDB_DIR, exist_ok=True)
    pdb_list = PDBList()
    pdb_file = pdb_list.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=PDB_DIR, overwrite=True)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rcsb", pdb_file)
    return structure[0][chain_id]

def structure_to_pdb_text(structure) -> str:
    """
    Converts a Bio.PDB structure to text by writing to a temporary file.
    """
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_file:
        io = PDBIO()
        io.set_structure(structure)
        io.save(temp_file.name)
        temp_file.seek(0)
        return temp_file.read()

def set_residue_bfactor(structure, values):
    """
    For each residue in 'structure', assigns the corresponding value from
    'values' to every atom’s B-factor.
    """
    residues = list(structure.get_residues())
    if len(residues) != len(values):
        raise ValueError(f"Structure has {len(residues)} residues but {len(values)} values were provided.")
    for res, val in zip(residues, values):
        for atom in res.get_atoms():
            atom.bfactor = float(val)

##############################################################################
# Custom colormaps for visualization
##############################################################################

def rwb_colormap_fn(value: float, vmin: float, vmax: float) -> str:
    """
    Red-White-Blue colormap: maps value in [vmin, vmax] to a color.
    0 -> red, 0.5 -> white, 1 -> blue.
    """
    if vmin == vmax:
        norm = 0.5
    else:
        norm = (value - vmin) / (vmax - vmin)
    
    if norm < 0.5:
        t = norm / 0.5
        r, g, b = 1.0, t, t
    else:
        t = (norm - 0.5) / 0.5
        r, g, b = 1.0 - t, 1.0 - t, 1.0
    return f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"

def mono_colormap_fn(value: float, vmin: float, vmax: float) -> str:
    """
    Monochromatic white-to-blue colormap.
    """
    if vmin == vmax:
        norm = 0.5
    else:
        norm = (value - vmin) / (vmax - vmin)
    R = int(255 * (1 - norm))
    G = int(255 * (1 - norm))
    B = 255
    return f"rgb({R},{G},{B})"

##############################################################################
# Visualization functions
##############################################################################

def view_single_protein(
    pdb_id: str | None = None,
    chain_id: str | None = None,
    uniprot_id: str | None = None,
    values_to_color: list | None = None,
    colormap_fn = rwb_colormap_fn,
    opacity_fn = None,
    default_color: str = "white",
    residues_to_highlight: list | None = None,
    highlight_color: str = "magenta",
    pymol_params: dict = None,
) -> py3Dmol.view:
    """
    Loads a protein structure (from RCSB or AlphaFold), colors residues using a colormap,
    and returns a py3Dmol viewer with PyMOL-style settings.
    
    Args:
        pdb_id: PDB identifier (if using RCSB).
        chain_id: Chain identifier for the PDB.
        uniprot_id: UniProt ID (if using AlphaFold).
        values_to_color: List of numeric values to color each residue.
        colormap_fn: Function to map a value to a color string.
        opacity_fn: Function to map a value to opacity (0-1). If None, uses 1.0.
        default_color: Base color for the structure.
        residues_to_highlight: List of residue indices (0-indexed) to highlight.
        highlight_color: Color for highlighted residues.
        pymol_params: Dict of parameters for the viewer (e.g., width, height).
    """
    if pymol_params is None:
        pymol_params = {"width": 600, "height": 500}

    if uniprot_id is not None:
        # For AlphaFold, assume chain "A" if not provided.
        chain_id = chain_id or "A"
        structure = get_single_chain_afdb_structure(uniprot_id)
    elif pdb_id and chain_id:
        structure = get_single_chain_pdb_structure(pdb_id, chain_id)
    else:
        raise ValueError("Must provide either uniprot_id or (pdb_id and chain_id).")
    
    if structure is None:
        raise ValueError("Structure not found or failed to load.")

    residues = list(structure.get_residues())
    n_res = len(residues)
    print(f"Structure has {n_res} residues in chain {chain_id}.")

    if values_to_color is None:
        values_to_color = [0.0] * n_res
    if len(values_to_color) != n_res:
        raise ValueError(f"Provided {len(values_to_color)} values but chain has {n_res} residues.")

    # Store values in the B-factor for hover information
    set_residue_bfactor(structure, values_to_color)
    pdb_text = structure_to_pdb_text(structure)

    view = py3Dmol.view(**pymol_params)
    view.addModel(pdb_text, "pdb")
    
    # Apply PyMOL-style settings for better appearance
    view.setBackgroundColor("white")  # White background like PyMOL setting
    view.setStyle({}, {"cartoon": {"color": default_color}})

    act_min = min(values_to_color)
    act_max = max(values_to_color)

    for i, (res, val) in enumerate(zip(residues, values_to_color)):
        res_id_in_pdb = res.id[1]  # residue id from PDB
        color_str = colormap_fn(val, act_min, act_max)
        # Use opacity function if provided, otherwise default to 1.0
        opacity = opacity_fn(val) if opacity_fn else 1.0
        view.setStyle({"chain": chain_id, "resi": res_id_in_pdb}, {"cartoon": {"color": color_str, "opacity": opacity}})
        if residues_to_highlight and i in residues_to_highlight:
            view.addStyle({"chain": chain_id, "resi": res_id_in_pdb}, {"stick": {"color": highlight_color}})
            resname = res.get_resname()
            view.addLabel(f"{resname}", {"fontOpacity": 1, "backgroundOpacity": 0.0, "fontSize": 20, "fontColor": highlight_color}, {"chain": chain_id, "resi": res_id_in_pdb})
    
    # Add hover callback to show residue and activation value
    hover_on = """
    function(atom, viewer, event, container) {
        if(!atom.label) {
            let text = atom.resn + atom.resi + ": " + atom.b.toFixed(3);
            atom.label = viewer.addLabel(text, {
                position: {x: atom.x, y: atom.y, z: atom.z},
                backgroundColor: 'black',
                fontColor: 'white',
                fontSize: 12,
                showBackground: true
            });
        }
    }
    """
    hover_off = """
    function(atom, viewer) {
        if(atom.label) {
            viewer.removeLabel(atom.label);
            delete atom.label;
        }
    }
    """
    view.setHoverable({}, True, hover_on, hover_off)
    view.zoomTo()
    return view

def show_colorbar(colormap_fn, vmin: float, vmax: float, steps: int = 10) -> HTML:
    """
    Displays a horizontal color bar (as HTML) using the provided colormap function.
    """
    step_vals = [vmin + i*(vmax-vmin)/(steps-1) for i in range(steps)]
    squares = [
        f'<div style="display:inline-block;width:20px;height:20px;background:{colormap_fn(val, vmin, vmax)};margin-right:1px;"></div>'
        for val in step_vals
    ]
    html_str = f"<div style='white-space:nowrap;font-family:monospace;'>{vmin:6.2f}&nbsp;{''.join(squares)}&nbsp;{vmax:6.2f}</div>"
    return HTML(html_str)

def vizualize_latent_on_protein(
    sequence: str,
    activations,
    latent_idx: int,
    seq_name: str,
    path: str,
    save: bool = True,
    prefix_index: int = 1,
    title_suffix: str = ""
):
    """
    Visualizes activations of a specific latent dimension along a protein sequence
    as a bar plot.
    
    Args:
        sequence: Protein sequence.
        activations: Activation tensor or array (shape: [sequence_length, ...]).
        latent_idx: Index of the latent dimension to plot.
        seq_name: Name of the protein.
        path: Directory to save the plot.
        save: If True, saves the plot; otherwise displays it.
        prefix_index: Starting index for residue numbering.
        title_suffix: Extra text to append to the title.
    """
    tensor_activations = np.array(activations)[:, latent_idx]
    
    # Map each amino acid to a distinct color
    unique_amino_acids = sorted(set(sequence))
    color_map = {acid: f"C{i}" for i, acid in enumerate(unique_amino_acids)}
    bar_colors = [color_map[acid] for acid in sequence]
    
    x_indices = np.arange(len(tensor_activations)) + prefix_index
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x_indices, tensor_activations, color=bar_colors)
    
    for x, acid, activation in zip(x_indices, sequence, tensor_activations):
        ax.text(x, activation + 0.01, acid, ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Activation Value")
    ax.set_title(f"Latent {latent_idx} Activations on {seq_name} {title_suffix}")
    ax.set_xticks(x_indices[::10])
    ax.set_xticklabels(x_indices[::10])
    
    plt.tight_layout()
    if save:
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f"{seq_name}_latent_{latent_idx}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

##############################################################################
# Domain-specific visualization functions
##############################################################################

def view_protein_domain(
    pdb_id: str,
    chain_id: str = "A",
    domain_start: int = None,
    domain_end: int = None,
    catalytic_residues: list = None,
    domain_name: str = "Protein Domain",
    colormap_fn = None,
    pymol_params: dict = None,
) -> py3Dmol.view:
    """
    Visualize a specific protein domain with optional catalytic site highlighting.
    
    Args:
        pdb_id: PDB identifier.
        chain_id: Chain identifier.
        domain_start: Starting residue number for domain (if None, use full structure).
        domain_end: Ending residue number for domain (if None, use full structure).
        catalytic_residues: List of residue numbers to highlight as catalytic sites.
        domain_name: Name of the domain for display.
        colormap_fn: Custom colormap function.
        pymol_params: Dict of parameters for the viewer.
    """
    if pymol_params is None:
        pymol_params = {"width": 800, "height": 600}
    
    # Load structure
    structure = get_single_chain_pdb_structure(pdb_id, chain_id)
    residues = list(structure.get_residues())
    
    # Filter to domain if specified
    if domain_start is not None and domain_end is not None:
        domain_residues = [res for res in residues 
                          if domain_start <= res.id[1] <= domain_end]
        print(f"Domain {domain_name}: residues {domain_start}-{domain_end} ({len(domain_residues)} residues)")
    else:
        domain_residues = residues
        print(f"Full structure: {len(domain_residues)} residues")
    
    # Create activation pattern
    n_res = len(residues)
    values_to_color = [0.0] * n_res
    
    # Highlight domain region
    if domain_start is not None and domain_end is not None:
        for i, res in enumerate(residues):
            res_num = res.id[1]
            if domain_start <= res_num <= domain_end:
                values_to_color[i] = 1.0
            else:
                values_to_color[i] = -0.5
    
    # Highlight catalytic residues
    residues_to_highlight = []
    if catalytic_residues:
        for i, res in enumerate(residues):
            res_num = res.id[1]
            if res_num in catalytic_residues:
                values_to_color[i] = 2.0
                residues_to_highlight.append(i)
        print(f"Highlighted catalytic residues: {catalytic_residues}")
    
    # Use default domain colormap if none provided
    if colormap_fn is None:
        def domain_colormap_fn(value: float, vmin: float, vmax: float) -> str:
            if value > 1.5:
                return "#FF0000"  # Red for catalytic
            elif value > 0.5:
                return "#4169E1"  # Blue for domain
            elif value >= 0:
                return "#F0F0F0"  # Light gray for background
            else:
                return "#FFA07A"  # Light salmon for outside domain
        colormap_fn = domain_colormap_fn
    
    # Create visualization
    view_obj = view_single_protein(
        pdb_id=pdb_id,
        chain_id=chain_id,
        values_to_color=values_to_color,
        colormap_fn=colormap_fn,
        default_color="white",
        residues_to_highlight=residues_to_highlight,
        highlight_color="lime",
        pymol_params=pymol_params
    )
    
    return view_obj

def get_alpha_beta_hydrolase_info():
    """
    Get information about well-characterized alpha/beta hydrolase structures.
    
    Returns:
        dict: Dictionary containing PDB IDs and their structural information.
    """
    return {
        "4EY4": {
            "name": "Human Acetylcholinesterase",
            "organism": "Homo sapiens",
            "resolution": "2.16 Å",
            "catalytic_triad": [200, 440, 327],  # Ser200, His440, Glu327
            "domain_range": [20, 530],
            "description": "Classic alpha/beta hydrolase fold with deep active site gorge",
            "cath_classification": "3.40.50.1820"
        },
        "1EA5": {
            "name": "Torpedo Acetylcholinesterase",
            "organism": "Tetronarce californica", 
            "resolution": "1.80 Å",
            "catalytic_triad": [200, 440, 327],
            "domain_range": [20, 530],
            "description": "High-resolution alpha/beta hydrolase structure",
            "cath_classification": "3.40.50.1820"
        },
        "3LII": {
            "name": "Human Acetylcholinesterase (recombinant)",
            "organism": "Homo sapiens",
            "resolution": "3.20 Å", 
            "catalytic_triad": [200, 440, 327],
            "domain_range": [20, 530],
            "description": "Recombinant human AChE with blocked active site",
            "cath_classification": "3.40.50.1820"
        }
    }

def visualize_alpha_beta_hydrolase(pdb_id: str = "4EY4", highlight_catalytic: bool = True):
    """
    Quick function to visualize alpha/beta hydrolase fold structures.
    
    Args:
        pdb_id: PDB ID of the structure to visualize.
        highlight_catalytic: Whether to highlight catalytic triad residues.
        
    Returns:
        py3Dmol.view: The 3D viewer object.
    """
    ab_hydrolase_info = get_alpha_beta_hydrolase_info()
    
    if pdb_id.upper() not in ab_hydrolase_info:
        raise ValueError(f"PDB ID {pdb_id} not in known alpha/beta hydrolase structures. "
                        f"Available: {list(ab_hydrolase_info.keys())}")
    
    info = ab_hydrolase_info[pdb_id.upper()]
    
    print(f"Visualizing {info['name']} (PDB: {pdb_id.upper()})")
    print(f"Resolution: {info['resolution']}")
    print(f"Description: {info['description']}")
    
    catalytic_residues = info['catalytic_triad'] if highlight_catalytic else None
    
    view_obj = view_protein_domain(
        pdb_id=pdb_id,
        chain_id="A",
        domain_start=info['domain_range'][0],
        domain_end=info['domain_range'][1],
        catalytic_residues=catalytic_residues,
        domain_name="Alpha/Beta Hydrolase Domain",
        pymol_params={"width": 900, "height": 650}
    )
    
    return view_obj
