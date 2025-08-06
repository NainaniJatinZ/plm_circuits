# %%
"""
Enhanced Protein Domain Structure Visualization Script
=====================================================

This script extends the basic protein visualization to support:
1. Visualization of conserved protein domains (like alpha/beta hydrolase fold)
2. Highlighting of structural features (catalytic triad, binding sites, etc.)
3. Domain extraction from full protein structures
4. Multiple structure comparison and alignment

Key Features:
- Support for alpha/beta hydrolase fold visualization
- Catalytic triad highlighting
- Secondary structure annotation
- Domain boundary visualization
- Multiple structure overlays
"""

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

import helpers.protein_viz_utils as viz
import torch
import numpy as np
import json
from functools import partial
from Bio.PDB import PDBParser, PDBIO, Chain, Model, Structure
from Bio.PDB.DSSP import DSSP
import requests
import tempfile
import os

# %%

class DomainStructureVisualizer:
    """Enhanced protein structure visualizer with domain support"""
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Well-known alpha/beta hydrolase structures with catalytic residues
        self.ab_hydrolase_structures = {
            "4EY4": {  # Human acetylcholinesterase (apo state)
                "name": "Human Acetylcholinesterase",
                "chain": "A",
                "catalytic_triad": [200, 440, 327],  # Ser200, His440, Glu327
                "domain_range": [20, 530],  # Approximate domain boundaries
                "description": "Classic alpha/beta hydrolase fold with deep active site gorge"
            },
            "1EA5": {  # Torpedo acetylcholinesterase 
                "name": "Torpedo Acetylcholinesterase",
                "chain": "A", 
                "catalytic_triad": [200, 440, 327],  # Ser200, His440, Glu327
                "domain_range": [20, 530],
                "description": "High-resolution structure of alpha/beta hydrolase"
            },
            "3LII": {  # Human acetylcholinesterase (another form)
                "name": "Human Acetylcholinesterase (recombinant)",
                "chain": "A",
                "catalytic_triad": [200, 440, 327],  # Ser200, His440, Glu327
                "domain_range": [20, 530],
                "description": "Recombinant human AChE structure"
            }
        }
        
        # Define secondary structure colors
        self.ss_colors = {
            'H': '#FF6B6B',  # Alpha helix - red
            'B': '#4ECDC4',  # Beta bridge - teal  
            'E': '#45B7D1',  # Beta strand - blue
            'G': '#96CEB4',  # 3-10 helix - light green
            'I': '#FFEAA7',  # Pi helix - yellow
            'T': '#DDA0DD',  # Turn - plum
            'S': '#F4A261',  # Bend - orange
            '-': '#E0E0E0'   # Coil/loop - light gray
        }
        
    def get_structure_with_domain_info(self, pdb_id: str, chain_id: str = "A"):
        """Load structure and extract domain information"""
        structure = viz.get_single_chain_pdb_structure(pdb_id, chain_id)
        residues = list(structure.get_residues())
        
        domain_info = self.ab_hydrolase_structures.get(pdb_id.upper(), {})
        
        return structure, residues, domain_info
    
    def annotate_catalytic_residues(self, residues, catalytic_triad):
        """Create activation pattern highlighting catalytic triad"""
        activation = np.zeros(len(residues))
        
        for i, res in enumerate(residues):
            res_num = res.id[1]
            if res_num in catalytic_triad:
                activation[i] = 2.0  # High activation for catalytic residues
        
        return activation
    
    def annotate_secondary_structure(self, pdb_file_path, chain_id="A"):
        """Use DSSP to annotate secondary structure"""
        try:
            # Note: DSSP requires installation (sudo apt-get install dssp)
            # For now, we'll create a simplified mock annotation
            return self._mock_secondary_structure_annotation()
        except:
            print("DSSP not available, using simplified secondary structure annotation")
            return self._mock_secondary_structure_annotation()
    
    def _mock_secondary_structure_annotation(self):
        """Create a mock secondary structure pattern for demonstration"""
        # This would be replaced with real DSSP analysis
        # For alpha/beta hydrolase, typical pattern is alternating beta-alpha-beta
        return ['E', 'E', 'E', 'T', 'H', 'H', 'H', 'H', 'T', 'E', 'E', 'E'] * 50
    
    def create_domain_activation_pattern(self, residues, domain_info, annotation_type="catalytic"):
        """Create activation patterns for different domain features"""
        activation = np.zeros(len(residues))
        
        if annotation_type == "catalytic" and "catalytic_triad" in domain_info:
            # Highlight catalytic triad
            catalytic_residues = domain_info["catalytic_triad"]
            for i, res in enumerate(residues):
                res_num = res.id[1]
                if res_num in catalytic_residues:
                    activation[i] = 2.0
                    
        elif annotation_type == "domain_boundary":
            # Highlight domain boundaries
            if "domain_range" in domain_info:
                start, end = domain_info["domain_range"]
                for i, res in enumerate(residues):
                    res_num = res.id[1]
                    if start <= res_num <= end:
                        activation[i] = 1.0
                    else:
                        activation[i] = -0.5
                        
        elif annotation_type == "conservation":
            # Mock conservation scores (would be from real data)
            activation = np.random.normal(0, 0.5, len(residues))
            # Make catalytic residues highly conserved
            if "catalytic_triad" in domain_info:
                for i, res in enumerate(residues):
                    res_num = res.id[1]
                    if res_num in domain_info["catalytic_triad"]:
                        activation[i] = 1.5
        
        return activation
    
    def visualize_alpha_beta_hydrolase(self, pdb_id="4EY4", annotation_type="catalytic", 
                                     custom_title=None, highlight_residues=None):
        """Visualize alpha/beta hydrolase fold with specified annotations"""
        
        pdb_id = pdb_id.upper()
        if pdb_id not in self.ab_hydrolase_structures:
            raise ValueError(f"PDB ID {pdb_id} not in known alpha/beta hydrolase structures. "
                           f"Available: {list(self.ab_hydrolase_structures.keys())}")
        
        domain_info = self.ab_hydrolase_structures[pdb_id]
        chain_id = domain_info["chain"]
        
        print(f"Loading {domain_info['name']} (PDB: {pdb_id})")
        print(f"Description: {domain_info['description']}")
        
        # Load structure
        structure, residues, _ = self.get_structure_with_domain_info(pdb_id, chain_id)
        
        # Create activation pattern based on annotation type
        activation = self.create_domain_activation_pattern(residues, domain_info, annotation_type)
        
        # Define residues to highlight (catalytic triad by default)
        if highlight_residues is None and "catalytic_triad" in domain_info:
            # Convert absolute residue numbers to relative indices
            highlight_residues = []
            catalytic_residues = domain_info["catalytic_triad"]
            for i, res in enumerate(residues):
                if res.id[1] in catalytic_residues:
                    highlight_residues.append(i)
        
        # Create custom colormap for domain features
        def domain_colormap_fn(value: float, vmin: float, vmax: float) -> str:
            """Custom colormap for domain visualization"""
            if annotation_type == "catalytic":
                if value > 1.5:
                    return "#FF0000"  # Bright red for catalytic residues
                elif value > 0.5:
                    return "#FFA500"  # Orange for important residues
                elif value == 0:
                    return "#F0F0F0"  # Light gray for background
                else:
                    return "#E0E0E0"  # Gray for less important
            elif annotation_type == "domain_boundary":
                if value > 0.5:
                    return "#4169E1"  # Royal blue for domain
                else:
                    return "#FFA07A"  # Light salmon for outside domain
            else:  # conservation
                if value > 1.0:
                    return "#8B0000"  # Dark red for highly conserved
                elif value > 0.5:
                    return "#FF6347"  # Tomato for moderately conserved
                elif value > 0:
                    return "#FFD700"  # Gold for weakly conserved
                elif value > -0.5:
                    return "#F0F0F0"  # Light gray for neutral
                else:
                    return "#ADD8E6"  # Light blue for variable
        
        # Create visualization
        title_suffix = custom_title or f"({annotation_type.replace('_', ' ').title()} annotation)"
        
        view_obj = viz.view_single_protein(
            pdb_id=pdb_id,
            chain_id=chain_id,
            values_to_color=activation,
            colormap_fn=domain_colormap_fn,
            default_color="white",
            residues_to_highlight=highlight_residues or [],
            highlight_color="lime",
            pymol_params={"width": 1000, "height": 700}
        )
        
        # Display results
        from IPython.display import display, HTML
        
        # Show information
        info_html = f"""
        <div style='margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px;'>
            <h3>{domain_info['name']} - Alpha/Beta Hydrolase Fold</h3>
            <p><strong>PDB ID:</strong> {pdb_id} | <strong>Chain:</strong> {chain_id}</p>
            <p><strong>Description:</strong> {domain_info['description']}</p>
            <p><strong>Annotation:</strong> {annotation_type.replace('_', ' ').title()}</p>
            {f"<p><strong>Catalytic Triad:</strong> {', '.join(map(str, domain_info['catalytic_triad']))}</p>" if 'catalytic_triad' in domain_info else ""}
            <p><strong>Features:</strong> This structure represents the classic alpha/beta hydrolase fold with 
            an alpha/beta sheet core and catalytic triad typical of serine hydrolases.</p>
        </div>
        """
        display(HTML(info_html))
        
        # Show colorbar
        display(viz.show_colorbar(domain_colormap_fn, 
                                min(activation), 
                                max(activation), 
                                steps=8))
        
        # Show 3D structure
        display(view_obj.show())
        
        return view_obj, activation, domain_info
    
    def compare_ab_hydrolase_structures(self, pdb_ids=["4EY4", "1EA5"], annotation_type="catalytic"):
        """Compare multiple alpha/beta hydrolase structures side by side"""
        
        print("Comparing Alpha/Beta Hydrolase Structures:")
        print("=" * 50)
        
        results = {}
        for pdb_id in pdb_ids:
            print(f"\nAnalyzing {pdb_id}...")
            view_obj, activation, domain_info = self.visualize_alpha_beta_hydrolase(
                pdb_id, annotation_type, custom_title=f"Comparison - {pdb_id}"
            )
            results[pdb_id] = {
                "view": view_obj,
                "activation": activation,
                "domain_info": domain_info
            }
        
        return results
    
    def extract_domain_region(self, pdb_id, chain_id="A", start_res=None, end_res=None):
        """Extract a specific domain region from a protein structure"""
        structure = viz.get_single_chain_pdb_structure(pdb_id, chain_id)
        
        if pdb_id.upper() in self.ab_hydrolase_structures:
            domain_info = self.ab_hydrolase_structures[pdb_id.upper()]
            if start_res is None or end_res is None:
                start_res, end_res = domain_info.get("domain_range", [1, 999])
        
        # Filter residues to domain range
        residues = [res for res in structure.get_residues() 
                   if start_res <= res.id[1] <= end_res]
        
        print(f"Extracted domain region {start_res}-{end_res} from {pdb_id}")
        print(f"Domain contains {len(residues)} residues")
        
        return residues

# %%

# Initialize the domain visualizer
domain_viz = DomainStructureVisualizer()

# %%

# Example 1: Visualize human acetylcholinesterase with catalytic triad highlighted
print("Example 1: Human Acetylcholinesterase - Catalytic Triad")
view1, activation1, info1 = domain_viz.visualize_alpha_beta_hydrolase(
    pdb_id="4EY4", 
    annotation_type="catalytic"
)

# %%

# Example 2: Show domain boundaries
print("\nExample 2: Human Acetylcholinesterase - Domain Boundaries")
view2, activation2, info2 = domain_viz.visualize_alpha_beta_hydrolase(
    pdb_id="4EY4", 
    annotation_type="domain_boundary"
)

# %%

# Example 3: Conservation pattern (mock data)
print("\nExample 3: Human Acetylcholinesterase - Conservation Pattern")
view3, activation3, info3 = domain_viz.visualize_alpha_beta_hydrolase(
    pdb_id="4EY4", 
    annotation_type="conservation"
)

# %%

# Example 4: Compare different alpha/beta hydrolase structures
print("\nExample 4: Comparison of Alpha/Beta Hydrolase Structures")
comparison_results = domain_viz.compare_ab_hydrolase_structures(
    pdb_ids=["4EY4", "1EA5"], 
    annotation_type="catalytic"
)

# %%

# Example 5: Extract domain information
print("\nExample 5: Domain Analysis")
for pdb_id, info in domain_viz.ab_hydrolase_structures.items():
    print(f"\n{pdb_id}: {info['name']}")
    print(f"  Catalytic Triad: {info.get('catalytic_triad', 'Not defined')}")
    print(f"  Domain Range: {info.get('domain_range', 'Not defined')}")
    print(f"  Description: {info['description']}")

# %%

# Example 6: Create a custom alpha/beta hydrolase visualization
def create_custom_ab_hydrolase_viz(pdb_id="4EY4"):
    """Create a comprehensive alpha/beta hydrolase visualization"""
    
    from IPython.display import display, HTML
    
    # Load structure info
    domain_info = domain_viz.ab_hydrolase_structures[pdb_id]
    
    # Create comprehensive HTML report
    report_html = f"""
    <div style='font-family: Arial, sans-serif; max-width: 900px; margin: 20px auto; padding: 20px; 
                border: 2px solid #ddd; border-radius: 10px; background-color: #fafafa;'>
        
        <h2 style='color: #2E86AB; text-align: center; margin-bottom: 30px;'>
            Alpha/Beta Hydrolase Fold Analysis: {domain_info['name']}
        </h2>
        
        <div style='background: #E8F4FD; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
            <h3 style='color: #1B4F72; margin-top: 0;'>Structural Classification</h3>
            <p><strong>Fold Family:</strong> Alpha/Beta Hydrolase (CATH: 3.40.50.1820)</p>
            <p><strong>PDB ID:</strong> {pdb_id} | <strong>Chain:</strong> {domain_info['chain']}</p>
            <p><strong>Enzyme Class:</strong> Hydrolase (EC 3.1.1.7 - Acetylcholinesterase)</p>
        </div>
        
        <div style='background: #FDF2E8; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
            <h3 style='color: #C55A11; margin-top: 0;'>Key Structural Features</h3>
            <ul>
                <li><strong>Core Fold:</strong> Central alpha/beta sheet with 8 beta strands</li>
                <li><strong>Catalytic Mechanism:</strong> Serine hydrolase with catalytic triad</li>
                <li><strong>Active Site:</strong> Deep gorge lined with aromatic residues</li>
                <li><strong>Catalytic Triad:</strong> Ser{domain_info['catalytic_triad'][0]}, 
                    His{domain_info['catalytic_triad'][1]}, Glu{domain_info['catalytic_triad'][2]}</li>
            </ul>
        </div>
        
        <div style='background: #E8F5E8; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
            <h3 style='color: #1B7A1B; margin-top: 0;'>Biological Function</h3>
            <p>{domain_info['description']}</p>
            <p>The alpha/beta hydrolase fold is one of the most common enzyme folds, found in 
            esterases, lipases, proteases, and dehalogenases. The conserved catalytic triad 
            and oxyanion hole are key to the hydrolytic mechanism.</p>
        </div>
        
        <div style='background: #F8E8F8; padding: 15px; border-radius: 8px;'>
            <h3 style='color: #7A1B7A; margin-top: 0;'>Conservation & Evolution</h3>
            <p>The alpha/beta hydrolase fold represents one of the largest superfamilies in the protein 
            structure universe. Despite low sequence similarity, the fold architecture and catalytic 
            mechanism are highly conserved across species and enzyme families.</p>
        </div>
    </div>
    """
    
    display(HTML(report_html))
    
    # Now show the structure with catalytic residues highlighted
    return domain_viz.visualize_alpha_beta_hydrolase(pdb_id, "catalytic")

# Create the comprehensive visualization
print("Comprehensive Alpha/Beta Hydrolase Fold Analysis:")
custom_view, custom_activation, custom_info = create_custom_ab_hydrolase_viz("4EY4")

# %%