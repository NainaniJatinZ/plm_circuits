# %%
"""
Simple Alpha/Beta Hydrolase Domain Visualization Example
========================================================

This script demonstrates how to visualize conserved protein domains,
specifically the alpha/beta hydrolase fold found in many enzymes.

The alpha/beta hydrolase fold (CATH superfamily 3.40.50.1820) is one of the 
most common protein folds, found in enzymes like:
- Acetylcholinesterases
- Lipases  
- Esterases
- Proteases
- Dehalogenases

Key structural features:
- Central alpha/beta sheet with ~8 beta strands
- Catalytic triad (typically Ser-His-Asp/Glu)
- Deep active site gorge (in some members)
- Conserved oxyanion hole for catalysis
"""

import sys
sys.path.append('../plm_circuits')

import helpers.protein_viz_utils as viz
from IPython.display import display, HTML

# %%

# Display information about available alpha/beta hydrolase structures
print("Available Alpha/Beta Hydrolase Structures:")
print("=" * 50)

ab_hydrolase_info = viz.get_alpha_beta_hydrolase_info()
for pdb_id, info in ab_hydrolase_info.items():
    print(f"\n{pdb_id}: {info['name']}")
    print(f"  Organism: {info['organism']}")
    print(f"  Resolution: {info['resolution']}")
    print(f"  CATH Classification: {info['cath_classification']}")
    print(f"  Description: {info['description']}")
    print(f"  Catalytic Triad: Ser{info['catalytic_triad'][0]}, His{info['catalytic_triad'][1]}, Glu{info['catalytic_triad'][2]}")

# %%

# Example 1: Quick visualization of human acetylcholinesterase
print("\nExample 1: Human Acetylcholinesterase (4EY4)")
print("-" * 45)

# This is the simplest way to visualize an alpha/beta hydrolase domain
view1 = viz.visualize_alpha_beta_hydrolase("4EY4", highlight_catalytic=True)
display(view1.show())

# %%

# Example 2: Compare different alpha/beta hydrolase structures
print("\nExample 2: High-resolution Torpedo Acetylcholinesterase (1EA5)")
print("-" * 58)

view2 = viz.visualize_alpha_beta_hydrolase("1EA5", highlight_catalytic=True)
display(view2.show())

# %%

# Example 3: Focus on domain boundaries without catalytic highlighting
print("\nExample 3: Domain boundaries only (no catalytic highlighting)")
print("-" * 64)

# Use the more flexible view_protein_domain function for custom visualization
view3 = viz.view_protein_domain(
    pdb_id="4EY4",
    chain_id="A", 
    domain_start=20,
    domain_end=530,
    catalytic_residues=None,  # Don't highlight catalytic residues
    domain_name="Alpha/Beta Hydrolase Domain",
    pymol_params={"width": 800, "height": 600}
)
display(view3.show())

# %%

# Example 4: Custom domain visualization with specific residues highlighted
print("\nExample 4: Custom highlighting of specific functional residues")
print("-" * 62)

# Highlight some key residues in the active site gorge
important_residues = [84, 200, 327, 330, 440]  # Including Trp84, Phe330 (aromatic residues in gorge)

view4 = viz.view_protein_domain(
    pdb_id="4EY4",
    chain_id="A",
    domain_start=20,
    domain_end=530, 
    catalytic_residues=important_residues,
    domain_name="Alpha/Beta Hydrolase with Active Site Residues",
    pymol_params={"width": 900, "height": 700}
)
display(view4.show())

# %%

# Example 5: Educational overview with detailed information
def create_educational_overview():
    """Create an educational overview of the alpha/beta hydrolase fold"""
    
    overview_html = """
    <div style='font-family: Arial, sans-serif; max-width: 1000px; margin: 20px auto; 
                padding: 25px; border: 3px solid #4169E1; border-radius: 15px; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);'>
        
        <h1 style='color: #2E86AB; text-align: center; margin-bottom: 30px; 
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
            🧬 Alpha/Beta Hydrolase Fold: A Structural Biology Perspective
        </h1>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;'>
            
            <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='color: #C55A11; margin-top: 0; border-bottom: 2px solid #C55A11; padding-bottom: 5px;'>
                    🏗️ Structural Architecture
                </h3>
                <ul style='line-height: 1.6;'>
                    <li><strong>Core Structure:</strong> Central alpha/beta sheet</li>
                    <li><strong>Topology:</strong> ~8 parallel beta strands</li>
                    <li><strong>Active Site:</strong> Located at C-terminal edge of sheet</li>
                    <li><strong>CATH ID:</strong> 3.40.50.1820</li>
                </ul>
            </div>
            
            <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='color: #1B7A1B; margin-top: 0; border-bottom: 2px solid #1B7A1B; padding-bottom: 5px;'>
                    ⚙️ Catalytic Mechanism
                </h3>
                <ul style='line-height: 1.6;'>
                    <li><strong>Mechanism:</strong> Nucleophilic attack by serine</li>
                    <li><strong>Catalytic Triad:</strong> Ser-His-Asp/Glu</li>
                    <li><strong>Oxyanion Hole:</strong> Stabilizes transition state</li>
                    <li><strong>Substrate Range:</strong> Esters, amides, peptides</li>
                </ul>
            </div>
            
        </div>
        
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 25px;'>
            <h3 style='color: #7A1B7A; margin-top: 0; border-bottom: 2px solid #7A1B7A; padding-bottom: 5px;'>
                🌍 Evolutionary Significance & Examples
            </h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px;'>
                <div>
                    <h4 style='color: #2E86AB; margin-bottom: 8px;'>Acetylcholinesterases</h4>
                    <p style='font-size: 0.9em; margin: 0;'>Neurotransmitter breakdown, synaptic transmission</p>
                </div>
                <div>
                    <h4 style='color: #2E86AB; margin-bottom: 8px;'>Lipases</h4>
                    <p style='font-size: 0.9em; margin: 0;'>Fat digestion, biotechnology applications</p>
                </div>
                <div>
                    <h4 style='color: #2E86AB; margin-bottom: 8px;'>Proteases</h4>
                    <p style='font-size: 0.9em; margin: 0;'>Protein processing, blood clotting</p>
                </div>
            </div>
        </div>
        
        <div style='background: #E8F4FD; padding: 20px; border-radius: 10px; border: 2px solid #4169E1;'>
            <h3 style='color: #1B4F72; margin-top: 0; text-align: center;'>
                💡 Key Learning Points
            </h3>
            <p style='line-height: 1.6; margin-bottom: 10px;'>
                The alpha/beta hydrolase fold demonstrates how evolution conserves <strong>structural architecture</strong> 
                and <strong>catalytic mechanism</strong> while allowing sequence diversity for different substrate specificities.
                This fold family shows that protein function can be understood through structural analysis.
            </p>
            <p style='line-height: 1.6; margin: 0;'>
                <strong>Clinical Relevance:</strong> Many drugs target alpha/beta hydrolases, including Alzheimer's medications 
                (acetylcholinesterase inhibitors) and cholesterol-lowering drugs (lipase inhibitors).
            </p>
        </div>
        
    </div>
    """
    
    display(HTML(overview_html))

print("\nEducational Overview:")
create_educational_overview()

# Now show the structure
print("\nStructural Visualization:")
view5 = viz.visualize_alpha_beta_hydrolase("4EY4", highlight_catalytic=True)
display(view5.show())

# %%