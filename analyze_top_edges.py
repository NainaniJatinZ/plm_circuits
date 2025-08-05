#!/usr/bin/env python3
"""
Analyze top edge attributions and identify interpretable connections.
"""

import json
from typing import Dict, List, Tuple
from collections import defaultdict
import sys
from io import StringIO

# Feature interpretations by layer
l4_latents: Dict[int, str] = {
    340: "FX'", 237: "FX'", 3788: "X'XI", 798: "D'XXGN", 1690: "X'XXF", 
    2277: "G", 2311: "X'XM", 3634: "X'XXXG", 1682: "PXXXXXX'", 3326: "H"
}

l8_latents: Dict[int, str] = {
    488: "AB_Hydrolase_fold", 2677: "FAD/NAD", 2775: "Transketolase", 2166: "DHFR"
}

l12_latents: Dict[int, str] = {
    2112: "AB_Hydrolase_fold", 3536: "SAM_mtases", 1256: "FAM", 2797: "Aldolase", 
    3794: "SAM_mtases", 3035: "WD40", 2302: "HotDog"
}

l16_latents: Dict[int, str] = {
    1504: "AB_Hydrolase_fold"
}

l20_latents: Dict[int, str] = {
    3615: "AB_Hydrolase_fold", 878: "Kinase"
}

l24_latents: Dict[int, str] = {
    2586: "Pectin lyase", 1822: "Kinase"
}

l28_latents: Dict[int, str] = {}  # No interpretable features in L28

# Map layer numbers to their interpretable features
layer_features = {
    4: l4_latents,
    8: l8_latents,
    12: l12_latents,
    16: l16_latents,
    20: l20_latents,
    24: l24_latents,
    28: l28_latents
}


def parse_layer_pair(layer_pair: str) -> Tuple[int, int]:
    """Parse layer pair string like 'L4_to_L8' into (4, 8)."""
    parts = layer_pair.split('_to_')
    up_layer = int(parts[0][1:])
    down_layer = int(parts[1][1:])
    return up_layer, down_layer


def get_all_edges(data: Dict) -> List[Tuple[int, int, int, int, float]]:
    """
    Extract all edges from the data structure.
    Returns list of (up_layer, up_feature, down_layer, down_feature, attribution).
    """
    all_edges = []
    
    for layer_pair, connections_data in data.items():
        up_layer, down_layer = parse_layer_pair(layer_pair)
        
        for connection in connections_data['all_connections']:
            up_feature, down_feature, attribution = connection
            all_edges.append((up_layer, up_feature, down_layer, down_feature, attribution))
    
    return all_edges


def analyze_top_edges(json_path: str, top_percent: float = 0.1, output_file: str = None):
    """Analyze top edges by absolute attribution value."""
    
    # Setup output capture if file specified
    original_stdout = sys.stdout
    output_buffer = StringIO() if output_file else None
    
    if output_file:
        sys.stdout = output_buffer
    
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get all edges
    all_edges = get_all_edges(data)
    
    # Sort by absolute attribution value
    all_edges.sort(key=lambda x: abs(x[4]), reverse=True)
    
    # Calculate statistics
    total_edges = len(all_edges)
    top_n = int(total_edges * top_percent)
    top_edges = all_edges[:top_n]
    
    # Calculate AUC statistics
    all_attributions = [abs(e[4]) for e in all_edges]
    top_attributions = [abs(e[4]) for e in top_edges]
    
    total_auc = sum(all_attributions)
    top_auc = sum(top_attributions)
    fraction_captured = top_auc / total_auc if total_auc > 0 else 0
    
    # Print header
    print("=" * 80)
    print(f"TOP {int(top_percent*100)}% EDGE ANALYSIS WITH INTERPRETABLE FEATURES")
    print("=" * 80)
    print()
    print(f"Analyzing top {int(top_percent*100)}% edges ({top_n} edges)")
    print(f"Area under curve for top {int(top_percent*100)}%: {top_auc:.4f}")
    print(f"Fraction of total AUC captured: {fraction_captured:.3f} ({fraction_captured*100:.1f}%)")
    print()
    
    # Find interpretable edges
    print("=" * 100)
    print("INTERPRETABLE EDGES (Both upstream and downstream latents are interpretable)")
    print("=" * 100)
    print(f"{'Rank':<7}{'Up Layer':<9}{'Up Latent':<11}{'Up Feature':<16}{'Down Layer':<11}{'Down Latent':<13}{'Down Feature':<16}{'Metric Change':<12}")
    print("-" * 100)
    
    interpretable_edges = []
    feature_pairs = defaultdict(int)
    
    for rank, edge in enumerate(top_edges, 1):
        up_layer, up_feature, down_layer, down_feature, attribution = edge
        
        # Check if both features are interpretable
        up_name = layer_features.get(up_layer, {}).get(up_feature)
        down_name = layer_features.get(down_layer, {}).get(down_feature)
        
        if up_name and down_name:
            interpretable_edges.append((rank, edge, up_name, down_name))
            feature_pairs[f"{up_name} → {down_name}"] += 1
            
            # Print the edge - preserve sign and format
            print(f"{rank:<7}{up_layer:<9}{up_feature:<11}{up_name:<16}{down_layer:<11}{down_feature:<13}{down_name:<16}{attribution:<12.6f}")
    
    print()
    print(f"Found {len(interpretable_edges)} interpretable edges in the top {int(top_percent*100)}%")
    print()
    
    # Print feature pair summary
    print("Feature pair summary:")
    for pair, count in sorted(feature_pairs.items()):
        print(f"{pair}: {count} edge{'s' if count > 1 else ''}")
    
    # Save to file if specified
    if output_file:
        sys.stdout = original_stdout
        output_content = output_buffer.getvalue()
        with open(output_file, 'w') as f:
            f.write(output_content)
        print(f"Analysis saved to {output_file}")
        # Also print to console
        print(output_content)


def main():
    import sys
    
    # Allow specifying json path as command line argument
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        output_dir = json_path.rsplit('/', 1)[0]
    else:
        json_path = "edge_attribution_results_all_layer_jvp_2B61A/detailed_results.json"
        output_dir = "edge_attribution_results_all_layer_jvp_2B61A"
    
    # Analyze top 10% edges and save to file
    analyze_top_edges(json_path, top_percent=0.1, output_file=f"{output_dir}/top_edges_analysis_10pct.txt")
    
    print("\n" + "=" * 80 + "\n")
    
    # Also analyze top 20% for comparison and save to file
    analyze_top_edges(json_path, top_percent=0.2, output_file=f"{output_dir}/top_edges_analysis_20pct.txt")


if __name__ == "__main__":
    main()