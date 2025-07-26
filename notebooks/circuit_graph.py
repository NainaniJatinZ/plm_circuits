import json 
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Load the layer-latent dictionary
with open('data/layer_latent_dict_2b61.json', 'r') as f:
    layer_latent_dict = json.load(f)

print("Layer-Latent Dictionary loaded:")
for layer, latents in layer_latent_dict.items():
    print(f"Layer {layer}: {len(latents)} latents")

# Create a directed graph
G = nx.DiGraph()

# Define colors for different node types
colors = {
    'latent': '#4CAF50',      # Green for latent nodes
    'error': '#F44336',       # Red for error nodes  
    'other': '#2196F3'        # Blue for other nodes
}

# Store node positions and types for visualization
node_positions = {}
node_types = {}
layer_info = {}

# Add nodes for each layer with better spatial distribution
layers = sorted([int(layer) for layer in layer_latent_dict.keys()])
layer_width = 4.0  # Width allocated for each layer
max_nodes_per_layer = max(len(layer_latent_dict[str(layer)]) + 2 for layer in layers)  # +2 for error and other
total_height = max_nodes_per_layer * 0.8  # Total height for distribution

for i, layer in enumerate(layers):
    layer_str = str(layer)
    latent_indices = layer_latent_dict[layer_str]
    
    # Calculate x position for this layer
    x_pos = i * layer_width
    
    # Calculate total nodes for this layer (latents + error + other)
    total_nodes_this_layer = len(latent_indices) + 2
    
    # Distribute nodes evenly across available vertical space
    if total_nodes_this_layer > 1:
        y_spacing = total_height / (total_nodes_this_layer - 1)
    else:
        y_spacing = 0
    
    # Add latent nodes with equal distribution
    layer_nodes = []
    for j, latent_idx in enumerate(latent_indices):
        node_id = f"L{layer}_latent_{latent_idx}"
        G.add_node(node_id)
        y_pos = j * y_spacing
        node_positions[node_id] = (x_pos, y_pos)
        node_types[node_id] = 'latent'
        layer_nodes.append(node_id)
    
    # Add error node
    error_node_id = f"L{layer}_error"
    G.add_node(error_node_id)
    y_pos = len(latent_indices) * y_spacing
    node_positions[error_node_id] = (x_pos, y_pos)
    node_types[error_node_id] = 'error'
    layer_nodes.append(error_node_id)
    
    # Add other node
    other_node_id = f"L{layer}_other"
    G.add_node(other_node_id)
    y_pos = (len(latent_indices) + 1) * y_spacing
    node_positions[other_node_id] = (x_pos, y_pos)
    node_types[other_node_id] = 'other'
    layer_nodes.append(other_node_id)
    
    layer_info[layer] = {
        'nodes': layer_nodes,
        'latent_count': len(latent_indices),
        'x_pos': x_pos,
        'total_nodes': total_nodes_this_layer
    }

# Add random edges between layers (early to later layers)
random.seed(42)  # For reproducible results
edge_probability = 0.02  # Probability of creating an edge (reduced drastically)

print("\nAdding random edges between layers...")
edge_count = 0

for i, source_layer in enumerate(layers[:-1]):  # All but last layer
    for j, target_layer in enumerate(layers[i+1:], i+1):  # All later layers
        source_nodes = layer_info[source_layer]['nodes']
        target_nodes = layer_info[target_layer]['nodes']
        
        # Create some random edges
        for source_node in source_nodes:
            for target_node in target_nodes:
                if random.random() < edge_probability:
                    G.add_edge(source_node, target_node)
                    edge_count += 1

print(f"Added {edge_count} random edges")

# Create two visualizations: custom layout and NetworkX spring layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

# First subplot: Custom layer-based layout
plt.sca(ax1)

# Draw nodes by type with larger sizes
node_sizes = {
    'latent': 300,  # Larger nodes for latents
    'error': 500,   # Even larger for error nodes
    'other': 500    # Even larger for other nodes
}

for node_type, color in colors.items():
    nodes_of_type = [node for node, ntype in node_types.items() if ntype == node_type]
    if nodes_of_type:
        pos_subset = {node: node_positions[node] for node in nodes_of_type}
        nx.draw_networkx_nodes(G, pos_subset, nodelist=nodes_of_type, 
                             node_color=color, node_size=node_sizes[node_type], alpha=0.8)

# Add labels to special nodes (error and other nodes) and some key latent nodes
labels = {}

# Add labels for error and other nodes
for node_id, node_type in node_types.items():
    if node_type == 'error':
        labels[node_id] = 'ERR'
    elif node_type == 'other':
        labels[node_id] = 'OTHER'
    elif node_type == 'latent':
        # Add labels only to first few latent nodes in each layer to avoid clutter
        layer_match = node_id.split('_')[0]  # Gets 'L4', 'L8', etc.
        latent_idx = int(node_id.split('_')[-1])  # Gets the latent index
        # Only label first 3 latent nodes in each layer
        layer_latents = [n for n in node_types.keys() if n.startswith(layer_match) and node_types[n] == 'latent']
        if node_id in sorted(layer_latents)[:3]:  # First 3 latent nodes in each layer
            labels[node_id] = str(latent_idx)

# Draw labels
nx.draw_networkx_labels(G, node_positions, labels, font_size=8, font_weight='bold')

# Draw edges
nx.draw_networkx_edges(G, node_positions, edge_color='gray', alpha=0.3, 
                      arrows=True, arrowsize=10, arrowstyle='->')

# Add layer labels and background rectangles
for layer in layers:
    info = layer_info[layer]
    x_pos = info['x_pos']
    max_y = max([node_positions[node][1] for node in info['nodes']])
    min_y = min([node_positions[node][1] for node in info['nodes']])
    
    # Add background rectangle for layer
    rect = Rectangle((x_pos - 0.4, min_y - 0.2), 0.8, max_y - min_y + 0.4, 
                    linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.2)
    ax1.add_patch(rect)
    
    # Add layer label
    plt.text(x_pos, max_y + 0.5, f'Layer {layer}', 
            horizontalalignment='center', fontsize=12, fontweight='bold')
    
    # Add latent count
    plt.text(x_pos, min_y - 0.4, f'{info["latent_count"]} latents', 
            horizontalalignment='center', fontsize=9)

# Create legend for first subplot
legend_elements = [
    mpatches.Patch(color=colors['latent'], label='Latent Nodes'),
    mpatches.Patch(color=colors['error'], label='Error Nodes'),
    mpatches.Patch(color=colors['other'], label='Other Nodes')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Set title and labels for first subplot
plt.title('Custom Layer Layout\n(Equally Distributed)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Network Layers', fontsize=12)
plt.ylabel('Node Index', fontsize=12)

# Remove axes ticks but keep labels
plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                left=False, right=False, labelbottom=True, labelleft=True)
plt.grid(True, alpha=0.3)

# Second subplot: NetworkX multipartite layout (layer-aware)
plt.sca(ax2)

# Add layer information to nodes for multipartite layout
for node_id, node_type in node_types.items():
    layer_num = int(node_id.split('_')[0][1:])  # Extract layer number from node ID
    G.nodes[node_id]['layer'] = layer_num

# Use multipartite layout which is designed for layered graphs
# Other layer-aware NetworkX layouts you could try:
# - nx.nx_agraph.graphviz_layout(G, prog='dot')  # Requires pygraphviz
# - nx.kamada_kawai_layout(G)  # Good for structured layouts  
# - nx.shell_layout(G, nlist=[layer_info[layer]['nodes'] for layer in layers])  # Concentric shells
multipartite_positions = nx.multipartite_layout(G, subset_key='layer', align='horizontal')

# Draw nodes by type with multipartite layout
for node_type, color in colors.items():
    nodes_of_type = [node for node, ntype in node_types.items() if ntype == node_type]
    if nodes_of_type:
        pos_subset = {node: multipartite_positions[node] for node in nodes_of_type}
        nx.draw_networkx_nodes(G, pos_subset, nodelist=nodes_of_type, 
                             node_color=color, node_size=node_sizes[node_type], alpha=0.8)

# Add labels for multipartite layout (fewer labels to avoid clutter)
multipartite_labels = {}
for node_id, node_type in node_types.items():
    if node_type == 'error':
        multipartite_labels[node_id] = 'ERR'
    elif node_type == 'other':
        multipartite_labels[node_id] = 'OTHER'

# Draw labels for multipartite layout
nx.draw_networkx_labels(G, multipartite_positions, multipartite_labels, font_size=8, font_weight='bold')

# Draw edges for multipartite layout
nx.draw_networkx_edges(G, multipartite_positions, edge_color='gray', alpha=0.3, 
                      arrows=True, arrowsize=10, arrowstyle='->')

# Set title for second subplot
plt.title('NetworkX Multipartite Layout\n(Layer-Aware)', fontsize=14, fontweight='bold', pad=20)
plt.axis('off')  # Turn off axes for cleaner look

# Create legend for second subplot
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Overall figure title
fig.suptitle('Circuit Diagram: Layer-Latent Network (Random Edges - Early to Later Layers)', 
             fontsize=16, fontweight='bold', y=0.95)

# Adjust layout and save
plt.tight_layout()
plt.savefig('data/circuit_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print(f"\nGraph Summary:")
print(f"- Total nodes: {G.number_of_nodes()}")
print(f"- Total edges: {G.number_of_edges()}")
print(f"- Number of layers: {len(layers)}")
print(f"- Layers: {layers}")

# Save the graph for later use
print("\nSaving graph data for future interactive use...")
graph_data = {
    'nodes': list(G.nodes()),
    'edges': list(G.edges()),
    'node_positions': node_positions,
    'multipartite_positions': multipartite_positions,
    'node_types': node_types,
    'layer_info': layer_info,
    'layers': layers
}

# with open('data/circuit_graph_data.json', 'w') as f:
#     json.dump(graph_data, f, indent=2)

print("Graph data saved to 'data/circuit_graph_data.json'")
