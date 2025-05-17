import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import (
    king_wen_sequence, hamming_distance,
    circular_distance, get_opposite_position,
    get_quarter_positions
)

def analyze_transformation_network():
    """Analyze the transformation network of the King Wen sequence."""
    print("\nRunning network analysis...")
    return analyze_transformation_network_internal()

def analyze_transformation_network_internal():
    """Analyze the King Wen sequence as a network of transformations, considering circularity."""
    
    # Create a directed graph where nodes are hexagrams and edges are transformations
    G = nx.DiGraph()
    
    # Add nodes
    for i, hexagram in enumerate(king_wen_sequence):
        G.add_node(i, hexagram=hexagram, label=f"{i+1}")
    
    # Define transformation functions
    transformations = {
        'identity': lambda h: h,
        'inversion': lambda h: ''.join('1' if bit == '0' else '0' for bit in h),
        'reversal': lambda h: h[::-1],
        'rev_inv': lambda h: ''.join('1' if bit == '0' else '0' for bit in h[::-1]),
        'swap_trigrams': lambda h: h[3:] + h[:3]
    }
    
    # Add edges for sequence order (including wrap-around)
    n = len(king_wen_sequence)
    for i in range(n):
        from_hex = king_wen_sequence[i]
        to_hex = king_wen_sequence[(i + 1) % n]  # Wrap around
        
        # Determine the transformation type
        trans_type = "complex"
        for name, func in transformations.items():
            if to_hex == func(from_hex):
                trans_type = name
                break
        
        # If no exact match, use Hamming distance
        if trans_type == "complex":
            h_dist = hamming_distance(from_hex, to_hex)
            if h_dist == 1:
                trans_type = "single_line"
            elif h_dist == 2:
                trans_type = "two_line"
        
        G.add_edge(i, (i + 1) % n, type=trans_type, sequence=True)
    
    # Add edges for all possible transformations (not just sequence order)
    for i, hex_i in enumerate(king_wen_sequence):
        for j, hex_j in enumerate(king_wen_sequence):
            if i == j:
                continue
                
            # Skip if already have a sequence edge
            if G.has_edge(i, j) and G.edges[i, j].get('sequence', False):
                continue
                
            # Check for transformations
            for name, func in transformations.items():
                if name == 'identity':
                    continue
                    
                if hex_j == func(hex_i):
                    G.add_edge(i, j, type=name, sequence=False)
    
    # Calculate network metrics
    sequence_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('sequence', False)]
    sequence_subgraph = G.edge_subgraph(sequence_edges)
    
    print("\nNetwork Analysis (Circular):")
    print(f"Total nodes (hexagrams): {G.number_of_nodes()}")
    print(f"Total edges (all transformations): {G.number_of_edges()}")
    print(f"Sequence edges: {len(sequence_edges)}")
    
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    top_central = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 central hexagrams (degree centrality):")
    for node, centrality in top_central:
        print(f"Hexagram {node+1}: {centrality:.4f}")
    
    # Calculate betweenness centrality (identifies "bridge" hexagrams)
    betweenness_centrality = nx.betweenness_centrality(G)
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("\nTop 10 bridge hexagrams (betweenness centrality):")
    for node, centrality in top_betweenness:
        print(f"Hexagram {node+1}: {centrality:.4f}")
    
    # Identify communities in the transformation network
    communities = nx.community.greedy_modularity_communities(G.to_undirected())
    
    print(f"\nFound {len(communities)} communities in the transformation network:")
    for i, community in enumerate(communities):
        if len(community) > 5:
            members = [node+1 for node in community]  # Convert to traditional numbering
            print(f"Community {i+1}: {len(community)} members. Sample: {members[:5]}...")
        else:
            members = [node+1 for node in community]
            print(f"Community {i+1}: {members}")
    
    # Analyze opposite positions
    print("\nAnalysis of opposite positions:")
    for i in range(n):
        opposite_pos = get_opposite_position(i)
        h_dist = hamming_distance(king_wen_sequence[i], king_wen_sequence[opposite_pos])
        print(f"Hexagram {i+1} ↔ {opposite_pos+1}: Hamming distance = {h_dist}")
    
    # Visualize the transformation network using toroidal embedding
    plt.figure(figsize=(15, 15))
    
    # Create a toroidal layout
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 1.0
    pos = {}
    for i in range(n):
        angle = angles[i]
        pos[i] = (radius * np.cos(angle), radius * np.sin(angle))
    
    # Create a colormap for the communities
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
    
    # Draw nodes colored by community
    node_colors = [community_map[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color=node_colors, cmap=plt.cm.tab20)
    
    # Draw sequence edges more prominently
    nx.draw_networkx_edges(G, pos, edgelist=sequence_edges, width=2, edge_color='black')
    
    # Draw other transformation edges lightly
    non_sequence_edges = [(u, v) for u, v, d in G.edges(data=True) if not d.get('sequence', False)]
    nx.draw_networkx_edges(G, pos, edgelist=non_sequence_edges, width=0.5, alpha=0.2, edge_color='gray')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Circular Transformation Network of the King Wen Sequence')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Visualize quarter positions
    quarter_positions = get_quarter_positions()
    plt.figure(figsize=(15, 15))
    
    # Draw the network again
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color=node_colors, cmap=plt.cm.tab20)
    nx.draw_networkx_edges(G, pos, edgelist=sequence_edges, width=2, edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Highlight quarter positions
    quarter_nodes = [pos[i] for i in quarter_positions]
    quarter_x = [p[0] for p in quarter_nodes]
    quarter_y = [p[1] for p in quarter_nodes]
    plt.scatter(quarter_x, quarter_y, s=300, color='red', marker='*', label='Quarter Positions')
    
    plt.title('Quarter Positions in the Circular Transformation Network')
    plt.axis('equal')
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return G, communities