import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import king_wen_sequence, hamming_distance

def analyze_cycles():
    """Analyze cycles in the King Wen sequence."""
    print("\nRunning cycle analysis...")
    return analyze_cycles_internal()

def analyze_cycles_internal():
    """Detect cycles in the King Wen sequence based on transformations."""
    
    # Build a directed graph of transformations
    G = nx.DiGraph()
    for i, hexagram in enumerate(king_wen_sequence):
        G.add_node(i, hexagram=hexagram, name=f"Hexagram {i+1}")
    
    # Define common transformations
    transformations = {
        'inversion': lambda h: ''.join('1' if bit == '0' else '0' for bit in h),
        'reversal': lambda h: h[::-1],
        'complement': lambda h: bin(63 - int(h, 2))[2:].zfill(6),
        'trigram_swap': lambda h: h[3:] + h[:3]
    }
    
    # Add edges for all possible transformations
    for i, hex_i in enumerate(king_wen_sequence):
        for j, hex_j in enumerate(king_wen_sequence):
            # Skip self-connections
            if i == j:
                continue
                
            # Check for transformations
            for trans_name, trans_func in transformations.items():
                if hex_j == trans_func(hex_i):
                    G.add_edge(i, j, transformation=trans_name)
    
    # --- CIRCULARITY MODIFICATION ---
    # Treat the sequence as circular by connecting last to first for each transformation type
    last_idx = len(king_wen_sequence) - 1
    first_idx = 0
    hex_last = king_wen_sequence[last_idx]
    hex_first = king_wen_sequence[first_idx]
    for trans_name, trans_func in transformations.items():
        if hex_first == trans_func(hex_last):
            G.add_edge(last_idx, first_idx, transformation=trans_name)
        if hex_last == trans_func(hex_first):
            G.add_edge(first_idx, last_idx, transformation=trans_name)
    # --- END CIRCULARITY MODIFICATION ---
    
    # Find all simple cycles in the graph
    cycles = list(nx.simple_cycles(G))
    
    # Filter for cycles of various lengths (2, 3, 4, etc.)
    cycles_by_length = {}
    for cycle in cycles:
        length = len(cycle)
        if length not in cycles_by_length:
            cycles_by_length[length] = []
        cycles_by_length[length].append(cycle)
    
    # Report and visualize findings
    print(f"Found {len(cycles)} cycles in total")
    
    for length, cycle_list in sorted(cycles_by_length.items()):
        print(f"\n{len(cycle_list)} cycles of length {length}:")
        
        # Limit output for clarity
        max_to_show = min(5, len(cycle_list))
        for i, cycle in enumerate(cycle_list[:max_to_show]):
            cycle_hex = [king_wen_sequence[idx] for idx in cycle]
            cycle_nums = [idx+1 for idx in cycle]  # Traditional hexagram numbers
            
            # Determine the transformations in this cycle
            transformations_in_cycle = []
            for k in range(len(cycle)):
                from_idx = cycle[k]
                to_idx = cycle[(k+1) % len(cycle)]
                
                # Find the transformation type
                for edge in G.edges(data=True):
                    if edge[0] == from_idx and edge[1] == to_idx:
                        transformations_in_cycle.append(edge[2]['transformation'])
                        break
            
            print(f"  Cycle {i+1}: {cycle_nums} (Transformations: {transformations_in_cycle})")
        
        if len(cycle_list) > max_to_show:
            print(f"  ... and {len(cycle_list) - max_to_show} more")
    
    # Visualize a few interesting cycles
    interesting_lengths = [2, 3, 4, 6, 8]  # Cycles of philosophical interest
    for length in interesting_lengths:
        if length in cycles_by_length and cycles_by_length[length]:
            # Take the first cycle of this length
            cycle = cycles_by_length[length][0]
            
            # Create a subgraph for this cycle
            cycle_nodes = set(cycle)
            cycle_graph = G.subgraph(cycle_nodes)
            
            plt.figure(figsize=(8, 8))
            pos = nx.circular_layout(cycle_graph)
            
            # Draw nodes
            nx.draw_networkx_nodes(cycle_graph, pos, node_size=700, node_color='lightblue')
            
            # Draw edges with labels
            edge_labels = {(u, v): d['transformation'] for u, v, d in cycle_graph.edges(data=True)}
            nx.draw_networkx_edges(cycle_graph, pos, width=2, arrowsize=20)
            nx.draw_networkx_edge_labels(cycle_graph, pos, edge_labels=edge_labels, font_size=10)
            
            # Draw node labels with hexagram number and binary representation
            node_labels = {i: f"{i+1}\n{king_wen_sequence[i]}" for i in cycle_nodes}
            nx.draw_networkx_labels(cycle_graph, pos, labels=node_labels, font_size=9)
            
            plt.title(f"Cycle of {length} Hexagrams in King Wen Sequence")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    return G, cycles_by_length