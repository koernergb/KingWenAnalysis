import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from utils import king_wen_sequence, hamming_distance

def analyze_map_transformations():
    """Analyze the transformation patterns in the King Wen sequence."""
    print("\nRunning transformation mapping analysis...")
    map_transformation_patterns()

def map_transformation_patterns():
    # Define transformation types
    def identity(hex_str):
        return hex_str
    
    def invert(hex_str):
        return ''.join('1' if bit == '0' else '0' for bit in hex_str)
    
    def reverse(hex_str):
        return hex_str[::-1]
    
    def complement(hex_str):
        # Get the hexagram that sums with this one to 63 (in decimal)
        return bin(63 - int(hex_str, 2))[2:].zfill(6)
    
    # Special trigram operations
    def swap_trigrams(hex_str):
        # Swap upper and lower trigrams
        return hex_str[3:] + hex_str[:3]
    
    def transform_upper_trigram(hex_str, operation):
        # Apply operation to upper trigram only
        upper = hex_str[:3]
        lower = hex_str[3:]
        return operation(upper) + lower
    
    def transform_lower_trigram(hex_str, operation):
        # Apply operation to lower trigram only
        upper = hex_str[:3]
        lower = hex_str[3:]
        return upper + operation(lower)
    
    # Analyze each pair of consecutive hexagrams (circular)
    transformations = []
    n = len(king_wen_sequence)
    for i in range(n):
        current = king_wen_sequence[i]
        next_hex = king_wen_sequence[(i + 1) % n]  # Wrap around
        
        # Check for exact matches with our defined transformations
        if next_hex == invert(current):
            trans = "Inversion"
        elif next_hex == reverse(current):
            trans = "Reversal"
        elif next_hex == complement(current):
            trans = "Complement"
        elif next_hex == swap_trigrams(current):
            trans = "Trigram Swap"
        elif next_hex == transform_upper_trigram(current, invert):
            trans = "Upper Trigram Inversion"
        elif next_hex == transform_lower_trigram(current, invert):
            trans = "Lower Trigram Inversion"
        elif next_hex == transform_upper_trigram(current, reverse):
            trans = "Upper Trigram Reversal"
        elif next_hex == transform_lower_trigram(current, reverse):
            trans = "Lower Trigram Reversal"
        else:
            # Hamming distance if no exact match
            h_dist = hamming_distance(current, next_hex)
            if h_dist == 1:
                # Find which line changed
                for j in range(6):
                    if current[j] != next_hex[j]:
                        line_num = 6 - j  # Traditional line numbering from bottom to top
                        trans = f"Line {line_num} Change"
                        break
            elif h_dist == 2:
                trans = "Two Line Change"
            else:
                trans = f"Complex ({h_dist} bits)"
        
        transformations.append((i, (i+1)%n, trans, current, next_hex))
    
    # Create a visualization of the transformation sequence
    # First, count the frequency of each transformation type
    trans_types = [t[2] for t in transformations]
    trans_counts = pd.Series(trans_types).value_counts()
    
    # Create a DataFrame for easy manipulation
    df = pd.DataFrame(transformations, columns=['From', 'To', 'Transformation', 'From_Hex', 'To_Hex'])
    
    # Print summary
    print("Transformation Pattern in the King Wen Sequence:")
    print("===============================================")
    for i, row in df.iterrows():
        print(f"{row['From']+1:2d} → {row['To']+1:2d}: {row['Transformation']} ({row['From_Hex']} → {row['To_Hex']})")
    
    print("\nSummary of Transformation Types:")
    print("===============================")
    for trans, count in trans_counts.items():
        print(f"{trans}: {count} times ({count/len(transformations)*100:.1f}%)")
    
    # Look for repeating patterns in the transformation sequence
    pattern_length = 8  # Try to identify patterns of this length
    for start in range(len(transformations) - pattern_length + 1):
        pattern = trans_types[start:start+pattern_length]
        
        # Search for this pattern elsewhere in the sequence
        matches = []
        for i in range(len(transformations) - pattern_length + 1):
            if i != start:  # Don't match with itself
                if pattern == trans_types[i:i+pattern_length]:
                    matches.append(i)
        
        if matches:
            print(f"\nFound repeating pattern starting at position {start+1}:")
            print(f"Pattern: {pattern}")
            print(f"Also occurs at positions: {[m+1 for m in matches]}")
    
    # Visualize the transformation sequence
    plt.figure(figsize=(15, 8))
    
    # Create a categorical color map
    unique_trans = df['Transformation'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_trans)))
    color_map = {trans: colors[i] for i, trans in enumerate(unique_trans)}
    
    # Plot each transformation as a colored segment
    for i, row in df.iterrows():
        plt.plot([row['From'], row['To']], [0, 0], linewidth=10, 
                 color=color_map[row['Transformation']])
        
        # Add a text label for every 5th transformation
        if i % 5 == 0:
            plt.text(row['From'], 0.1, f"{row['From']+1}", 
                     ha='center', va='bottom', fontsize=9)
    
    # Create a legend
    legend_elements = [plt.Line2D([0], [0], color=color_map[trans], lw=4, label=trans) 
                       for trans in unique_trans]
    plt.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.title('Transformation Sequence in the King Wen Arrangement')
    plt.xlim(-1, len(king_wen_sequence))
    plt.ylim(-0.5, 0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Create a transformation network graph
    G = nx.DiGraph()
    
    # Add nodes (hexagrams)
    for i, hex_str in enumerate(king_wen_sequence):
        G.add_node(i, label=f"{i+1}", hexagram=hex_str)
    
    # Add edges (transformations)
    for from_idx, to_idx, trans, _, _ in transformations:
        G.add_edge(from_idx, to_idx, type=trans)
    
    # Visualize the graph
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightblue')
    
    # Draw edges with different colors based on transformation type
    edge_colors = [color_map[G.edges[edge]['type']] for edge in G.edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1.5, arrowsize=10)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Transformation Network of the King Wen Sequence')
    plt.axis('off')
    plt.tight_layout()
    plt.show()