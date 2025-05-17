import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from scipy.fft import fft
from sklearn.metrics import mutual_info_score
from matplotlib import patches
from matplotlib.lines import Line2D
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from utils import (
    king_wen_sequence, hamming_distance, get_opposite_position, get_quarter_positions,
    circular_distance, generate_random_sequence
)

# 1. Circular Visualization

def circular_visualization():
    """Visualize the King Wen sequence as a circle, color-coded by various properties and with relationship arcs."""
    n = len(king_wen_sequence)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Color by number of yang lines
    yang_counts = [h.count('1') for h in king_wen_sequence]
    plt.figure(figsize=(10, 10))
    sc = plt.scatter(x, y, c=yang_counts, cmap='coolwarm', s=120)
    plt.colorbar(sc, label='Number of Yang Lines')
    plt.title('King Wen Sequence: Circular Visualization (Yang Count)')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Color by upper trigram
    upper_trigrams = [h[:3] for h in king_wen_sequence]
    trigram_set = sorted(set(upper_trigrams))
    trigram_to_color = {t: i for i, t in enumerate(trigram_set)}
    colors = [trigram_to_color[t] for t in upper_trigrams]
    plt.figure(figsize=(10, 10))
    sc = plt.scatter(x, y, c=colors, cmap='tab10', s=120)
    plt.colorbar(sc, ticks=range(len(trigram_set)), label='Upper Trigram')
    plt.title('King Wen Sequence: Circular Visualization (Upper Trigram)')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Draw lines to opposites
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, color='gray', s=80)
    for i in range(n):
        opp = get_opposite_position(i)
        plt.plot([x[i], x[opp]], [y[i], y[opp]], 'r-', alpha=0.3)
    plt.title('King Wen Sequence: Opposite Connections')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Draw lines for fixed circular distances (e.g., 8, 16)
    for dist in [8, 16]:
        plt.figure(figsize=(10, 10))
        plt.scatter(x, y, color='gray', s=80)
        for i in range(n):
            j = (i + dist) % n
            plt.plot([x[i], x[j]], [y[i], y[j]], 'b-', alpha=0.2)
        plt.title(f'King Wen Sequence: Connections at Circular Distance {dist}')
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# 2. Circular Distance Analysis

def circular_distance_analysis():
    """Plot average Hamming distance and mutual information as a function of circular distance."""
    n = len(king_wen_sequence)
    max_dist = n // 2
    avg_hamming = []
    avg_mi = []
    seq_ints = [int(h, 2) for h in king_wen_sequence]
    for d in range(1, max_dist + 1):
        hamming_dists = [hamming_distance(king_wen_sequence[i], king_wen_sequence[(i + d) % n]) for i in range(n)]
        avg_hamming.append(np.mean(hamming_dists))
        seq2 = [seq_ints[(i + d) % n] for i in range(n)]
        avg_mi.append(mutual_info_score(seq_ints, seq2))
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, max_dist + 1), avg_hamming, label='Avg Hamming Distance')
    plt.xlabel('Circular Distance')
    plt.ylabel('Average Hamming Distance')
    plt.title('Average Hamming Distance vs Circular Distance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, max_dist + 1), avg_mi, label='Mutual Information')
    plt.xlabel('Circular Distance')
    plt.ylabel('Mutual Information')
    plt.title('Mutual Information vs Circular Distance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 3. Circular Symmetry Analysis

def circular_symmetry_analysis():
    """Fourier analysis and detection of rotational symmetries in the King Wen sequence."""
    n = len(king_wen_sequence)
    yang_counts = np.array([h.count('1') for h in king_wen_sequence])
    # Fourier transform
    fft_vals = np.abs(fft(yang_counts - np.mean(yang_counts)))
    freqs = np.fft.fftfreq(n)
    plt.figure(figsize=(12, 6))
    plt.stem(freqs[:n//2], fft_vals[:n//2])
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Fourier Spectrum of Yang Line Count (Circular)')
    plt.tight_layout()
    plt.show()
    # Highlight dominant periodicities
    dominant_freqs = np.argsort(fft_vals[1:n//2])[::-1][:5]
    print("Dominant rotational periodicities (as fraction of circle):")
    for idx in dominant_freqs:
        print(f"  Frequency {freqs[idx]:.3f} (Period {1/freqs[idx]:.1f} positions), Amplitude {fft_vals[idx]:.2f}")

# 4. Circular Networks

def circular_networks():
    """Visualize the transformation network on the circle, with chord diagrams for transformation types."""
    n = len(king_wen_sequence)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    # Build network for single-line changes
    plt.figure(figsize=(12, 12))
    plt.scatter(x, y, color='gray', s=80)
    for i in range(n):
        for j in range(n):
            if i != j and hamming_distance(king_wen_sequence[i], king_wen_sequence[j]) == 1:
                plt.plot([x[i], x[j]], [y[i], y[j]], 'g-', alpha=0.1)
    plt.title('Chord Diagram: Single-Line Changes')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # Highlight clusters (e.g., by upper trigram)
    upper_trigrams = [h[:3] for h in king_wen_sequence]
    trigram_set = sorted(set(upper_trigrams))
    trigram_to_color = {t: i for i, t in enumerate(trigram_set)}
    colors = [trigram_to_color[t] for t in upper_trigrams]
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, c=colors, cmap='tab10', s=120)
    plt.title('Clusters by Upper Trigram')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 5. Circular Transition Entropy

def circular_transition_entropy():
    """Plot transition entropy as a function of circular distance, compared to random sequences."""
    n = len(king_wen_sequence)
    max_dist = n // 2
    window_size = 3
    def entropy_for_distance(seq, d):
        windows = [[seq[(i + j * d) % n] for j in range(window_size)] for i in range(n)]
        window_ints = [int(''.join(w), 2) if isinstance(w, list) else int(w, 2) for w in windows]
        values, counts = np.unique(window_ints, return_counts=True)
        probabilities = counts / len(window_ints)
        return -np.sum(probabilities * np.log2(probabilities))
    entropies = [entropy_for_distance(king_wen_sequence, d) for d in range(1, max_dist + 1)]
    # Random baseline
    random_entropies = []
    for _ in range(50):
        rand_seq = generate_random_sequence(n)
        random_entropies.append([entropy_for_distance(rand_seq, d) for d in range(1, max_dist + 1)])
    avg_random_entropy = np.mean(random_entropies, axis=0)
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, max_dist + 1), entropies, label='King Wen Sequence')
    plt.plot(range(1, max_dist + 1), avg_random_entropy, 'r--', label='Random Sequences')
    plt.xlabel('Circular Distance')
    plt.ylabel('Transition Entropy (bits)')
    plt.title('Transition Entropy vs Circular Distance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 6. Circular Opposite Analysis

def circular_opposite_analysis():
    """Analyze and visualize relationships between hexagrams and their opposites on the circle."""
    n = len(king_wen_sequence)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    hamming_opposites = [hamming_distance(king_wen_sequence[i], king_wen_sequence[get_opposite_position(i)]) for i in range(n)]
    plt.figure(figsize=(10, 10))
    sc = plt.scatter(x, y, c=hamming_opposites, cmap='viridis', s=120)
    plt.colorbar(sc, label='Hamming Distance to Opposite')
    for i in range(n):
        opp = get_opposite_position(i)
        plt.plot([x[i], x[opp]], [y[i], y[opp]], 'r-', alpha=0.2)
    plt.title('Hamming Distance to Opposite Hexagram')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(hamming_opposites, bins=range(7), align='left', rwidth=0.8)
    plt.xlabel('Hamming Distance to Opposite')
    plt.ylabel('Count')
    plt.title('Distribution of Hamming Distances to Opposite Hexagram')
    plt.tight_layout()
    plt.show()

# 7. Advanced Circular Analysis Methods

def circular_autocorrelation(sequence=king_wen_sequence, max_lag=32):
    """Calculate autocorrelation with circular distance."""
    n = len(sequence)
    sequence_ints = np.array([int(x, 2) for x in sequence])
    
    circular_autocorr = []
    for lag in range(max_lag+1):
        # Calculate correlation with circular wrapping
        shifted = np.roll(sequence_ints, -lag)
        # Use numpy's correlation coefficient function
        correlation = np.corrcoef(sequence_ints, shifted)[0,1]
        circular_autocorr.append(correlation)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(range(max_lag+1), circular_autocorr, 'b-', linewidth=2)
    plt.title('Circular Autocorrelation of King Wen Sequence')
    plt.xlabel('Lag (Circular Distance)')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.show()
    
    return circular_autocorr

def circular_transformation_map():
    """Create a circular visualization of transformation types."""
    n = len(king_wen_sequence)
    
    # Identify transformation type between each hexagram and next (with wrap-around)
    transformations = []
    for i in range(n):
        current = king_wen_sequence[i]
        next_idx = (i + 1) % n  # Circular indexing
        next_hex = king_wen_sequence[next_idx]
        
        # Determine transformation type
        if next_hex == ''.join('1' if bit == '0' else '0' for bit in current):
            trans_type = "Inversion"
        elif next_hex == current[::-1]:
            trans_type = "Reversal"
        elif next_hex == current[3:] + current[:3]:
            trans_type = "Trigram Swap"
        else:
            # Calculate Hamming distance
            h_dist = sum(c1 != c2 for c1, c2 in zip(current, next_hex))
            if h_dist == 1:
                trans_type = "Single Line"
            elif h_dist == 2:
                trans_type = "Two Line"
            else:
                trans_type = f"Complex ({h_dist} bits)"
        
        transformations.append(trans_type)
    
    # Define colors for transformation types
    colors = {
        "Inversion": "blue",
        "Reversal": "green",
        "Trigram Swap": "purple",
        "Single Line": "cyan",
        "Two Line": "pink"
    }
    # Default for complex transformations
    for i in range(3, 7):
        colors[f"Complex ({i} bits)"] = f"orangered"
    
    # Create circular plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': 'polar'})
    
    # Plot points on circle
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    # Draw transformation arcs
    for i in range(n):
        start_angle = theta[i]
        end_angle = theta[(i+1) % n]
        
        trans_type = transformations[i]
        color = colors.get(trans_type, "gray")
        
        # Draw arc
        arc = patches.Arc((0, 0), 2, 2, 
                         theta1=np.degrees(start_angle), 
                         theta2=np.degrees(end_angle),
                         color=color, linewidth=2.5)
        ax.add_patch(arc)
        
        # Add hexagram number at each point
        ax.text(start_angle, 1.1, str(i+1), 
                ha='center', va='center', fontsize=9)
    
    # Add a legend
    legend_elements = [Line2D([0], [0], color=color, lw=2, label=trans)
                      for trans, color in colors.items()]
    ax.legend(handles=legend_elements, loc='center')
    
    plt.title('Circular Transformation Map of King Wen Sequence')
    ax.set_rticks([])  # Remove radial ticks
    ax.grid(False)
    plt.tight_layout()
    plt.show()

def circular_similarity_plot():
    """Create a similarity plot using circular distance."""
    n = len(king_wen_sequence)
    similarity_matrix = np.zeros((n, n))
    
    # Calculate Hamming distance with circular distance consideration
    for i in range(n):
        for j in range(n):
            # Calculate circular distance
            circular_dist = min(abs(i-j), n-abs(i-j))
            
            # Calculate similarity based on both hexagram structure and circular distance
            hex_i = king_wen_sequence[i]
            hex_j = king_wen_sequence[j]
            
            # Hamming similarity (6 - Hamming distance)
            ham_similarity = 6 - sum(c1 != c2 for c1, c2 in zip(hex_i, hex_j))
            
            # Combine with circular distance effect
            similarity_matrix[i, j] = ham_similarity / (1 + 0.1*circular_dist)
    
    # Plot the circular similarity matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(similarity_matrix, cmap='viridis', aspect='equal')
    plt.colorbar(label='Similarity Score')
    plt.title('Circular Similarity Plot of King Wen Sequence')
    plt.xlabel('Hexagram Position')
    plt.ylabel('Hexagram Position')
    plt.tight_layout()
    plt.show()

def circular_network_embedding():
    """Create a network visualization with circular layout."""
    G = nx.DiGraph()
    
    # Add nodes
    n = len(king_wen_sequence)
    for i in range(n):
        G.add_node(i, label=f"{i+1}", hexagram=king_wen_sequence[i])
    
    # Add edges for sequence order (with wrap-around)
    for i in range(n):
        next_i = (i + 1) % n  # Circular indexing
        G.add_edge(i, next_i, type="sequence")
    
    # Add edges for transformations
    for i in range(n):
        for j in range(n):
            if i != j and (i, j) not in G.edges():
                hex_i = king_wen_sequence[i]
                hex_j = king_wen_sequence[j]
                
                # Check for transformation relationships
                if hex_j == ''.join('1' if bit == '0' else '0' for bit in hex_i):
                    G.add_edge(i, j, type="inversion")
                elif hex_j == hex_i[::-1]:
                    G.add_edge(i, j, type="reversal")
                elif hex_j == hex_i[3:] + hex_i[:3]:
                    G.add_edge(i, j, type="trigram_swap")
    
    # Create a circular layout
    pos = nx.circular_layout(G)
    
    # Define edge colors
    edge_colors = {
        "sequence": "black",
        "inversion": "blue",
        "reversal": "green",
        "trigram_swap": "purple"
    }
    
    plt.figure(figsize=(14, 14))
    
    # Draw nodes in circular arrangement
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue')
    
    # Draw different edge types
    for edge_type, color in edge_colors.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, 
                              width=1.5, alpha=0.7, arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # Add a legend
    legend_elements = [Line2D([0], [0], color=color, lw=2, label=edge_type)
                      for edge_type, color in edge_colors.items()]
    plt.legend(handles=legend_elements)
    
    plt.title('Circular Network Embedding of King Wen Sequence')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def fourier_analysis():
    """Apply Fourier analysis to identify dominant frequencies."""
    # Extract properties to analyze
    yang_count = [king_wen_sequence[i].count('1') for i in range(len(king_wen_sequence))]
    upper_trigram_value = [int(king_wen_sequence[i][:3], 2) for i in range(len(king_wen_sequence))]
    lower_trigram_value = [int(king_wen_sequence[i][3:], 2) for i in range(len(king_wen_sequence))]
    
    # Apply Fourier transform
    yang_fft = np.abs(np.fft.fft(yang_count))
    upper_fft = np.abs(np.fft.fft(upper_trigram_value))
    lower_fft = np.abs(np.fft.fft(lower_trigram_value))
    
    # Get frequencies
    n = len(king_wen_sequence)
    freqs = np.fft.fftfreq(n)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(freqs[:n//2], yang_fft[:n//2], 'b-')
    plt.title('Fourier Analysis of Yang Line Count')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(freqs[:n//2], upper_fft[:n//2], 'g-')
    plt.title('Fourier Analysis of Upper Trigram')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(freqs[:n//2], lower_fft[:n//2], 'r-')
    plt.title('Fourier Analysis of Lower Trigram')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def circular_community_analysis():
    """Detect communities with circular distance consideration."""
    # Create distance matrix based on circular distance and transformation similarity
    n = len(king_wen_sequence)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            # Calculate circular distance
            circ_dist = min(abs(i-j), n-abs(i-j))
            
            # Calculate transformation distance
            hex_i = king_wen_sequence[i]
            hex_j = king_wen_sequence[j]
            
            # Check for transformation relationships
            if hex_j == ''.join('1' if bit == '0' else '0' for bit in hex_i):
                trans_dist = 1  # Inversion
            elif hex_j == hex_i[::-1]:
                trans_dist = 1  # Reversal
            elif hex_j == hex_i[3:] + hex_i[:3]:
                trans_dist = 1  # Trigram swap
            else:
                # Use Hamming distance
                trans_dist = sum(c1 != c2 for c1, c2 in zip(hex_i, hex_j))
            
            # Combine distances
            distance_matrix[i, j] = trans_dist + 0.1 * circ_dist
    
    # Perform hierarchical clustering
    Z = linkage(squareform(distance_matrix), method='ward')
    
    # Detect communities at different levels
    for num_clusters in [2, 4, 8, 16]:
        clusters = fcluster(Z, num_clusters, criterion='maxclust')
        
        # Create circular visualization of clusters
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # Calculate positions
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        radii = [1] * n
        
        # Assign colors to clusters
        colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))
        node_colors = [colors[cluster-1] for cluster in clusters]
        
        # Draw points
        ax.scatter(theta, radii, c=node_colors, s=100)
        
        # Add labels
        for i in range(n):
            ax.text(theta[i], 1.1, str(i+1), 
                    ha='center', va='center', fontsize=8)
        
        plt.title(f'Circular Community Detection ({num_clusters} Clusters)')
        ax.set_rticks([])  # Remove radial ticks
        ax.grid(False)
        plt.tight_layout()
        plt.show()

def diametric_symmetry_analysis():
    """Analyze symmetry between points on opposite sides of the circle."""
    n = len(king_wen_sequence)
    half_n = n // 2
    
    # Calculate relationship strength between diametrically opposed hexagrams
    diametric_relationships = []
    
    for i in range(half_n):
        opposite_i = (i + half_n) % n
        
        hex_i = king_wen_sequence[i]
        hex_opposite = king_wen_sequence[opposite_i]
        
        # Check for special relationships
        if hex_opposite == ''.join('1' if bit == '0' else '0' for bit in hex_i):
            relationship = "Inversion"
        elif hex_opposite == hex_i[::-1]:
            relationship = "Reversal"
        elif hex_opposite == hex_i[3:] + hex_i[:3]:
            relationship = "Trigram Swap"
        else:
            h_dist = sum(c1 != c2 for c1, c2 in zip(hex_i, hex_opposite))
            relationship = f"Hamming Distance {h_dist}"
        
        diametric_relationships.append((i+1, opposite_i+1, relationship))
    
    # Count relationship types
    relationship_counts = Counter([r[2] for r in diametric_relationships])
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(relationship_counts.keys(), relationship_counts.values())
    plt.title('Relationships Between Diametrically Opposed Hexagrams')
    plt.xlabel('Relationship Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return diametric_relationships

# Update the entrypoint to include all analyses
def run_all_circular_analyses():
    print("\n--- Circular Visualization ---")
    circular_visualization()
    print("\n--- Circular Distance Analysis ---")
    circular_distance_analysis()
    print("\n--- Circular Symmetry Analysis ---")
    circular_symmetry_analysis()
    print("\n--- Circular Networks ---")
    circular_networks()
    print("\n--- Circular Transition Entropy ---")
    circular_transition_entropy()
    print("\n--- Circular Opposite Analysis ---")
    circular_opposite_analysis()
    print("\n--- Circular Autocorrelation ---")
    circular_autocorrelation()
    print("\n--- Circular Transformation Map ---")
    circular_transformation_map()
    print("\n--- Circular Similarity Plot ---")
    circular_similarity_plot()
    print("\n--- Circular Network Embedding ---")
    circular_network_embedding()
    print("\n--- Fourier Analysis ---")
    fourier_analysis()
    print("\n--- Circular Community Analysis ---")
    circular_community_analysis()
    print("\n--- Diametric Symmetry Analysis ---")
    diametric_symmetry_analysis()