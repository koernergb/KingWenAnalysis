import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
import pandas as pd
from itertools import combinations

def binary_to_coordinates(binary_string):
    """Convert a binary string to 6D coordinates (0,1) for hypercube visualization."""
    return [int(bit) for bit in binary_string]

def hamming_distance(str1, str2):
    """Calculate the Hamming distance between two binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def create_hypercube_graph():
    """Create a networkx graph representing the 6D hypercube."""
    G = nx.Graph()
    
    # Add all 64 vertices (hexagrams)
    for i in range(64):
        binary = format(i, '06b')
        G.add_node(i, binary=binary, pos=binary_to_coordinates(binary))
    
    # Add edges between vertices that differ by exactly one bit (Hamming distance = 1)
    for i in range(64):
        for j in range(i+1, 64):
            binary_i = format(i, '06b')
            binary_j = format(j, '06b')
            if hamming_distance(binary_i, binary_j) == 1:
                G.add_edge(i, j)
    
    return G

def king_wen_to_binary(king_wen_sequence):
    """Convert a list of King Wen hexagrams (as binary strings or integers) to binary representations."""
    binary_sequence = []
    for hexagram in king_wen_sequence:
        if isinstance(hexagram, str):
            # If already a binary string, convert to int
            index = int(hexagram, 2)
        elif isinstance(hexagram, int):
            # If integer, use as is (assume 1-indexed)
            index = hexagram - 1
        else:
            raise ValueError(f"Unexpected hexagram type: {type(hexagram)}")
        binary = format(index, '06b')
        binary_sequence.append(binary)
    return binary_sequence

def analyze_king_wen_path(king_wen_sequence, hypercube_graph):
    """Analyze how the King Wen sequence traverses the hypercube, treating the sequence as circular."""
    binary_sequence = king_wen_to_binary(king_wen_sequence)
    # Create a path graph representing the King Wen sequence
    path_graph = nx.DiGraph()
    for i in range(len(binary_sequence)):
        hex_num = king_wen_sequence[i]
        binary = binary_sequence[i]
        path_graph.add_node(hex_num, binary=binary, pos=i)
    # Add edges between consecutive hexagrams in the sequence, including last-to-first
    for i in range(len(king_wen_sequence)):
        hex1 = king_wen_sequence[i]
        hex2 = king_wen_sequence[(i+1) % len(king_wen_sequence)]  # Wrap around
        bin1 = binary_sequence[i]
        bin2 = binary_sequence[(i+1) % len(binary_sequence)]
        h_dist = hamming_distance(bin1, bin2)
        path_graph.add_edge(hex1, hex2, hamming=h_dist)
    # Calculate statistics about the path
    hamming_distances = [hamming_distance(binary_sequence[i], binary_sequence[(i+1) % len(binary_sequence)])
                         for i in range(len(binary_sequence))]
    edge_types = {}
    for dist in range(1, 7):
        edge_types[dist] = hamming_distances.count(dist)
    # Calculate how many hypercube edges the King Wen sequence traverses
    hypercube_edges_used = sum(1 for i in range(len(king_wen_sequence))
                              if hamming_distances[i] == 1)
    hypercube_edge_percentage = hypercube_edges_used / len(king_wen_sequence) * 100
    return {
        'path_graph': path_graph,
        'hamming_distances': hamming_distances,
        'edge_types': edge_types,
        'hypercube_edges_used': hypercube_edges_used,
        'hypercube_edge_percentage': hypercube_edge_percentage
    }

def project_hypercube_to_3d(hypercube_graph, king_wen_sequence=None):
    """Project the 6D hypercube to 3D for visualization."""
    # Extract 6D coordinates
    coords_6d = np.array([hypercube_graph.nodes[i]['pos'] for i in range(64)])
    
    # Use PCA to reduce to 3D
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(coords_6d)
    
    # Store 3D coordinates
    for i, coords in enumerate(coords_3d):
        hypercube_graph.nodes[i]['pos_3d'] = coords
    
    return coords_3d, pca.explained_variance_ratio_

def visualize_hypercube_and_king_wen(hypercube_graph, king_wen_sequence, coords_3d=None):
    """Visualize the 6D hypercube in 3D with the King Wen sequence highlighted."""
    if coords_3d is None:
        coords_3d, _ = project_hypercube_to_3d(hypercube_graph)
    
    # Convert King Wen sequence to 0-indexed for the graph
    king_wen_0indexed = [hwn - 1 for hwn in king_wen_sequence]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot hypercube vertices
    x = coords_3d[:, 0]
    y = coords_3d[:, 1]
    z = coords_3d[:, 2]
    
    # Color points based on whether they're in the King Wen sequence
    colors = ['blue' if i in king_wen_0indexed else 'lightgray' for i in range(64)]
    sizes = [50 if i in king_wen_0indexed else 20 for i in range(64)]
    
    ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.8)
    
    # Plot hypercube edges
    for i, j in hypercube_graph.edges():
        xi, yi, zi = coords_3d[i]
        xj, yj, zj = coords_3d[j]
        ax.plot([xi, xj], [yi, yj], [zi, zj], 'gray', alpha=0.2)
    
    # Plot King Wen sequence path
    for i in range(len(king_wen_sequence) - 1):
        idx1 = king_wen_sequence[i] - 1  # Convert to 0-indexed
        idx2 = king_wen_sequence[i+1] - 1
        xi, yi, zi = coords_3d[idx1]
        xj, yj, zj = coords_3d[idx2]
        ax.plot([xi, xj], [yi, yj], [zi, zj], 'red', linewidth=2, alpha=0.7)
    
    # Add labels for some key hexagrams
    for i, hw in enumerate(king_wen_sequence):
        if i % 8 == 0:  # Label every 8th hexagram for clarity
            idx = hw - 1
            ax.text(coords_3d[idx, 0], coords_3d[idx, 1], coords_3d[idx, 2], 
                   str(hw), color='black', fontsize=10)
    
    ax.set_title('King Wen Sequence as Path Through 6D Hypercube (3D Projection)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_king_wen_hypercube_path_efficiency():
    """Analyze the efficiency of the King Wen sequence as a path through the hypercube."""
    # Create all possible pairs of vertices in the hypercube
    all_pairs = list(combinations(range(64), 2))
    
    # Calculate Hamming distance for each pair
    hamming_distances = [hamming_distance(format(i, '06b'), format(j, '06b')) for i, j in all_pairs]
    
    # Distribution of Hamming distances in the hypercube
    hamming_dist_counts = {}
    for d in range(1, 7):
        hamming_dist_counts[d] = hamming_distances.count(d)
    
    # Calculate expected vs. actual distribution in King Wen sequence
    total_pairs = len(all_pairs)
    expected_distribution = {d: count/total_pairs for d, count in hamming_dist_counts.items()}
    
    # Compare with actual King Wen sequence
    # This would use the edge_types from analyze_king_wen_path()
    
    return hamming_dist_counts, expected_distribution

def analyze_triadic_structure_in_hypercube(king_wen_sequence):
    """Analyze how triadic groups in the King Wen sequence relate to hypercube geometry, treating the sequence as circular."""
    binary_sequence = king_wen_to_binary(king_wen_sequence)
    triads = []
    n = len(binary_sequence)
    for i in range(0, n, 3):
        thesis = binary_sequence[i % n]
        antithesis = binary_sequence[(i+1) % n]
        synthesis = binary_sequence[(i+2) % n]
        d_t_a = hamming_distance(thesis, antithesis)
        d_t_s = hamming_distance(thesis, synthesis)
        d_a_s = hamming_distance(antithesis, synthesis)
        thesis_coords = binary_to_coordinates(thesis)
        antithesis_coords = binary_to_coordinates(antithesis)
        synthesis_coords = binary_to_coordinates(synthesis)
        path_ratio = sum(abs((t + a)/2 - s) for t, a, s in zip(thesis_coords, antithesis_coords, synthesis_coords)) / 6
        triads.append({
            'triad_num': i//3 + 1,
            'thesis': thesis,
            'antithesis': antithesis,
            'synthesis': synthesis,
            'hamming_t_a': d_t_a,
            'hamming_t_s': d_t_s,
            'hamming_a_s': d_a_s,
            'path_ratio': path_ratio
        })
    return triads

def visualize_triads_in_hypercube(triads, coords_3d):
    """Visualize triadic relationships in the 3D projection of the hypercube."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a mapping from binary string to index
    binary_to_idx = {format(i, '06b'): i for i in range(64)}
    
    # Plot each triad with distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(triads)))
    
    for i, triad in enumerate(triads):
        # Get indices
        t_idx = binary_to_idx[triad['thesis']]
        a_idx = binary_to_idx[triad['antithesis']]
        s_idx = binary_to_idx[triad['synthesis']]
        
        # Get 3D coordinates
        t_coords = coords_3d[t_idx]
        a_coords = coords_3d[a_idx]
        s_coords = coords_3d[s_idx]
        
        # Plot points
        ax.scatter(t_coords[0], t_coords[1], t_coords[2], color=colors[i], s=100, label=f"Triad {triad['triad_num']}")
        ax.scatter(a_coords[0], a_coords[1], a_coords[2], color=colors[i], s=100)
        ax.scatter(s_coords[0], s_coords[1], s_coords[2], color=colors[i], s=100)
        
        # Plot connecting lines
        ax.plot([t_coords[0], a_coords[0]], [t_coords[1], a_coords[1]], [t_coords[2], a_coords[2]], 
                color=colors[i], linestyle='--', alpha=0.7)
        ax.plot([t_coords[0], s_coords[0]], [t_coords[1], s_coords[1]], [t_coords[2], s_coords[2]], 
                color=colors[i], alpha=0.7)
        ax.plot([a_coords[0], s_coords[0]], [a_coords[1], s_coords[1]], [a_coords[2], s_coords[2]], 
                color=colors[i], alpha=0.7)
        
        # Add labels
        ax.text(t_coords[0], t_coords[1], t_coords[2], f"T{triad['triad_num']}", fontsize=8)
        ax.text(a_coords[0], a_coords[1], a_coords[2], f"A{triad['triad_num']}", fontsize=8)
        ax.text(s_coords[0], s_coords[1], s_coords[2], f"S{triad['triad_num']}", fontsize=8)
    
    # Add a legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
    
    ax.set_title('Triadic Relationships in 6D Hypercube (3D Projection)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def analyze_subspace_structure(king_wen_sequence):
    """Analyze how the King Wen sequence traverses different subspaces of the 6D hypercube."""
    binary_sequence = king_wen_to_binary(king_wen_sequence)
    
    # Look at how the sequence traverses 2D, 3D, and 4D subspaces
    subspace_traversals = {}
    
    for dim in [2, 3, 4]:
        # Generate all possible dim-sized subsets of the 6 dimensions
        subspaces = list(combinations(range(6), dim))
        
        subspace_data = {}
        for subspace in subspaces:
            # Project the sequence onto this subspace
            projected_sequence = []
            for binary in binary_sequence:
                projected = ''.join(binary[i] for i in subspace)
                projected_sequence.append(projected)
            
            # Count how many unique vertices are visited in this subspace
            unique_vertices = len(set(projected_sequence))
            
            # Calculate efficiency of traversal
            repeat_visits = len(projected_sequence) - unique_vertices
            
            subspace_data[subspace] = {
                'unique_vertices': unique_vertices,
                'max_possible': 2**dim,
                'coverage': unique_vertices / (2**dim),
                'repeat_visits': repeat_visits
            }
        
        subspace_traversals[dim] = subspace_data
    
    return subspace_traversals

def analyze_distance_matrix_in_hypercube(king_wen_sequence):
    """Analyze the distance matrix of the King Wen sequence in the 6D hypercube."""
    binary_sequence = king_wen_to_binary(king_wen_sequence)
    
    # Calculate Hamming distance matrix
    n = len(binary_sequence)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = hamming_distance(binary_sequence[i], binary_sequence[j])
    
    # Calculate various properties of the distance matrix
    avg_distance = np.mean(distance_matrix)
    max_distance = np.max(distance_matrix)
    min_distance = np.min(distance_matrix[distance_matrix > 0])  # Minimum non-zero distance
    
    # Calculate distribution of distances
    unique_distances = np.unique(distance_matrix)
    distance_counts = {int(d): np.sum(distance_matrix == d) for d in unique_distances if d > 0}
    
    return {
        'distance_matrix': distance_matrix,
        'avg_distance': avg_distance,
        'max_distance': max_distance,
        'min_distance': min_distance,
        'distance_distribution': distance_counts
    }

def run_hypercube_analysis(king_wen_sequence):
    """Run comprehensive hypercube analysis on the King Wen sequence."""
    # Create hypercube graph
    hypercube = create_hypercube_graph()
    
    # Analyze how King Wen traverses the hypercube
    path_analysis = analyze_king_wen_path(king_wen_sequence, hypercube)
    print("King Wen Path Analysis:")
    print(f"Hypercube edges used: {path_analysis['hypercube_edges_used']} ({path_analysis['hypercube_edge_percentage']:.2f}%)")
    print("Edge types distribution:")
    for dist, count in path_analysis['edge_types'].items():
        print(f"  Hamming distance {dist}: {count} edges ({count/(len(king_wen_sequence)-1)*100:.2f}%)")
    
    # Project to 3D
    coords_3d, variance_explained = project_hypercube_to_3d(hypercube)
    print("\nPCA Projection Quality:")
    print(f"Variance explained by 3 PCs: {sum(variance_explained)*100:.2f}%")
    
    # Analyze triadic structure
    triads = analyze_triadic_structure_in_hypercube(king_wen_sequence)
    print("\nTriadic Structure in Hypercube:")
    for triad in triads[:5]:  # Show first 5 triads
        print(f"Triad {triad['triad_num']}:")
        print(f"  Hamming distances: T↔A={triad['hamming_t_a']}, T↔S={triad['hamming_t_s']}, A↔S={triad['hamming_a_s']}")
        print(f"  Path ratio: {triad['path_ratio']:.4f}")
    
    # Analyze subspace traversal
    subspace_analysis = analyze_subspace_structure(king_wen_sequence)
    print("\nSubspace Analysis:")
    for dim, data in subspace_analysis.items():
        avg_coverage = sum(d['coverage'] for d in data.values()) / len(data)
        print(f"{dim}D subspaces: Avg coverage {avg_coverage*100:.2f}%")
    
    # Analyze distance matrix
    distance_analysis = analyze_distance_matrix_in_hypercube(king_wen_sequence)
    print("\nDistance Matrix Analysis:")
    print(f"Average Hamming distance: {distance_analysis['avg_distance']:.4f}")
    print(f"Distance distribution: {distance_analysis['distance_distribution']}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    fig1 = visualize_hypercube_and_king_wen(hypercube, king_wen_sequence, coords_3d)
    fig2 = visualize_triads_in_hypercube(triads[:7], coords_3d)  # Show first 7 triads
    
    return {
        'hypercube': hypercube,
        'path_analysis': path_analysis,
        'coords_3d': coords_3d,
        'triads': triads,
        'subspace_analysis': subspace_analysis,
        'distance_analysis': distance_analysis,
        'figures': [fig1, fig2]
    }

# Example King Wen sequence (1-indexed) - Replace with actual sequence
king_wen_sequence = list(range(1, 65))  # Placeholder - use actual King Wen sequence

# Run analysis
results = run_hypercube_analysis(king_wen_sequence)

# At the bottom, ensure run_hypercube_analysis is only called with a list of integers (1-64)
# If king_wen_sequence is a list of binary strings, convert to 1-64 integers for this call
if __name__ == "__main__":
    # Try to detect type and convert if needed
    from utils import king_wen_sequence as kw_seq
    if isinstance(kw_seq[0], str):
        # Convert binary strings to 1-64 indices
        kw_seq_int = [int(h, 2) + 1 for h in kw_seq]
    else:
        kw_seq_int = kw_seq
    results = run_hypercube_analysis(kw_seq_int)