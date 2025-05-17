import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from utils import (
    king_wen_sequence, hamming_distance,
    circular_distance, get_opposite_position,
    get_quarter_positions
)

def analyze_hierarchical_structure():
    """Analyze the hierarchical structure of the King Wen sequence."""
    print("\nRunning hierarchical structure analysis...")
    return analyze_hierarchical_structure_internal()

def analyze_hierarchical_structure_internal():
    """Analyze the hierarchical structure of the King Wen sequence, considering circularity."""
    
    # Calculate a similarity matrix based on transformations and circular distances
    n = len(king_wen_sequence)
    similarity_matrix = np.zeros((n, n))
    
    # Define transformations to consider
    def get_transformations(hex_a, hex_b):
        transformations = []
        
        # Simple transformations
        if hex_b == ''.join('1' if bit == '0' else '0' for bit in hex_a):
            transformations.append('inversion')
        if hex_b == hex_a[::-1]:
            transformations.append('reversal')
        if hex_b == hex_a[3:] + hex_a[:3]:
            transformations.append('trigram_swap')
            
        # Hamming distance
        h_dist = hamming_distance(hex_a, hex_b)
        if h_dist == 1:
            transformations.append('single_line')
        elif h_dist == 2:
            transformations.append('double_line')
            
        # Upper/lower trigram relationships
        if hex_a[:3] == hex_b[:3]:  # Same upper trigram
            transformations.append('same_upper')
        if hex_a[3:] == hex_b[3:]:  # Same lower trigram
            transformations.append('same_lower')
            
        return transformations
    
    # Fill the similarity matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity is maximum
                continue
                
            # Get all transformations between these hexagrams
            trans = get_transformations(king_wen_sequence[i], king_wen_sequence[j])
            
            # Calculate circular distance
            circ_dist = circular_distance(i, j)
            
            # Convert to a similarity score (more transformations = higher similarity)
            # Weight by circular distance (closer positions are more similar)
            base_similarity = len(trans) / 7.0  # Normalized by max possible transformations
            distance_factor = 1 - (circ_dist / (n/2))  # Normalize by max circular distance
            similarity_matrix[i, j] = base_similarity * (0.7 + 0.3 * distance_factor)
    
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Perform hierarchical clustering
    Z = linkage(distance_matrix, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(20, 10))
    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=8.,
        labels=[f"{i+1}" for i in range(n)]
    )
    plt.title('Hierarchical Clustering of King Wen Sequence (Circular)')
    plt.xlabel('Hexagram Number')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()
    
    # Identify clusters at different thresholds
    for num_clusters in [2, 4, 8, 16]:
        clusters = fcluster(Z, num_clusters, criterion='maxclust')
        
        print(f"\nCluster analysis with {num_clusters} clusters:")
        for cluster_id in range(1, num_clusters + 1):
            members = [i+1 for i, c in enumerate(clusters) if c == cluster_id]
            print(f"Cluster {cluster_id}: {members}")
        
        # Visualize clusters in circular order
        plt.figure(figsize=(15, 15))
        
        # Create circular layout
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 1.0
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        # Plot points colored by cluster
        colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))
        for i in range(n):
            cluster_id = clusters[i] - 1  # Convert to 0-based for indexing colors
            plt.scatter(x[i], y[i], color=colors[cluster_id], s=100)
            plt.text(x[i], y[i], str(i+1), fontsize=8, ha='center', va='center')
        
        # Highlight quarter positions
        quarter_positions = get_quarter_positions()
        quarter_x = [radius * np.cos(angles[i]) for i in quarter_positions]
        quarter_y = [radius * np.sin(angles[i]) for i in quarter_positions]
        plt.scatter(quarter_x, quarter_y, s=200, color='red', marker='*', label='Quarter Positions')
        
        plt.title(f'Circular Cluster Visualization ({num_clusters} Clusters)')
        plt.axis('equal')
        plt.axis('off')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Analyze the structure of transitions between clusters
    for num_clusters in [2, 4, 8]:
        clusters = fcluster(Z, num_clusters, criterion='maxclust')
        
        # Create a transition matrix between clusters (including wrap-around)
        transition_matrix = np.zeros((num_clusters, num_clusters))
        
        for i in range(n):
            from_cluster = clusters[i] - 1  # Convert to 0-based
            to_cluster = clusters[(i + 1) % n] - 1   # Convert to 0-based
            transition_matrix[from_cluster, to_cluster] += 1
        
        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.where(row_sums > 0, transition_matrix / row_sums, 0)
        
        # Visualize transition matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(transition_probs, annot=True, cmap='YlGnBu', fmt='.2f',
                    xticklabels=range(1, num_clusters+1),
                    yticklabels=range(1, num_clusters+1))
        plt.title(f'Circular Transition Probabilities Between {num_clusters} Clusters')
        plt.xlabel('To Cluster')
        plt.ylabel('From Cluster')
        plt.tight_layout()
        plt.show()
    
    # Analyze opposite positions in clusters
    print("\nAnalysis of opposite positions in clusters:")
    for num_clusters in [2, 4, 8]:
        clusters = fcluster(Z, num_clusters, criterion='maxclust')
        print(f"\nWith {num_clusters} clusters:")
        
        for i in range(n):
            opposite_pos = get_opposite_position(i)
            if clusters[i] == clusters[opposite_pos]:
                print(f"Hexagrams {i+1} and {opposite_pos+1} are in the same cluster ({clusters[i]})")
    
    return similarity_matrix, Z