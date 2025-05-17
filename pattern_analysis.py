import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mutual_info_score
from utils import (
    king_wen_sequence,
    hamming_distance, generate_random_sequence,
    invert, reverse, complementary, circular_distance,
    get_opposite_position, get_quarter_positions
)

def analyze_patterns():
    """Analyze patterns and transformations within the King Wen sequence, considering circularity."""
    print("\nRunning pattern analysis...")
    return analyze_patterns_in_king_wen()

def analyze_patterns_in_king_wen():
    """Analyze patterns and transformations within the King Wen sequence, considering circularity."""
    n = len(king_wen_sequence)
    
    # Check for pattern relationships between consecutive hexagrams (including wrap-around)
    relationships = []
    for i in range(n):
        current = king_wen_sequence[i]
        next_hex = king_wen_sequence[(i + 1) % n]  # Wrap around
        
        # Check for various relationships
        if next_hex == invert(current):
            rel = "Inversion"
        elif next_hex == reverse(current):
            rel = "Reversal"
        elif next_hex == complementary(current):
            rel = "Complementary"
        elif hamming_distance(current, next_hex) == 1:
            rel = "Single Line Change"
        elif hamming_distance(current, next_hex) == 2:
            rel = "Two Line Change"
        else:
            rel = "Other"
        
        relationships.append(rel)
    
    # Count the frequency of each relationship
    relationship_counts = pd.Series(relationships).value_counts()
    total = len(relationships)
    relationship_pcts = relationship_counts / total * 100
    
    print("Circular relationships between consecutive hexagrams in the King Wen sequence:")
    for rel, count in relationship_counts.items():
        print(f"{rel}: {count} ({relationship_pcts[rel]:.1f}%)")
    
    # Compare with expected frequencies in random sequences
    random_relationships = []
    for _ in range(1000):
        random_seq = generate_random_sequence(n)
        for i in range(n):
            current = random_seq[i]
            next_hex = random_seq[(i + 1) % n]  # Wrap around
            
            if next_hex == invert(current):
                rel = "Inversion"
            elif next_hex == reverse(current):
                rel = "Reversal"
            elif next_hex == complementary(current):
                rel = "Complementary"
            elif hamming_distance(current, next_hex) == 1:
                rel = "Single Line Change"
            elif hamming_distance(current, next_hex) == 2:
                rel = "Two Line Change"
            else:
                rel = "Other"
            
            random_relationships.append(rel)
    
    random_rel_counts = pd.Series(random_relationships).value_counts()
    random_rel_pcts = random_rel_counts / len(random_relationships) * 100
    
    print("\nExpected relationships in random sequences:")
    for rel in relationship_counts.index:
        if rel in random_rel_counts:
            print(f"{rel}: {random_rel_pcts[rel]:.1f}%")
        else:
            print(f"{rel}: 0.0%")
    
    # Plot the comparison
    plt.figure(figsize=(12, 6))
    
    rels = list(set(list(relationship_counts.index) + list(random_rel_counts.index)))
    
    kw_values = [relationship_pcts[rel] if rel in relationship_pcts else 0 for rel in rels]
    random_values = [random_rel_pcts[rel] if rel in random_rel_pcts else 0 for rel in rels]
    
    x = np.arange(len(rels))
    width = 0.35
    
    plt.bar(x - width/2, kw_values, width, label='King Wen Sequence')
    plt.bar(x + width/2, random_values, width, label='Random Sequences')
    
    plt.xlabel('Relationship Type')
    plt.ylabel('Percentage (%)')
    plt.title('Comparison of Circular Hexagram Relationships')
    plt.xticks(x, rels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Analyze the grouping of hexagrams in the King Wen sequence
    # Split into trigrams (upper and lower)
    upper_trigrams = [hexagram[:3] for hexagram in king_wen_sequence]
    lower_trigrams = [hexagram[3:] for hexagram in king_wen_sequence]
    
    # Count frequency of each trigram
    upper_counts = pd.Series(upper_trigrams).value_counts()
    lower_counts = pd.Series(lower_trigrams).value_counts()
    
    # Check if certain trigrams tend to appear together
    trigram_pairs = list(zip(upper_trigrams, lower_trigrams))
    pair_counts = pd.Series(trigram_pairs).value_counts()
    
    # Calculate mutual information between upper and lower trigrams
    trigrams = np.unique(upper_trigrams + lower_trigrams)
    trigram_to_idx = {t: i for i, t in enumerate(trigrams)}
    
    upper_idx = [trigram_to_idx[t] for t in upper_trigrams]
    lower_idx = [trigram_to_idx[t] for t in lower_trigrams]
    
    mi = mutual_info_score(upper_idx, lower_idx)
    
    print(f"\nMutual information between upper and lower trigrams: {mi:.4f}")
    
    # Generate random sequences and calculate their mutual information
    random_mis = []
    for _ in range(1000):
        random_seq = generate_random_sequence(n)
        random_upper = [hexagram[:3] for hexagram in random_seq]
        random_lower = [hexagram[3:] for hexagram in random_seq]
        
        # Convert to indices, handling unknown trigrams
        random_upper_idx = []
        random_lower_idx = []
        for t in random_upper:
            if t in trigram_to_idx:
                random_upper_idx.append(trigram_to_idx[t])
        for t in random_lower:
            if t in trigram_to_idx:
                random_lower_idx.append(trigram_to_idx[t])
        
        # Only calculate mutual information if we have enough data
        if len(random_upper_idx) > 1 and len(random_lower_idx) > 1:
            random_mi = mutual_info_score(random_upper_idx, random_lower_idx)
            random_mis.append(random_mi)
    
    if random_mis:  # Only calculate statistics if we have valid results
        avg_random_mi = np.mean(random_mis)
        print(f"Average mutual information in random sequences: {avg_random_mi:.4f}")
        print(f"Percentage of random sequences with higher mutual information: {np.mean(np.array(random_mis) > mi) * 100:.1f}%")
    else:
        print("Could not calculate random sequence statistics due to insufficient data")
    
    # Analyze patterns at different circular distances
    print("\nAnalyzing patterns at different circular distances:")
    for distance in [1, 2, 4, 8, 16, 32]:
        hamming_dists = []
        for i in range(n):
            j = (i + distance) % n
            h_dist = hamming_distance(king_wen_sequence[i], king_wen_sequence[j])
            hamming_dists.append(h_dist)
        
        avg_dist = np.mean(hamming_dists)
        print(f"Average Hamming distance at circular distance {distance}: {avg_dist:.2f}")
    
    # Analyze opposite positions
    print("\nAnalyzing opposite positions:")
    opposite_hamming_dists = []
    for i in range(n):
        opposite_pos = get_opposite_position(i)
        h_dist = hamming_distance(king_wen_sequence[i], king_wen_sequence[opposite_pos])
        opposite_hamming_dists.append(h_dist)
    
    print(f"Average Hamming distance between opposite positions: {np.mean(opposite_hamming_dists):.2f}")
    
    # Visualize patterns at different circular distances
    plt.figure(figsize=(15, 10))
    
    # Create circular layout
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 1.0
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    # Plot hexagrams
    plt.scatter(x, y, c=opposite_hamming_dists, cmap='viridis', s=100)
    
    # Add labels
    for i in range(n):
        plt.text(x[i], y[i], str(i+1), fontsize=8, ha='center', va='center')
    
    # Highlight quarter positions
    quarter_positions = get_quarter_positions()
    quarter_x = [radius * np.cos(angles[i]) for i in quarter_positions]
    quarter_y = [radius * np.sin(angles[i]) for i in quarter_positions]
    plt.scatter(quarter_x, quarter_y, s=200, color='red', marker='*', label='Quarter Positions')
    
    plt.colorbar(label='Hamming Distance to Opposite Position')
    plt.title('Circular Pattern Analysis of King Wen Sequence')
    plt.axis('equal')
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return relationship_counts, pair_counts, opposite_hamming_dists 