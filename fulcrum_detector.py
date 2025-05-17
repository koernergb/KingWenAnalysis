import numpy as np
import matplotlib.pyplot as plt
from utils import king_wen_sequence, hamming_distance

def analyze_fulcrums():
    """Analyze fulcrum points and special positions in the King Wen sequence."""
    print("\nRunning fulcrum analysis...")
    return analyze_special_positions()

def analyze_special_positions():
    """Identify special positions and transitions in the King Wen sequence."""
    
    n = len(king_wen_sequence)
    
    # Calculate complexity of each transition
    transition_complexity = []
    for i in range(n-1):
        from_hex = king_wen_sequence[i]
        to_hex = king_wen_sequence[i+1]
        
        # Simple Hamming distance as one measure of complexity
        h_dist = hamming_distance(from_hex, to_hex)
        
        # Check if this is a standard transformation
        is_inversion = to_hex == ''.join('1' if bit == '0' else '0' for bit in from_hex)
        is_reversal = to_hex == from_hex[::-1]
        is_trigram_swap = to_hex == from_hex[3:] + from_hex[:3]
        
        # Calculate a complexity score
        # Lower score for standard transformations, higher for complex ones
        complexity = h_dist
        if is_inversion or is_reversal or is_trigram_swap:
            complexity = complexity * 0.5  # Standard transformations are less complex
            
        transition_complexity.append(complexity)
    
    # Identify positions with unusually high or low complexity
    mean_complexity = np.mean(transition_complexity)
    std_complexity = np.std(transition_complexity)
    
    significant_positions = []
    for i, complexity in enumerate(transition_complexity):
        # Transitions more than 1.5 standard deviations from the mean
        if abs(complexity - mean_complexity) > 1.5 * std_complexity:
            significant = "high" if complexity > mean_complexity else "low"
            significant_positions.append((i, i+1, complexity, significant))
    
    print("Significant transition positions:")
    for from_pos, to_pos, complexity, sig_type in significant_positions:
        print(f"Transition {from_pos+1} → {to_pos+1}: {sig_type} complexity ({complexity:.2f})")
    
    # Calculate a 'pivot score' for each position
    # A position with high pivot score has transitions before and after that are very different
    pivot_scores = []
    for i in range(1, n-1):
        before_complexity = transition_complexity[i-1]
        after_complexity = transition_complexity[i]
        pivot_scores.append(abs(after_complexity - before_complexity))
    
    # Identify significant pivots
    mean_pivot = np.mean(pivot_scores)
    std_pivot = np.std(pivot_scores)
    
    pivots = []
    for i, score in enumerate(pivot_scores):
        # Pivots more than 1.5 standard deviations from the mean
        if score > mean_pivot + 1.5 * std_pivot:
            pivots.append((i+1, score))  # i+1 because we started at position 1
    
    print("\nSignificant pivot positions:")
    for pos, score in sorted(pivots, key=lambda x: x[1], reverse=True)[:10]:  # Top 10
        print(f"Position {pos+1}: pivot score {score:.2f}")
    
    # Visualize the transition complexity and pivots
    plt.figure(figsize=(15, 6))
    
    # Plot complexity of each transition
    plt.plot(range(1, n), transition_complexity, 'b-', linewidth=2)
    plt.axhline(y=mean_complexity, color='r', linestyle='--', label='Mean Complexity')
    
    # Highlight significant pivots
    for pos, _ in pivots:
        plt.axvline(x=pos, color='g', alpha=0.3)
    
    plt.title('Transition Complexity and Pivot Positions in King Wen Sequence')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Transition Complexity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Analyze transitions within the traditional arrangement of hexagrams by houses
    # In traditional I Ching, hexagrams are often grouped into 8 houses (related to the 8 trigrams)
    houses = []
    for i in range(8):
        house = []
        for j in range(8):
            idx = i*8 + j
            if idx < n:
                house.append(idx)
        houses.append(house)
    
    # Calculate the transitions within and between houses
    intra_house_transitions = 0
    inter_house_transitions = 0
    
    for i in range(n-1):
        from_house = None
        to_house = None
        
        for h, house in enumerate(houses):
            if i in house:
                from_house = h
            if i+1 in house:
                to_house = h
        
        if from_house == to_house:
            intra_house_transitions += 1
        else:
            inter_house_transitions += 1
    
    print(f"\nTransitions within same house: {intra_house_transitions}")
    print(f"Transitions between different houses: {inter_house_transitions}")
    
    # Expected values if sequence was random
    expected_intra = n / 8  # n/8 transitions would be expected within houses by chance
    print(f"Expected intra-house transitions if random: {expected_intra:.1f}")
    print(f"Ratio of actual to expected: {intra_house_transitions / expected_intra:.2f}x")
    
    return transition_complexity, pivot_scores, houses