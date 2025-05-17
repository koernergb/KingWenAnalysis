import numpy as np
import matplotlib.pyplot as plt
from utils import (
    king_wen_sequence, hamming_distance,
    get_opposite_position, get_quarter_positions,
    circular_distance
)

def analyze_symmetry():
    """Analyze symmetry and balance in the King Wen sequence, considering circularity."""
    print("\nRunning symmetry analysis...")
    return analyze_symmetry_internal()

def analyze_symmetry_internal():
    """Analyze symmetry and balance in the King Wen sequence, considering circularity."""
    
    n = len(king_wen_sequence)
    
    # Define transformation functions
    transformations = {
        'identity': lambda h: h,
        'inversion': lambda h: ''.join('1' if bit == '0' else '0' for bit in h),
        'reversal': lambda h: h[::-1],
        'rev_inv': lambda h: ''.join('1' if bit == '0' else '0' for bit in h[::-1])
    }
    
    # Check for symmetry around each possible pivot
    symmetry_scores = {}
    best_transformation = {}
    
    for pivot in range(n):
        symmetry_scores[pivot] = {}
        
        for trans_name, trans_func in transformations.items():
            score = 0
            pairs = 0
            
            # Check pairs equidistant from the pivot
            for offset in range(1, n//2 + 1):
                left_idx = (pivot - offset) % n
                right_idx = (pivot + offset) % n
                
                left_hex = king_wen_sequence[left_idx]
                right_hex = king_wen_sequence[right_idx]
                
                if right_hex == trans_func(left_hex):
                    score += 1
                
                pairs += 1
            
            symmetry_scores[pivot][trans_name] = score / pairs if pairs > 0 else 0
        
        # Find best transformation for this pivot
        best_trans = max(symmetry_scores[pivot].items(), key=lambda x: x[1])
        best_transformation[pivot] = best_trans
    
    # Find pivots with highest symmetry
    best_pivots = sorted(best_transformation.items(), key=lambda x: x[1][1], reverse=True)
    
    print("Top symmetry pivots in the King Wen sequence:")
    for pivot, (trans, score) in best_pivots[:10]:  # Show top 10
        print(f"Pivot at position {pivot+1} with {trans} transformation: {score:.2f} symmetry score")
    
    # Analyze quarter positions
    quarter_positions = get_quarter_positions()
    print("\nSymmetry at quarter positions:")
    for pos in quarter_positions:
        best_trans = best_transformation[pos]
        print(f"Position {pos+1}: {best_trans[0]} transformation with score {best_trans[1]:.2f}")
    
    # Visualize symmetry around the best pivot
    best_pivot, (best_trans, _) = best_pivots[0]
    trans_func = transformations[best_trans]
    
    plt.figure(figsize=(15, 6))
    
    # Plot hexagrams as points on a circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    
    plt.scatter(x, y, color='blue', s=100, alpha=0.7)
    
    # Highlight the pivot
    pivot_x = np.cos(angles[best_pivot])
    pivot_y = np.sin(angles[best_pivot])
    plt.scatter(pivot_x, pivot_y, color='red', s=200, marker='*')
    
    # Connect symmetric pairs
    for offset in range(1, n//2 + 1):
        left_idx = (best_pivot - offset) % n
        right_idx = (best_pivot + offset) % n
        
        left_hex = king_wen_sequence[left_idx]
        right_hex = king_wen_sequence[right_idx]
        
        if right_hex == trans_func(left_hex):
            plt.plot([x[left_idx], x[right_idx]], [y[left_idx], y[right_idx]], 'g-', alpha=0.5)
    
    plt.title(f'Circular Symmetry Around Pivot {best_pivot+1} with {best_trans.title()} Transformation')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Yin-Yang Balance Analysis
    yin_count = []  # Count of broken lines (0) in each hexagram
    yang_count = []  # Count of solid lines (1) in each hexagram
    
    for hex_str in king_wen_sequence:
        yin = hex_str.count('0')
        yang = hex_str.count('1')
        yin_count.append(yin)
        yang_count.append(yang)
    
    # Calculate cumulative balance
    balance = np.cumsum(np.array(yang_count) - np.array(yin_count))
    
    # Plot the balance
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, n+1), balance, 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Highlight quarter positions
    for pos in quarter_positions:
        plt.axvline(x=pos+1, color='g', linestyle=':', alpha=0.5)
    
    plt.title('Cumulative Yin-Yang Balance Throughout King Wen Sequence')
    plt.xlabel('Position in Sequence')
    plt.ylabel('Cumulative Yang - Yin (Positive = Yang Dominant)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Analyze opposite positions
    opposite_correlations = []
    for i in range(n):
        opposite_pos = get_opposite_position(i)
        h_dist = hamming_distance(king_wen_sequence[i], king_wen_sequence[opposite_pos])
        opposite_correlations.append(h_dist)
    
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, n+1), opposite_correlations, 'b-', linewidth=2)
    plt.title('Hamming Distance Between Opposite Positions in King Wen Sequence')
    plt.xlabel('Position')
    plt.ylabel('Hamming Distance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return symmetry_scores, balance