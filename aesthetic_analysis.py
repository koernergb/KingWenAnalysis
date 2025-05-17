import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from utils import king_wen_sequence, hamming_distance

def fibonacci_analysis(sequence=king_wen_sequence):
    """Analyze the King Wen sequence for Fibonacci-related patterns.
    
    Args:
        sequence: List of hexagrams in binary string format
        
    Returns:
        dict: Analysis results including Fibonacci positions, hexagrams, and relationships
    """
    # Generate Fibonacci numbers up to position 64
    fib_numbers = [1, 1]
    while len(fib_numbers) < 20:  # First 20 Fibonacci numbers
        fib_numbers.append(fib_numbers[-1] + fib_numbers[-2])
    
    # Examine hexagrams at Fibonacci positions
    fib_positions = [f for f in fib_numbers if f <= 64]
    fib_hexagrams = [sequence[pos-1] for pos in fib_positions]
    
    # Analyze relationships between these hexagrams
    relationships = []
    for i in range(len(fib_positions)-1):
        pos1 = fib_positions[i]
        pos2 = fib_positions[i+1]
        hex1 = sequence[pos1-1]
        hex2 = sequence[pos2-1]
        
        # Calculate transformation type
        if hex2 == ''.join('1' if bit == '0' else '0' for bit in hex1):
            trans_type = "Inversion"
        elif hex2 == hex1[::-1]:
            trans_type = "Reversal"
        else:
            # Calculate Hamming distance
            h_dist = sum(c1 != c2 for c1, c2 in zip(hex1, hex2))
            trans_type = f"Hamming distance {h_dist}"
            
        relationships.append((pos1, pos2, trans_type))
    
    # Visualize Fibonacci positions
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 65), [0]*64, 'k-', alpha=0.3)
    plt.scatter(fib_positions, [0]*len(fib_positions), c='r', s=100)
    for pos in fib_positions:
        plt.text(pos, 0.1, str(pos), ha='center', va='bottom')
    plt.title('Fibonacci Positions in King Wen Sequence')
    plt.xlabel('Position')
    plt.yticks([])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        "fibonacci_positions": fib_positions,
        "fibonacci_hexagrams": fib_hexagrams,
        "relationships": relationships
    }

def golden_ratio_analysis(sequence=king_wen_sequence):
    """Analyze hexagram positions based on Golden Ratio proportions.
    
    Args:
        sequence: List of hexagrams in binary string format
        
    Returns:
        list: Pairs of positions and their relationships based on Golden Ratio
    """
    # Golden Ratio ≈ 1.618
    phi = (1 + 5**0.5) / 2
    
    # For each position, check the hexagram at position[i*phi] (rounded)
    golden_pairs = []
    for i in range(1, 40):  # Check first 40 positions
        position1 = i
        position2 = round(i * phi)
        
        if position2 <= 64:
            hex1 = sequence[position1-1]
            hex2 = sequence[position2-1]
            
            # Calculate relationship
            if hex2 == ''.join('1' if bit == '0' else '0' for bit in hex1):
                trans_type = "Inversion"
            elif hex2 == hex1[::-1]:
                trans_type = "Reversal"
            else:
                h_dist = sum(c1 != c2 for c1, c2 in zip(hex1, hex2))
                trans_type = f"Hamming distance {h_dist}"
                
            golden_pairs.append((position1, position2, trans_type))
    
    # Visualize Golden Ratio relationships
    plt.figure(figsize=(12, 6))
    for pos1, pos2, _ in golden_pairs:
        plt.plot([pos1, pos2], [0, 0], 'b-', alpha=0.3)
        plt.scatter([pos1, pos2], [0, 0], c='g', s=50)
        plt.text(pos1, 0.1, str(pos1), ha='center', va='bottom')
        plt.text(pos2, 0.1, str(pos2), ha='center', va='bottom')
    plt.title('Golden Ratio Relationships in King Wen Sequence')
    plt.xlabel('Position')
    plt.yticks([])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return golden_pairs

def aesthetic_proportion_analysis(sequence=king_wen_sequence):
    """Analyze aesthetic proportions in the distribution of transformations.
    
    Args:
        sequence: List of hexagrams in binary string format
        
    Returns:
        dict: Analysis of transformation proportions and their relationship to aesthetic ratios
    """
    # Calculate transformations between consecutive hexagrams
    transformations = []
    for i in range(len(sequence)-1):
        hex1 = sequence[i]
        hex2 = sequence[i+1]
        
        # Determine transformation type
        if hex2 == ''.join('1' if bit == '0' else '0' for bit in hex1):
            trans_type = "Inversion"
        elif hex2 == hex1[::-1]:
            trans_type = "Reversal"
        elif hex2 == hex1[3:] + hex1[:3]:
            trans_type = "Trigram Swap"
        else:
            h_dist = sum(c1 != c2 for c1, c2 in zip(hex1, hex2))
            trans_type = f"Hamming_{h_dist}"
            
        transformations.append(trans_type)
    
    # Count transformation types
    trans_counts = Counter(transformations)
    
    # Calculate proportions
    total = len(transformations)
    proportions = {t: count/total for t, count in trans_counts.items()}
    
    # Check if these proportions are close to golden ratio or Fibonacci ratios
    golden_ratio = (1 + 5**0.5) / 2
    fibonacci_ratios = [1/1, 2/1, 3/2, 5/3, 8/5, 13/8]  # Consecutive Fibonacci number ratios
    
    ratio_analysis = {}
    for t1 in proportions:
        for t2 in proportions:
            if t1 != t2 and proportions[t1] > 0 and proportions[t2] > 0:
                ratio = proportions[t1] / proportions[t2]
                
                # Check closeness to golden ratio
                golden_proximity = abs(ratio - golden_ratio)
                
                # Check closeness to Fibonacci ratios
                fib_proximities = [abs(ratio - fr) for fr in fibonacci_ratios]
                closest_fib = min(fib_proximities)
                closest_fib_idx = fib_proximities.index(closest_fib)
                
                if golden_proximity < 0.1 or closest_fib < 0.1:
                    ratio_analysis[(t1, t2)] = {
                        "ratio": ratio,
                        "golden_proximity": golden_proximity,
                        "closest_fibonacci": fibonacci_ratios[closest_fib_idx],
                        "fibonacci_proximity": closest_fib
                    }
    
    # Visualize transformation proportions
    plt.figure(figsize=(12, 6))
    plt.bar(proportions.keys(), proportions.values())
    plt.title('Proportions of Transformation Types')
    plt.xlabel('Transformation Type')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return {
        "transformation_counts": trans_counts,
        "transformation_proportions": proportions,
        "aesthetic_ratios": ratio_analysis
    }

def analyze_aesthetic_patterns():
    """Run all aesthetic analyses and print results."""
    print("\nAnalyzing Fibonacci patterns...")
    fib_results = fibonacci_analysis()
    print(f"\nFound {len(fib_results['fibonacci_positions'])} Fibonacci positions")
    print("Relationships between consecutive Fibonacci positions:")
    for pos1, pos2, trans_type in fib_results['relationships']:
        print(f"Position {pos1} → {pos2}: {trans_type}")
    
    print("\nAnalyzing Golden Ratio relationships...")
    golden_results = golden_ratio_analysis()
    print(f"\nFound {len(golden_results)} Golden Ratio pairs")
    print("Sample relationships:")
    for pos1, pos2, trans_type in golden_results[:5]:
        print(f"Position {pos1} ↔ {pos2}: {trans_type}")
    
    print("\nAnalyzing aesthetic proportions...")
    prop_results = aesthetic_proportion_analysis()
    print("\nTransformation type proportions:")
    for trans_type, prop in prop_results['transformation_proportions'].items():
        print(f"{trans_type}: {prop:.3f}")
    
    print("\nAesthetic ratio relationships:")
    for (t1, t2), analysis in prop_results['aesthetic_ratios'].items():
        print(f"\n{t1} : {t2} = {analysis['ratio']:.3f}")
        if analysis['golden_proximity'] < 0.1:
            print(f"Close to Golden Ratio (proximity: {analysis['golden_proximity']:.3f})")
        if analysis['fibonacci_proximity'] < 0.1:
            print(f"Close to Fibonacci ratio {analysis['closest_fibonacci']} (proximity: {analysis['fibonacci_proximity']:.3f})")

if __name__ == "__main__":
    analyze_aesthetic_patterns()
