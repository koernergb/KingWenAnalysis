import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import Counter
from utils import king_wen_sequence, hamming_distance

def analyze_trigram_polarities() -> List[str]:
    """Analyze the distribution and relationship of polar trigram pairs in the sequence.
    
    Returns:
        List[str]: List of polarity transition types between consecutive hexagrams
        
    Example:
        >>> transitions = analyze_trigram_polarities()
        >>> print(f"Found {len(transitions)} polarity transitions")
    """
    # Define polar trigram pairs
    polar_pairs = {
        "111": "000",  # Heaven-Earth
        "101": "010",  # Fire-Water
        "100": "011",  # Thunder-Wind
        "110": "001"   # Lake-Mountain
    }
    
    # Analyze distribution of polar pairs in adjacent hexagrams
    polarity_transitions = []
    for i in range(len(king_wen_sequence) - 1):
        hex1 = king_wen_sequence[i]
        hex2 = king_wen_sequence[i + 1]
        
        upper1, lower1 = hex1[:3], hex1[3:]
        upper2, lower2 = hex2[:3], hex2[3:]
        
        # Check for polar relationships
        upper_polar = upper2 == polar_pairs.get(upper1, "")
        lower_polar = lower2 == polar_pairs.get(lower1, "")
        
        if upper_polar and lower_polar:
            polarity_transitions.append("Double polarity")
        elif upper_polar:
            polarity_transitions.append("Upper polarity")
        elif lower_polar:
            polarity_transitions.append("Lower polarity")
        else:
            polarity_transitions.append("No polarity")
    
    # Visualize polarity transitions
    plt.figure(figsize=(12, 6))
    transition_counts = Counter(polarity_transitions)
    plt.bar(transition_counts.keys(), transition_counts.values())
    plt.title('Distribution of Polarity Transitions')
    plt.xlabel('Transition Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return polarity_transitions

def analyze_nuclear_hexagrams() -> Dict[int, Optional[int]]:
    """Analyze relationships between hexagrams and their nuclear derivatives.
    
    Returns:
        Dict[int, Optional[int]]: Dictionary mapping hexagram positions to their nuclear positions
        
    Example:
        >>> nuclear_relations = analyze_nuclear_hexagrams()
        >>> print(f"Found {len(nuclear_relations)} nuclear relationships")
    """
    nuclear_relations = {}
    
    for i, hexagram in enumerate(king_wen_sequence):
        # Extract nuclear hexagram (middle four lines form two trigrams)
        nuclear = hexagram[1:5]
        nuclear_hex = nuclear[0:2] + nuclear[1:3] + nuclear[2:4]
        
        # Find position of this nuclear hexagram in the sequence
        try:
            nuclear_pos = king_wen_sequence.index(nuclear_hex)
            nuclear_relations[i+1] = nuclear_pos + 1  # 1-indexed
        except ValueError:
            nuclear_relations[i+1] = None
    
    # Visualize nuclear relationships
    plt.figure(figsize=(12, 6))
    valid_relations = {k: v for k, v in nuclear_relations.items() if v is not None}
    plt.scatter(list(valid_relations.keys()), list(valid_relations.values()), alpha=0.6)
    plt.plot([1, 64], [1, 64], 'k--', alpha=0.3)  # Diagonal line
    plt.title('Nuclear Hexagram Relationships')
    plt.xlabel('Hexagram Position')
    plt.ylabel('Nuclear Hexagram Position')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return nuclear_relations

def vector_representation_analysis() -> Dict[str, List[int]]:
    """Analyze hexagrams as vectors and their transformations in sequence.
    
    Returns:
        Dict[str, List[int]]: Dictionary of transformation patterns and their positions
        
    Example:
        >>> transform_classes = vector_representation_analysis()
        >>> print(f"Found {len(transform_classes)} transformation classes")
    """
    # Represent hexagrams as numerical vectors (converting binary to integers)
    hex_vectors = []
    for hexagram in king_wen_sequence:
        # Convert each line to numerical value (0 or 1)
        vector = [int(bit) for bit in hexagram]
        hex_vectors.append(vector)
    
    # Calculate vector differences between consecutive hexagrams
    vector_transformations = []
    for i in range(len(hex_vectors) - 1):
        # Element-wise subtraction module 2 (XOR operation in binary)
        transform = [(a - b) % 2 for a, b in zip(hex_vectors[i+1], hex_vectors[i])]
        vector_transformations.append(transform)
    
    # Classify transformation types by vector patterns
    transformation_classes = {}
    for i, transform in enumerate(vector_transformations):
        transform_key = ''.join(str(x) for x in transform)
        if transform_key in transformation_classes:
            transformation_classes[transform_key].append(i+1)
        else:
            transformation_classes[transform_key] = [i+1]
    
    # Visualize transformation patterns
    plt.figure(figsize=(12, 6))
    pattern_counts = {k: len(v) for k, v in transformation_classes.items()}
    plt.bar(pattern_counts.keys(), pattern_counts.values())
    plt.title('Distribution of Vector Transformation Patterns')
    plt.xlabel('Transformation Pattern')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return transformation_classes

def analyze_transformation_groups() -> Dict[str, List[int]]:
    """Identify groups of hexagrams connected by similar transformations.
    
    Returns:
        Dict[str, List[int]]: Dictionary of transformation operations and their target positions
        
    Example:
        >>> transform_groups = analyze_transformation_groups()
        >>> print(f"Found {len(transform_groups)} transformation groups")
    """
    # Define transformation operations based on Körtvélyessy's work
    operations = {
        "identity": lambda h: h,
        "reversal": lambda h: h[::-1],
        "inversion": lambda h: ''.join('1' if bit == '0' else '0' for bit in h),
        "line_1_change": lambda h: h[0] + h[1:],
        "line_2_change": lambda h: h[:1] + ('1' if h[1] == '0' else '0') + h[2:],
        # Additional operations can be defined
    }
    
    # Find groups of hexagrams connected by these operations
    transformation_groups = {}
    
    for i, hex1 in enumerate(king_wen_sequence):
        for op_name, op_func in operations.items():
            transformed = op_func(hex1)
            
            # Find all hexagrams that can be reached by this transformation
            targets = []
            for j, hex2 in enumerate(king_wen_sequence):
                if transformed == hex2:
                    targets.append(j+1)
            
            if targets:
                key = f"{i+1}_{op_name}"
                transformation_groups[key] = targets
    
    # Visualize transformation groups
    plt.figure(figsize=(12, 6))
    group_sizes = [len(v) for v in transformation_groups.values()]
    plt.hist(group_sizes, bins=range(min(group_sizes), max(group_sizes) + 2))
    plt.title('Distribution of Transformation Group Sizes')
    plt.xlabel('Group Size')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return transformation_groups

def compare_king_wen_to_shao_yong() -> List[int]:
    """Compare King Wen sequence to Shao Yong's mathematical arrangement.
    
    Returns:
        List[int]: List of transformation distances between corresponding hexagrams
        
    Example:
        >>> distances = compare_king_wen_to_shao_yong()
        >>> print(f"Average transformation distance: {np.mean(distances):.2f}")
    """
    # Generate Shao Yong's "Prior Heaven" sequence (essentially binary ordering)
    shao_yong_sequence = [''.join(format(i, '06b')) for i in range(64)]
    
    # Calculate transformation distance from King Wen to Shao Yong
    transformation_distances = []
    for hex_kw, hex_sy in zip(king_wen_sequence, shao_yong_sequence):
        # Calculate Hamming distance
        distance = sum(c1 != c2 for c1, c2 in zip(hex_kw, hex_sy))
        transformation_distances.append(distance)
    
    # Visualize transformation distances
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 65), transformation_distances, 'b-', alpha=0.6)
    plt.title('Transformation Distance from King Wen to Shao Yong')
    plt.xlabel('Hexagram Position')
    plt.ylabel('Transformation Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return transformation_distances

def analyze_octagonal_symmetry() -> Dict[int, str]:
    """Analyze octagonal symmetry in the King Wen sequence compared to Shao Yong's arrangement.
    
    Returns:
        Dict[int, str]: Dictionary mapping pure hexagram positions to compass directions
        
    Example:
        >>> compass_map = analyze_octagonal_symmetry()
        >>> print(f"Found {len(compass_map)} pure hexagrams with compass positions")
    """
    # Define cardinal and ordinal compass positions in Shao Yong's arrangement
    compass_positions = {
        "111": "South",       # Heaven
        "000": "North",       # Earth
        "100": "Northeast",   # Thunder
        "011": "Southwest",   # Wind
        "010": "West",        # Water
        "101": "East",        # Fire
        "001": "Northwest",   # Mountain
        "110": "Southeast"    # Lake
    }
    
    # Map the eight pure hexagrams in King Wen to compass positions
    king_wen_compass = {}
    for i, hexagram in enumerate(king_wen_sequence):
        upper = hexagram[:3]
        lower = hexagram[3:]
        
        if upper == lower:  # Pure hexagram
            king_wen_compass[i+1] = compass_positions.get(upper, "Unknown")
    
    # Visualize compass positions
    plt.figure(figsize=(10, 10))
    compass_counts = Counter(king_wen_compass.values())
    plt.pie(compass_counts.values(), labels=compass_counts.keys(), autopct='%1.1f%%')
    plt.title('Distribution of Pure Hexagrams by Compass Position')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    return king_wen_compass

if __name__ == "__main__":
    # Run all analyses
    print("Running Marshall-Yong-Körtvélyessy analysis...")
    
    print("\n1. Analyzing trigram polarities...")
    polarity_transitions = analyze_trigram_polarities()
    print(f"Found {len(polarity_transitions)} polarity transitions")
    
    print("\n2. Analyzing nuclear hexagrams...")
    nuclear_relations = analyze_nuclear_hexagrams()
    print(f"Found {len(nuclear_relations)} nuclear relationships")
    
    print("\n3. Analyzing vector representations...")
    transform_classes = vector_representation_analysis()
    print(f"Found {len(transform_classes)} transformation classes")
    
    print("\n4. Analyzing transformation groups...")
    transform_groups = analyze_transformation_groups()
    print(f"Found {len(transform_groups)} transformation groups")
    
    print("\n5. Comparing to Shao Yong arrangement...")
    distances = compare_king_wen_to_shao_yong()
    print(f"Average transformation distance: {np.mean(distances):.2f}")
    
    print("\n6. Analyzing octagonal symmetry...")
    compass_map = analyze_octagonal_symmetry()
    print(f"Found {len(compass_map)} pure hexagrams with compass positions")