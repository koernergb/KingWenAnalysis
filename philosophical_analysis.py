import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from utils import (
    king_wen_sequence, hamming_distance,
    circular_distance, get_opposite_position,
    get_quarter_positions
)

def analyze_philosophy():
    """Analyze patterns related to traditional Chinese philosophy in the King Wen sequence, considering circularity."""
    print("\nRunning philosophical analysis...")
    return analyze_philosophical_patterns()

def analyze_philosophical_patterns():
    """Analyze patterns related to traditional Chinese philosophy in the King Wen sequence, considering circularity."""
    n = len(king_wen_sequence)
    
    # Convert hexagrams to numeric form for easier manipulation
    hex_numeric = [int(h, 2) for h in king_wen_sequence]
    
    # 1. Yin-Yang Oscillation Analysis (Circular)
    yin_yang_counts = []
    
    for hex_str in king_wen_sequence:
        # Count yin (0) and yang (1) lines
        yin = hex_str.count('0')
        yang = hex_str.count('1')
        yin_yang_counts.append((yin, yang))
    
    # Calculate the balance over time (circular)
    balance = [yang - yin for yin, yang in yin_yang_counts]
    cumulative_balance = np.cumsum(balance)
    
    # Identify trends of increasing/decreasing yang influence
    trends = []
    current_trend = 'none'
    trend_start = 0
    
    for i in range(1, n):
        if cumulative_balance[i] > cumulative_balance[i-1] and current_trend != 'increasing':
            if current_trend != 'none':
                trends.append((trend_start, i-1, current_trend))
            current_trend = 'increasing'
            trend_start = i
        elif cumulative_balance[i] < cumulative_balance[i-1] and current_trend != 'decreasing':
            if current_trend != 'none':
                trends.append((trend_start, i-1, current_trend))
            current_trend = 'decreasing'
            trend_start = i
    
    # Add the final trend
    if current_trend != 'none':
        trends.append((trend_start, n-1, current_trend))
    
    # Plot the yin-yang balance
    plt.figure(figsize=(15, 6))
    plt.plot(range(1, n+1), cumulative_balance, 'b-', linewidth=2)
    
    # Color the trend regions
    for start, end, trend in trends:
        color = 'red' if trend == 'increasing' else 'blue'
        plt.axvspan(start+1, end+1, alpha=0.2, color=color)
    
    # Highlight quarter positions
    quarter_positions = get_quarter_positions()
    for pos in quarter_positions:
        plt.axvline(x=pos+1, color='g', linestyle=':', alpha=0.5)
    
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Circular Yin-Yang Balance with Growth/Decline Phases')
    plt.xlabel('Position in King Wen Sequence')
    plt.ylabel('Cumulative Yang - Yin')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 2. The Five Elements Correspondence (Circular)
    # In Chinese philosophy, the five elements (Wu Xing) are Wood, Fire, Earth, Metal, Water
    # We can map these to trigram properties
    
    # Define the element correspondences for trigrams
    element_map = {
        '000': 'Water',  # Kan
        '001': 'Earth',  # Gen
        '010': 'Fire',   # Li
        '011': 'Wood',   # Zhen
        '100': 'Earth',  # Kun
        '101': 'Metal',  # Dui
        '110': 'Metal',  # Qian
        '111': 'Wood'    # Xun
    }
    
    # Analyze element transitions in the sequence (circular)
    element_transitions = []
    
    for i in range(n):
        # Get upper trigrams
        upper_from = king_wen_sequence[i][:3]
        upper_to = king_wen_sequence[(i + 1) % n][:3]
        
        # Get lower trigrams
        lower_from = king_wen_sequence[i][3:]
        lower_to = king_wen_sequence[(i + 1) % n][3:]
        
        # Get elements
        upper_from_element = element_map[upper_from]
        upper_to_element = element_map[upper_to]
        lower_from_element = element_map[lower_from]
        lower_to_element = element_map[lower_to]
        
        element_transitions.append({
            'upper': (upper_from_element, upper_to_element),
            'lower': (lower_from_element, lower_to_element)
        })
    
    # Count element transition types
    upper_transitions = [t['upper'] for t in element_transitions]
    lower_transitions = [t['lower'] for t in element_transitions]
    
    upper_counts = Counter(upper_transitions)
    lower_counts = Counter(lower_transitions)
    
    # Define the traditional generative and controlling cycles
    generative_cycle = [('Wood', 'Fire'), ('Fire', 'Earth'), ('Earth', 'Metal'), 
                        ('Metal', 'Water'), ('Water', 'Wood')]
    
    controlling_cycle = [('Wood', 'Earth'), ('Earth', 'Water'), ('Water', 'Fire'), 
                         ('Fire', 'Metal'), ('Metal', 'Wood')]
    
    # Count generative, controlling, and other transitions
    generative_upper = sum(upper_counts[t] for t in generative_cycle)
    controlling_upper = sum(upper_counts[t] for t in controlling_cycle)
    other_upper = sum(upper_counts.values()) - generative_upper - controlling_upper
    
    generative_lower = sum(lower_counts[t] for t in generative_cycle)
    controlling_lower = sum(lower_counts[t] for t in controlling_cycle)
    other_lower = sum(lower_counts.values()) - generative_lower - controlling_lower
    
    # Calculate expected values for random sequence
    total_transitions = len(element_transitions)
    expected_generative = total_transitions * 5 / 25  # 5 out of 25 possible transitions
    expected_controlling = total_transitions * 5 / 25
    
    print("\nFive Elements Analysis (Circular):")
    print(f"Upper Trigram Transitions: {generative_upper} generative, {controlling_upper} controlling, {other_upper} other")
    print(f"Lower Trigram Transitions: {generative_lower} generative, {controlling_lower} controlling, {other_lower} other")
    print(f"Expected if random: {expected_generative:.1f} generative, {expected_controlling:.1f} controlling")
    
    print(f"\nUpper Trigram Ratio: {generative_upper/expected_generative:.2f}x generative, {controlling_upper/expected_controlling:.2f}x controlling")
    print(f"Lower Trigram Ratio: {generative_lower/expected_generative:.2f}x generative, {controlling_lower/expected_controlling:.2f}x controlling")
    
    # Visualize the element transitions
    plt.figure(figsize=(15, 6))
    
    # Plot upper trigram element transitions
    categories = ['Generative', 'Controlling', 'Other']
    upper_values = [generative_upper, controlling_upper, other_upper]
    lower_values = [generative_lower, controlling_lower, other_lower]
    expected_values = [expected_generative, expected_controlling, total_transitions - 2*expected_generative]
    
    x = np.arange(len(categories))
    width = 0.2
    
    plt.bar(x - width, upper_values, width, label='Upper Trigram', color='blue')
    plt.bar(x, lower_values, width, label='Lower Trigram', color='green')
    plt.bar(x + width, expected_values, width, label='Expected Random', color='red')
    
    plt.ylabel('Number of Transitions')
    plt.title('Circular Five Elements Transition Patterns')
    plt.xticks(x, categories)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 3. Bagua (Eight Trigrams) Analysis (Circular)
    # Analyze how the eight trigrams are distributed and transition in the sequence
    
    # Define the trigrams and their traditional associations
    trigrams = {
        '000': 'Kun (Earth)',
        '001': 'Gen (Mountain)',
        '010': 'Kan (Water)',
        '011': 'Xun (Wind)',
        '100': 'Zhen (Thunder)',
        '101': 'Li (Fire)',
        '110': 'Dui (Lake)',
        '111': 'Qian (Heaven)'
    }
    
    # Count occurrences and transitions of trigrams
    upper_trigram_counts = Counter([h[:3] for h in king_wen_sequence])
    lower_trigram_counts = Counter([h[3:] for h in king_wen_sequence])
    
    upper_transitions = [(king_wen_sequence[i][:3], king_wen_sequence[(i + 1) % n][:3]) 
                         for i in range(n)]
    lower_transitions = [(king_wen_sequence[i][3:], king_wen_sequence[(i + 1) % n][3:]) 
                         for i in range(n)]
    
    upper_transition_counts = Counter(upper_transitions)
    lower_transition_counts = Counter(lower_transitions)
    
    # Calculate the expected distribution for a random sequence
    expected_count = n / 8
    
    print("\nTrigram Distribution Analysis (Circular):")
    print("\nUpper Trigram Counts (Expected: {:.1f} each):".format(expected_count))
    for trigram, name in sorted(trigrams.items()):
        actual = upper_trigram_counts[trigram]
        ratio = actual / expected_count
        print(f"{name}: {actual} ({ratio:.2f}x expected)")
    
    print("\nLower Trigram Counts (Expected: {:.1f} each):".format(expected_count))
    for trigram, name in sorted(trigrams.items()):
        actual = lower_trigram_counts[trigram]
        ratio = actual / expected_count
        print(f"{name}: {actual} ({ratio:.2f}x expected)")
    
    # Visualize the trigram distribution
    plt.figure(figsize=(15, 6))
    
    sorted_trigrams = sorted(trigrams.items())
    x = np.arange(len(sorted_trigrams))
    width = 0.35
    
    upper_values = [upper_trigram_counts[t] for t, _ in sorted_trigrams]
    lower_values = [lower_trigram_counts[t] for t, _ in sorted_trigrams]
    expected_value = [expected_count] * len(sorted_trigrams)
    
    plt.bar(x - width/2, upper_values, width, label='Upper Trigram')
    plt.bar(x + width/2, lower_values, width, label='Lower Trigram')
    plt.plot(x, expected_value, 'r--', label='Expected')
    
    plt.title('Circular Distribution of Trigrams in King Wen Sequence')
    plt.xlabel('Trigram')
    plt.ylabel('Count')
    plt.xticks(x, [name.split()[0] for _, name in sorted_trigrams], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 4. Analyze the 8 Palaces or Houses structure (Circular)
    # In traditional I Ching interpretation, the 64 hexagrams are divided into 8 houses
    # Each house is headed by a hexagram with the same trigram on top and bottom
    
    # Define the pure hexagrams (same trigram repeated)
    pure_hexagrams = {t+t: name.split()[0] for t, name in trigrams.items()}
    
    # Find the positions of pure hexagrams in the King Wen sequence
    pure_positions = {}
    for i, hexagram in enumerate(king_wen_sequence):
        if hexagram in pure_hexagrams:
            pure_positions[hexagram] = i
    
    # Sort by position
    pure_hexagrams_by_position = sorted(pure_positions.items(), key=lambda x: x[1])
    
    print("\nPure Trigram Hexagrams (House Heads) in King Wen Sequence:")
    for hexagram, position in pure_hexagrams_by_position:
        print(f"Position {position+1}: {pure_hexagrams[hexagram]} ({hexagram})")
    
    # Analyze if hexagrams between pure hexagrams form coherent houses
    houses = []
    for i in range(len(pure_hexagrams_by_position)):
        start_pos = pure_hexagrams_by_position[i][1]
        if i < len(pure_hexagrams_by_position) - 1:
            end_pos = pure_hexagrams_by_position[i+1][1]
        else:
            end_pos = n
        
        house = list(range(start_pos, end_pos))
        houses.append(house)
    
    # Check for patterns within each house
    print("\nHouse Analysis (Circular):")
    for i, house in enumerate(houses):
        if not house:
            continue
            
        start_hex = king_wen_sequence[house[0]]
        house_trigram = start_hex[:3]  # Usually the upper trigram of the pure hexagram
        
        # Count hexagrams in the house with this trigram as upper or lower
        upper_match = sum(1 for pos in house if king_wen_sequence[pos][:3] == house_trigram)
        lower_match = sum(1 for pos in house if king_wen_sequence[pos][3:] == house_trigram)
        
        print(f"\nHouse {i+1} ({pure_hexagrams.get(start_hex, 'Unknown')}), Positions {min(house)+1}-{max(house)+1}:")
        print(f"Hexagrams with house trigram as upper: {upper_match}/{len(house)} ({upper_match/len(house)*100:.1f}%)")
        print(f"Hexagrams with house trigram as lower: {lower_match}/{len(house)} ({lower_match/len(house)*100:.1f}%)")
    
    # Visualize the houses
    plt.figure(figsize=(15, 6))
    
    for i, house in enumerate(houses):
        if not house:
            continue
        
        # Plot a colored region for this house
        color = plt.cm.tab10(i % 10)
        start = min(house)
        end = max(house) + 1
        plt.axvspan(start+1, end+1, color=color, alpha=0.3, 
                   label=f"House {i+1}: {pure_hexagrams.get(king_wen_sequence[house[0]], 'Unknown')}")
    
    # Plot house transitions
    for i in range(n):
        # Find which house i and i+1 belong to
        house_i, house_i1 = None, None
        for h, house in enumerate(houses):
            if i in house:
                house_i = h
            if (i+1) % n in house:
                house_i1 = h
        
        # If crossing house boundary, draw a vertical line
        if house_i is not None and house_i1 is not None and house_i != house_i1:
            plt.axvline(x=i+1.5, color='red', linestyle='--', alpha=0.7)
    
    # Highlight quarter positions
    for pos in quarter_positions:
        plt.axvline(x=pos+1, color='g', linestyle=':', alpha=0.5)
    
    plt.xlim(1, n+1)
    plt.title('Circular House Structure of the King Wen Sequence')
    plt.xlabel('Position in Sequence')
    plt.ylabel('')
    plt.yticks([])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.tight_layout()
    plt.show()
    
    # 5. Analyze opposite positions in houses
    print("\nAnalysis of opposite positions in houses:")
    for i, house in enumerate(houses):
        if not house:
            continue
            
        print(f"\nHouse {i+1}:")
        for pos in house:
            opposite_pos = get_opposite_position(pos)
            opposite_house = None
            for h, h_house in enumerate(houses):
                if opposite_pos in h_house:
                    opposite_house = h + 1
                    break
            
            if opposite_house is not None:
                print(f"Position {pos+1} ↔ {opposite_pos+1} (House {opposite_house})")
    
    return trends, element_transitions, houses