import numpy as np
import random

# Define the King Wen sequence hexagrams as binary strings
# Each hexagram is represented bottom-to-top, with 1 for yang (solid line) and 0 for yin (broken line)
king_wen_sequence = [
    "111111", "000000", "100010", "010001", "111010", "010111", "010000", "000010",
    "111011", "110111", "111000", "000111", "101111", "111101", "001000", "000100",
    "100110", "011001", "110000", "000011", "100101", "101001", "000001", "100000",
    "100111", "111001", "100001", "100110", "010010", "101101", "001110", "011100",
    "001111", "111100", "000101", "101000", "101011", "110101", "001010", "010100",
    "110001", "100011", "111110", "011111", "000110", "011000", "010011", "110010",
    "011010", "010110", "101110", "011101", "100100", "001001", "001011", "110100",
    "101100", "001101", "011011", "110110", "010101", "101010", "001100", "001100"
]

# Convert binary strings to integers for easier computation
king_wen_integers = [int(hexagram, 2) for hexagram in king_wen_sequence]

def hamming_distance(str1, str2):
    """Calculate the Hamming distance between two binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def circular_distance(pos1, pos2, sequence_length=64):
    """Calculate the circular distance between two positions in the sequence."""
    direct_distance = abs(pos1 - pos2)
    return min(direct_distance, sequence_length - direct_distance)

def circular_hamming_distance(hex1, hex2, pos1, pos2):
    """Calculate the Hamming distance between two hexagrams, considering their circular positions."""
    base_distance = hamming_distance(hex1, hex2)
    circ_distance = circular_distance(pos1, pos2)
    return base_distance, circ_distance

def get_circular_neighbors(pos, radius=1, sequence_length=64):
    """Get the positions of neighbors within a given radius in the circular sequence."""
    neighbors = []
    for i in range(-radius, radius + 1):
        if i == 0:
            continue
        neighbor_pos = (pos + i) % sequence_length
        neighbors.append(neighbor_pos)
    return neighbors

def calculate_circular_hamming_distances(sequence):
    """Calculate Hamming distances between consecutive hexagrams, including wrap-around."""
    distances = []
    n = len(sequence)
    for i in range(n):
        current = sequence[i]
        next_hex = sequence[(i + 1) % n]  # Wrap around to beginning
        distances.append(hamming_distance(current, next_hex))
    return distances

def generate_random_sequence(n, length=6):
    """Generate a random sequence of binary strings."""
    return [''.join(random.choice('01') for _ in range(length)) for _ in range(n)]

def invert(hexagram):
    """Invert a hexagram (change all 0s to 1s and vice versa)."""
    return ''.join('1' if bit == '0' else '0' for bit in hexagram)

def reverse(hexagram):
    """Reverse the order of lines in a hexagram."""
    return hexagram[::-1]

def complementary(hexagram):
    """Get the complementary hexagram (sum of positions = 63)."""
    return bin(63 - int(hexagram, 2))[2:].zfill(6)

def get_circular_transformations(sequence):
    """Get all transformations between consecutive hexagrams, including wrap-around."""
    transformations = []
    n = len(sequence)
    for i in range(n):
        current = sequence[i]
        next_hex = sequence[(i + 1) % n]
        
        # Check for various transformations
        if next_hex == invert(current):
            trans = "Inversion"
        elif next_hex == reverse(current):
            trans = "Reversal"
        elif next_hex == complementary(current):
            trans = "Complementary"
        elif hamming_distance(current, next_hex) == 1:
            trans = "Single Line Change"
        elif hamming_distance(current, next_hex) == 2:
            trans = "Two Line Change"
        else:
            trans = "Other"
            
        transformations.append((i, (i + 1) % n, trans, current, next_hex))
    
    return transformations

def get_opposite_position(pos, sequence_length=64):
    """Get the position exactly opposite to the given position in the circular sequence."""
    return (pos + sequence_length // 2) % sequence_length

def get_quarter_positions(sequence_length=64):
    """Get the positions that divide the sequence into quarters."""
    return [0, sequence_length // 4, sequence_length // 2, 3 * sequence_length // 4] 