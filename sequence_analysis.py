import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from utils import (
    king_wen_sequence, king_wen_integers,
    calculate_circular_hamming_distances, generate_random_sequence,
    circular_distance, get_circular_neighbors, get_opposite_position,
    hamming_distance
)

def calculate_sequence_entropy(sequence, window_size=3):
    """Calculate the sliding window entropy of the sequence, considering circularity."""
    entropies = []
    n = len(sequence)
    for i in range(n):
        # Get window elements, wrapping around if necessary
        window = [sequence[(i + j) % n] for j in range(window_size)]
        # Convert window elements to unique integers
        window_ints = [int(x, 2) for x in window]
        # Calculate probabilities
        values, counts = np.unique(window_ints, return_counts=True)
        probabilities = counts / len(window_ints)
        entropies.append(entropy(probabilities, base=2))
    return entropies

def calculate_mutual_information(sequence, max_distance=32):  # Changed to 32 for circular analysis
    """Calculate mutual information between hexagrams at different circular distances."""
    sequence_ints = [int(x, 2) for x in sequence]
    mutual_info = []
    n = len(sequence)
    
    for distance in range(1, max_distance + 1):
        sequence1 = sequence_ints
        sequence2 = [sequence_ints[(i + distance) % n] for i in range(n)]
        mi = mutual_info_score(sequence1, sequence2)
        mutual_info.append(mi)
    
    return mutual_info

def analyze_sequence():
    """Analyze the King Wen sequence using circular information theory metrics."""
    print("\nRunning sequence analysis...")
    return analyze_king_wen_sequence()

def analyze_king_wen_sequence():
    """Analyze the King Wen sequence using circular information theory metrics."""
    # Calculate circular Hamming distances
    hamming_distances = calculate_circular_hamming_distances(king_wen_sequence)
    
    # Generate random sequences for comparison
    num_random_sequences = 100
    random_sequences = [generate_random_sequence(len(king_wen_sequence)) for _ in range(num_random_sequences)]
    random_hamming_distances = [calculate_circular_hamming_distances(seq) for seq in random_sequences]
    avg_random_hamming = np.mean(random_hamming_distances, axis=0)
    
    # Calculate entropy of the sequence
    entropy_values = calculate_sequence_entropy(king_wen_sequence)
    random_entropies = [calculate_sequence_entropy(seq) for seq in random_sequences]
    avg_random_entropy = np.mean(random_entropies, axis=0)
    
    # Calculate mutual information
    mutual_info = calculate_mutual_information(king_wen_sequence)
    random_mutual_info = [calculate_mutual_information(seq) for seq in random_sequences]
    avg_random_mutual_info = np.mean(random_mutual_info, axis=0)
    
    # Plot the results
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot Hamming distances
    axs[0].plot(hamming_distances, 'b-', label='King Wen Sequence')
    axs[0].plot(avg_random_hamming, 'r--', label='Average Random Sequence')
    axs[0].set_title('Circular Hamming Distance Between Consecutive Hexagrams')
    axs[0].set_xlabel('Position in Sequence')
    axs[0].set_ylabel('Hamming Distance')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot entropy
    axs[1].plot(entropy_values, 'b-', label='King Wen Sequence')
    axs[1].plot(avg_random_entropy, 'r--', label='Average Random Sequence')
    axs[1].set_title('Circular Entropy of Sliding Window (Window Size = 3)')
    axs[1].set_xlabel('Starting Position of Window')
    axs[1].set_ylabel('Entropy (bits)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot mutual information
    axs[2].plot(range(1, len(mutual_info) + 1), mutual_info, 'b-', label='King Wen Sequence')
    axs[2].plot(range(1, len(avg_random_mutual_info) + 1), avg_random_mutual_info, 'r--', label='Average Random Sequence')
    axs[2].set_title('Circular Mutual Information Between Hexagrams at Different Distances')
    axs[2].set_xlabel('Circular Distance')
    axs[2].set_ylabel('Mutual Information')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()

    # Analyze transitions between hexagrams (circular)
    transition_matrix = np.zeros((64, 64))
    n = len(king_wen_integers)
    for i in range(n):
        from_state = king_wen_integers[i]
        to_state = king_wen_integers[(i + 1) % n]  # Wrap around
        transition_matrix[from_state, to_state] += 1
    
    # Calculate the probability of the King Wen sequence
    # compared to random permutations
    random_permutations = [np.random.permutation(king_wen_integers) for _ in range(1000)]
    
    # Calculate transition probabilities, avoiding division by zero
    row_sums = np.sum(transition_matrix, axis=1, keepdims=True)
    transition_probabilities = np.where(row_sums > 0, 
                                      transition_matrix / row_sums,
                                      0)
    
    king_wen_prob = 1.0
    for i in range(n):
        from_state = king_wen_integers[i]
        to_state = king_wen_integers[(i + 1) % n]  # Wrap around
        if transition_probabilities[from_state, to_state] > 0:
            king_wen_prob *= transition_probabilities[from_state, to_state]
    
    random_probs = []
    for perm in random_permutations:
        prob = 1.0
        for i in range(n):
            from_state = perm[i]
            to_state = perm[(i + 1) % n]  # Wrap around
            if transition_probabilities[from_state, to_state] > 0:
                prob *= transition_probabilities[from_state, to_state]
        random_probs.append(prob)
    
    print(f"Probability of King Wen sequence: {king_wen_prob}")
    print(f"Average probability of random permutations: {np.mean(random_probs)}")
    print(f"Percentage of random permutations with higher probability: {np.mean(np.array(random_probs) > king_wen_prob) * 100}%")

    # Create additional visualizations
    # Plot the distribution of random permutation probabilities
    plt.figure(figsize=(10, 6))
    plt.hist(np.log10(random_probs), bins=30, alpha=0.7, label='Random Permutations')
    plt.axvline(np.log10(king_wen_prob), color='red', linestyle='dashed', linewidth=2, label='King Wen Sequence')
    plt.title('Log Probability Distribution of Random Permutations vs King Wen Sequence (Circular)')
    plt.xlabel('Log10 Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Analyze opposite positions
    opposite_correlations = []
    for i in range(n):
        opposite_pos = get_opposite_position(i)
        h_dist = hamming_distance(king_wen_sequence[i], king_wen_sequence[opposite_pos])
        opposite_correlations.append(h_dist)
    
    plt.figure(figsize=(10, 6))
    plt.plot(opposite_correlations, 'b-')
    plt.title('Hamming Distance Between Opposite Positions in King Wen Sequence')
    plt.xlabel('Position')
    plt.ylabel('Hamming Distance')
    plt.grid(True)
    plt.show() 