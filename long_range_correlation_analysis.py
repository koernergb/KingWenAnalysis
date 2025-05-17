import numpy as np
import matplotlib.pyplot as plt
from utils import king_wen_sequence, hamming_distance

def analyze_autocorrelation():
    # Convert hexagrams to integers for computation
    sequence_ints = np.array([int(x, 2) for x in king_wen_sequence])
    n = len(sequence_ints)
    # Calculate autocorrelation for different lags (circular)
    max_lag = 32  # Half the sequence length
    autocorr = []
    mean = np.mean(sequence_ints)
    var = np.var(sequence_ints)
    for lag in range(max_lag):
        if lag == 0:
            corr = 1.0
        else:
            x = sequence_ints
            y = np.roll(sequence_ints, -lag)  # Circular shift
            corr = np.corrcoef(x, y)[0, 1]
        autocorr.append(corr)
    # Generate random sequences for comparison
    random_autocorrs = []
    for _ in range(100):
        random_seq = np.random.permutation(sequence_ints)
        random_corrs = []
        for lag in range(max_lag):
            if lag == 0:
                corr = 1.0
            else:
                x = random_seq
                y = np.roll(random_seq, -lag)
                corr = np.corrcoef(x, y)[0, 1]
            random_corrs.append(corr)
        random_autocorrs.append(random_corrs)
    avg_random_autocorr = np.mean(random_autocorrs, axis=0)
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_lag), autocorr, 'b-', label='King Wen Sequence')
    plt.plot(range(max_lag), avg_random_autocorr, 'r--', label='Average Random Sequence')
    plt.title('Autocorrelation at Different Lags (Circular)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_recurrence_plot():
    sequence_ints = np.array([int(x, 2) for x in king_wen_sequence])
    n = len(sequence_ints)
    recurrence_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if sequence_ints[i] == sequence_ints[j]:
                recurrence_matrix[i, j] = 1
    plt.figure(figsize=(10, 8))
    plt.imshow(recurrence_matrix, cmap='binary', aspect='equal')
    plt.colorbar(label='Recurrence')
    plt.title('Recurrence Plot of King Wen Sequence')
    plt.xlabel('Hexagram Position')
    plt.ylabel('Hexagram Position')
    plt.tight_layout()
    plt.show()
    # Circular Hamming recurrence
    recurrence_matrix_hamming = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            h_dist = hamming_distance(king_wen_sequence[i], king_wen_sequence[j])
            if h_dist <= 2:
                recurrence_matrix_hamming[i, j] = 1
    plt.figure(figsize=(10, 8))
    plt.imshow(recurrence_matrix_hamming, cmap='binary', aspect='equal')
    plt.colorbar(label='Similarity')
    plt.title('Similarity Plot of King Wen Sequence (Hamming Distance ≤ 2)')
    plt.xlabel('Hexagram Position')
    plt.ylabel('Hexagram Position')
    plt.tight_layout()
    plt.show()

def analyze_distance_patterns():
    sequence_ints = [int(x, 2) for x in king_wen_sequence]
    n = len(sequence_ints)
    max_lag = 32
    pattern_counts = {}
    for lag in range(1, max_lag + 1):
        patterns = []
        for i in range(n):
            # Circular: wrap around
            j = (i + lag) % n
            pattern = (sequence_ints[i], sequence_ints[j])
            patterns.append(pattern)
        unique_patterns = set(patterns)
        pattern_counts[lag] = len(unique_patterns)
    random_pattern_counts = []
    for _ in range(50):
        random_seq = np.random.permutation(sequence_ints)
        random_counts = {}
        for lag in range(1, max_lag + 1):
            patterns = []
            for i in range(n):
                j = (i + lag) % n
                pattern = (random_seq[i], random_seq[j])
                patterns.append(pattern)
            random_counts[lag] = len(set(patterns))
        random_pattern_counts.append(random_counts)
    avg_random_counts = {}
    for lag in range(1, max_lag + 1):
        avg_random_counts[lag] = np.mean([counts[lag] for counts in random_pattern_counts])
    plt.figure(figsize=(10, 6))
    plt.plot(list(pattern_counts.keys()), list(pattern_counts.values()), 'b-', label='King Wen Sequence')
    plt.plot(list(avg_random_counts.keys()), list(avg_random_counts.values()), 'r--', label='Average Random Sequence')
    plt.title('Number of Unique Transformation Patterns at Different Distances (Circular)')
    plt.xlabel('Distance Between Hexagrams')
    plt.ylabel('Number of Unique Patterns')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_long_range_correlations():
    """Analyze long-range correlations in the King Wen sequence."""
    print("\nRunning long-range correlation analysis...")
    
    # Run all correlation analyses
    analyze_autocorrelation()
    create_recurrence_plot()
    analyze_distance_patterns()