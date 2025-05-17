import numpy as np
import matplotlib.pyplot as plt
from utils import king_wen_sequence, hamming_distance

def analyze_autocorrelation():
    # Convert hexagrams to integers for computation
    sequence_ints = np.array([int(x, 2) for x in king_wen_sequence])
    
    # Calculate autocorrelation for different lags
    max_lag = 32  # Half the sequence length
    autocorr = []
    mean = np.mean(sequence_ints)
    var = np.var(sequence_ints)
    
    for lag in range(max_lag):
        # Calculate autocorrelation at this lag
        if lag == 0:
            corr = 1.0  # Autocorrelation at lag 0 is always 1
        else:
            # Ensure arrays have the same length
            x = sequence_ints[:-lag]
            y = sequence_ints[lag:]
            if len(x) == len(y):  # Only calculate if arrays have same length
                corr = np.corrcoef(x, y)[0, 1]
            else:
                corr = 0.0
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
                x = random_seq[:-lag]
                y = random_seq[lag:]
                if len(x) == len(y):
                    corr = np.corrcoef(x, y)[0, 1]
                else:
                    corr = 0.0
            random_corrs.append(corr)
        random_autocorrs.append(random_corrs)
    
    avg_random_autocorr = np.mean(random_autocorrs, axis=0)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_lag), autocorr, 'b-', label='King Wen Sequence')
    plt.plot(range(max_lag), avg_random_autocorr, 'r--', label='Average Random Sequence')
    plt.title('Autocorrelation at Different Lags')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_recurrence_plot():
    # Convert hexagrams to integers
    sequence_ints = np.array([int(x, 2) for x in king_wen_sequence])
    
    # Create recurrence matrix (1 if hexagrams are identical, 0 otherwise)
    n = len(sequence_ints)
    recurrence_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Can use different metrics (exact match, hamming distance threshold, etc.)
            if sequence_ints[i] == sequence_ints[j]:
                recurrence_matrix[i, j] = 1
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(recurrence_matrix, cmap='binary', aspect='equal')
    plt.colorbar(label='Recurrence')
    plt.title('Recurrence Plot of King Wen Sequence')
    plt.xlabel('Hexagram Position')
    plt.ylabel('Hexagram Position')
    plt.tight_layout()
    plt.show()
    
    # Create a similar plot but using Hamming distance with a threshold
    recurrence_matrix_hamming = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            h_dist = hamming_distance(king_wen_sequence[i], king_wen_sequence[j])
            if h_dist <= 2:  # Consider "similar" if 2 or fewer bits differ
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
    # Find patterns occurring at specific distances
    sequence_ints = [int(x, 2) for x in king_wen_sequence]
    n = len(sequence_ints)
    
    # For each distance/lag, count how often specific patterns occur
    max_lag = 32
    pattern_counts = {}
    
    for lag in range(1, max_lag + 1):
        patterns = []
        for i in range(n - lag):
            # Record the transformation between hexagrams at distance 'lag'
            # We can use various representations of the transformation
            # Here we'll use a simplified approach: (from_hex, to_hex)
            pattern = (sequence_ints[i], sequence_ints[i + lag])
            patterns.append(pattern)
        
        # Count unique patterns at this lag
        unique_patterns = set(patterns)
        pattern_counts[lag] = len(unique_patterns)
    
    # Generate same for random sequences
    random_pattern_counts = []
    for _ in range(50):
        random_seq = np.random.permutation(sequence_ints)
        random_counts = {}
        
        for lag in range(1, max_lag + 1):
            patterns = []
            for i in range(n - lag):
                pattern = (random_seq[i], random_seq[i + lag])
                patterns.append(pattern)
            
            random_counts[lag] = len(set(patterns))
        
        random_pattern_counts.append(random_counts)
    
    # Calculate average random pattern counts
    avg_random_counts = {}
    for lag in range(1, max_lag + 1):
        avg_random_counts[lag] = np.mean([counts[lag] for counts in random_pattern_counts])
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(list(pattern_counts.keys()), list(pattern_counts.values()), 'b-', label='King Wen Sequence')
    plt.plot(list(avg_random_counts.keys()), list(avg_random_counts.values()), 'r--', label='Average Random Sequence')
    plt.title('Number of Unique Transformation Patterns at Different Distances')
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