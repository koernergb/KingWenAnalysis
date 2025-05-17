import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score
from utils import king_wen_sequence, hamming_distance

def analyze_permutations():
    """Analyze how unique the King Wen sequence is among all possible permutations."""
    print("\nRunning permutation analysis...")
    return permutation_analysis()

def permutation_analysis():
    """Analyze how unique the King Wen sequence is among all possible permutations."""
    
    # Define a set of metrics to evaluate sequence properties
    def calculate_metrics(sequence):
        metrics = {}
        
        # Hamming distance between consecutive hexagrams
        hamming_distances = []
        for i in range(len(sequence) - 1):
            hamming_distances.append(hamming_distance(sequence[i], sequence[i + 1]))
        
        metrics['mean_hamming'] = np.mean(hamming_distances)
        metrics['var_hamming'] = np.var(hamming_distances)
        
        # Count specific transformations
        transformation_counts = {
            'inversion': 0,
            'reversal': 0,
            'swap_trigrams': 0
        }
        
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_hex = sequence[i + 1]
            
            if next_hex == ''.join('1' if bit == '0' else '0' for bit in current):
                transformation_counts['inversion'] += 1
            if next_hex == current[::-1]:
                transformation_counts['reversal'] += 1
            if next_hex == current[3:] + current[:3]:
                transformation_counts['swap_trigrams'] += 1
        
        metrics.update(transformation_counts)
        
        # Mutual information between positions
        sequence_ints = [int(x, 2) for x in sequence]
        mutual_info = []
        
        for distance in range(1, min(11, len(sequence))):
            sequence1 = sequence_ints[:-distance]
            sequence2 = sequence_ints[distance:]
            mi = mutual_info_score(sequence1, sequence2)
            mutual_info.append(mi)
        
        metrics['mutual_info_1'] = mutual_info[0]
        metrics['mutual_info_avg'] = np.mean(mutual_info)
        
        # Entropy of subsequences
        entropies = []
        window_size = 3
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            window_ints = [int(x, 2) for x in window]
            values, counts = np.unique(window_ints, return_counts=True)
            probabilities = counts / len(window_ints)
            entropies.append(stats.entropy(probabilities, base=2))
        
        metrics['mean_entropy'] = np.mean(entropies)
        
        return metrics
    
    # Calculate metrics for the King Wen sequence
    king_wen_metrics = calculate_metrics(king_wen_sequence)
    
    # Generate random permutations and calculate their metrics
    n_permutations = 1000
    random_metrics = []
    
    for _ in range(n_permutations):
        random_sequence = np.random.permutation(king_wen_sequence).tolist()
        metrics = calculate_metrics(random_sequence)
        random_metrics.append(metrics)
    
    # Convert to DataFrame for easier analysis
    df_random = pd.DataFrame(random_metrics)
    
    # Compare King Wen to random permutations
    print("\nKing Wen Sequence vs Random Permutations:")
    print("=========================================")
    
    for metric, value in king_wen_metrics.items():
        random_values = df_random[metric].values
        percentile = stats.percentileofscore(random_values, value)
        extremeness = min(percentile, 100 - percentile)
        
        print(f"{metric}:")
        print(f"  King Wen value: {value:.4f}")
        print(f"  Random mean: {np.mean(random_values):.4f} ± {np.std(random_values):.4f}")
        print(f"  Percentile: {percentile:.1f}%")
        print(f"  Extremeness: {extremeness:.1f}% (0% = most extreme, 50% = average)")
        print()
    
    # Calculate an overall uniqueness score based on all metrics
    # Standardize each metric and calculate how many std devs away from mean
    standardized_scores = {}
    for metric, value in king_wen_metrics.items():
        random_values = df_random[metric].values
        mean = np.mean(random_values)
        std = np.std(random_values)
        if std > 0:
            z_score = (value - mean) / std
            standardized_scores[metric] = z_score
    
    # Overall uniqueness is the Euclidean distance of z-scores from origin
    uniqueness = np.sqrt(sum(z**2 for z in standardized_scores.values()))
    
    print(f"Overall uniqueness of King Wen sequence: {uniqueness:.2f} standard deviations from random")
    
    # Estimate probability of randomly generating a sequence with King Wen's properties
    # Assuming multivariate normal distribution of metrics
    num_metrics = len(standardized_scores)
    p_value = 1 - stats.chi2.cdf(uniqueness**2, num_metrics)
    
    print(f"Estimated p-value: {p_value:.10f}")
    print(f"Approximately 1 in {1/p_value:.1e} random permutations would have properties this extreme")
    
    # Visualize the comparison
    plt.figure(figsize=(15, 10))
    
    for i, (metric, value) in enumerate(king_wen_metrics.items()):
        plt.subplot(3, 3, i+1)
        
        random_values = df_random[metric].values
        plt.hist(random_values, bins=30, alpha=0.7)
        plt.axvline(x=value, color='red', linestyle='dashed', linewidth=2)
        
        plt.title(metric)
        plt.xlabel('Value')
        plt.ylabel('Count')
        
        if i >= 6:  # Only add legend to bottom row
            plt.legend(['King Wen', 'Random'], loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return king_wen_metrics, df_random