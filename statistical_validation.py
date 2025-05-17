import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from utils import king_wen_sequence, hamming_distance

# =====================
# Utility Functions
# =====================

def is_reversal(h1, h2):
    return h2 == h1[::-1]

def is_inversion(h1, h2):
    return h2 == ''.join('1' if bit == '0' else '0' for bit in h1)

def calculate_hamming_distance(h1, h2):
    return sum(a != b for a, b in zip(h1, h2))

def apply_reversal(h):
    return h[::-1]

def apply_inversion(h):
    return ''.join('1' if bit == '0' else '0' for bit in h)

def apply_hamming_change(h, d):
    h = list(h)
    idxs = np.random.choice(len(h), d, replace=False)
    for idx in idxs:
        h[idx] = '1' if h[idx] == '0' else '0'
    return ''.join(h)

# =====================
# Null Model Generation
# =====================

def generate_null_models(sequence, n_samples=1000):
    """
    Generate a set of null models with increasing constraints.
    """
    hexagrams = [format(i, '06b') for i in range(64)]
    null_models = {}

    # Model 1: Simple random permutation
    null_models["simple_random"] = [np.random.permutation(hexagrams).tolist() for _ in range(n_samples)]

    # Model 2: Pure trigram positions preserved
    pure_positions = [0, 1, 28, 29, 52, 53, 58, 59]
    null_models["pure_trigram_preserved"] = []
    for _ in range(n_samples):
        seq = hexagrams.copy()
        non_pure = [h for i, h in enumerate(hexagrams) if i not in pure_positions]
        np.random.shuffle(non_pure)
        for i in range(64):
            if i not in pure_positions:
                seq[i] = non_pure.pop()
        null_models["pure_trigram_preserved"].append(seq)

    # Model 3: Transition type preserved (Markov)
    transition_types = []
    for i in range(len(sequence) - 1):
        h1, h2 = sequence[i], sequence[i+1]
        if is_reversal(h1, h2):
            transition_types.append("reversal")
        elif is_inversion(h1, h2):
            transition_types.append("inversion")
        else:
            transition_types.append(f"hamming_{calculate_hamming_distance(h1, h2)}")
    markov_sequences = []
    for _ in range(n_samples):
        seq = [sequence[0]]
        shuffled_types = transition_types.copy()
        np.random.shuffle(shuffled_types)
        for t_type in shuffled_types:
            current = seq[-1]
            if t_type == "reversal":
                next_hex = apply_reversal(current)
            elif t_type == "inversion":
                next_hex = apply_inversion(current)
            else:
                h_dist = int(t_type.split("_")[1])
                next_hex = apply_hamming_change(current, h_dist)
            seq.append(next_hex)
        markov_sequences.append(seq)
    null_models["transition_preserved"] = markov_sequences

    return null_models

# =====================
# Metric Calculation Functions
# =====================

def calculate_all_metrics(sequence):
    """Calculate a comprehensive set of metrics for a sequence, treating it as circular."""
    metrics = {}
    # Example metrics (expand as needed)
    transformations = []
    n = len(sequence)
    for i in range(n):
        h1, h2 = sequence[i], sequence[(i+1) % n]  # Wrap around
        if is_reversal(h1, h2):
            transformations.append("reversal")
        elif is_inversion(h1, h2):
            transformations.append("inversion")
        else:
            transformations.append(f"hamming_{calculate_hamming_distance(h1, h2)}")
    metrics['reversal_count'] = transformations.count('reversal')
    metrics['inversion_count'] = transformations.count('inversion')
    metrics['hamming1_count'] = sum(1 for t in transformations if t == 'hamming_1')
    metrics['hamming2_count'] = sum(1 for t in transformations if t == 'hamming_2')
    metrics['hamming3_count'] = sum(1 for t in transformations if t == 'hamming_3')
    metrics['hamming4_count'] = sum(1 for t in transformations if t == 'hamming_4')
    # Add more metrics as needed...
    return metrics

# =====================
# Statistical Validation Class
# =====================

class StatisticalValidator:
    """Class for rigorous statistical validation of King Wen sequence patterns."""
    def __init__(self, sequence: List[str] = None):
        self.sequence = sequence or king_wen_sequence
        self.n_samples = 1000
    def evaluate_sequence(self, metrics: Optional[List[str]] = None) -> Dict:
        # Use generate_null_models and calculate_all_metrics
        null_models = generate_null_models(self.sequence, self.n_samples)
        results = {model_type: {} for model_type in null_models}
        king_wen_metrics = calculate_all_metrics(self.sequence)
        for model_type, sequences in null_models.items():
            for metric in king_wen_metrics:
                null_values = [calculate_all_metrics(seq)[metric] for seq in sequences]
                kw_value = king_wen_metrics[metric]
                mean = np.mean(null_values)
                std = np.std(null_values)
                z_score = (kw_value - mean) / std if std > 0 else float('inf')
                percentile = stats.percentileofscore(null_values, kw_value)
                p_value = 2 * min(percentile / 100, 1 - percentile / 100)
                results[model_type][metric] = {
                    'king_wen_value': kw_value,
                    'null_mean': mean,
                    'null_std': std,
                    'z_score': z_score,
                    'percentile': percentile,
                    'p_value': p_value
                }
        return results

# =====================
# Advanced Protocols
# =====================

def establish_falsifiable_hypotheses():
    """
    Create clearly falsifiable hypotheses about the King Wen sequence.
    """
    # Example: Add your hypothesis logic here
    pass

def comprehensive_metric_reporting(king_wen_sequence, null_models):
    """
    Report all metrics transparently, not just those showing significant results.
    """
    # Example: Add your reporting logic here
    pass

def blind_analysis_protocol(king_wen_sequence, n_random_sequences=10):
    """
    Implement a blind analysis protocol to reduce confirmation bias.
    """
    # Example: Add your blind analysis logic here
    pass

def pre_registered_hypothesis_test(king_wen_sequence):
    """
    Test specific pre-registered hypotheses to avoid post-hoc rationalization.
    """
    # Example: Add your pre-registration logic here
    pass

# =====================
# Main API
# =====================

def run_statistical_validation():
    """Run comprehensive statistical validation of the King Wen sequence."""
    validator = StatisticalValidator()
    results = validator.evaluate_sequence()
    print("\nStatistical Validation Results:")
    print("==============================")
    for model_type, metrics in results.items():
        print(f"\n{model_type.upper()} Model:")
        print("-" * len(model_type))
        for metric, stats in metrics.items():
            print(f"\n{metric}:")
            print(f"  King Wen value: {stats['king_wen_value']:.4f}")
            print(f"  Null mean: {stats['null_mean']:.4f}")
            print(f"  Null std: {stats['null_std']:.4f}")
            print(f"  Z-score: {stats['z_score']:.2f}")
            print(f"  P-value: {stats['p_value']:.4f}")
            # Add interpretation
            if stats['p_value'] < 0.05:
                print("  Interpretation: Statistically significant")
            else:
                print("  Interpretation: Not statistically significant")
    return results

if __name__ == "__main__":
    run_statistical_validation() 