import numpy as np
from scipy import stats
from collections import Counter
from sequence_analysis import analyze_sequence
from pattern_analysis import analyze_patterns
from long_range_correlation_analysis import analyze_long_range_correlations
from map_transformations import analyze_map_transformations
from cycle_detector import analyze_cycles
from fulcrum_detector import analyze_fulcrums
from hierarchical_analysis import analyze_hierarchical_structure
from network_analysis import analyze_transformation_network
from permutation_analysis import analyze_permutations
from philosophical_analysis import analyze_philosophy
from symmetry_analysis import analyze_symmetry
from circular_analysis import (
    circular_visualization,
    circular_distance_analysis,
    circular_symmetry_analysis,
    circular_networks,
    circular_transition_entropy,
    circular_opposite_analysis,
    circular_autocorrelation,
    circular_transformation_map,
    circular_similarity_plot,
    circular_network_embedding,
    fourier_analysis,
    circular_community_analysis,
    diametric_symmetry_analysis,
    run_all_circular_analyses
)
from aesthetic_analysis import (
    analyze_aesthetic_patterns,
    fibonacci_analysis,
    golden_ratio_analysis,
    aesthetic_proportion_analysis
)
from dialectical_analysis import (
    analyze_triadic_relationships,
    visualize_dialectical_flow,
    visualize_dialectical_spiral,
    visualize_hexagram_evolution_network,
    analyze_bit_contributions,
    analyze_golden_ratio_in_triads
)
from utils import king_wen_sequence
from hypercube_analysis import (
    create_hypercube_graph,
    analyze_king_wen_path,
    project_hypercube_to_3d,
    visualize_hypercube_and_king_wen,
    analyze_triadic_structure_in_hypercube,
    visualize_triads_in_hypercube,
    analyze_subspace_structure,
    analyze_distance_matrix_in_hypercube,
    run_hypercube_analysis
)
from statistical_validation import (
    run_statistical_validation,
    StatisticalValidator,
    generate_null_models,
    calculate_all_metrics,
    establish_falsifiable_hypotheses,
    comprehensive_metric_reporting,
    blind_analysis_protocol,
    pre_registered_hypothesis_test
)
from marshal_yong_kortvelyessy_analysis import (
    analyze_trigram_polarities,
    analyze_nuclear_hexagrams,
    vector_representation_analysis,
    analyze_transformation_groups,
    compare_king_wen_to_shao_yong,
    analyze_octagonal_symmetry
)

def run_all_analyses():
    print("\n================ ALL ANALYSES BEGIN ================\n")
    # --- Statistical Validation ---
    print("\n--- Statistical Validation ---\n")
    run_statistical_validation()
    validator = StatisticalValidator()
    validator.evaluate_sequence()
    generate_null_models(king_wen_sequence)
    calculate_all_metrics(king_wen_sequence)
    establish_falsifiable_hypotheses()
    comprehensive_metric_reporting(king_wen_sequence, generate_null_models(king_wen_sequence))
    blind_analysis_protocol(king_wen_sequence)
    pre_registered_hypothesis_test(king_wen_sequence)

    # --- Hypercube Analysis ---
    print("\n--- Hypercube Analysis ---\n")
    # Convert king_wen_sequence to 1-64 integers for hypercube analysis
    king_wen_sequence_int = [int(h, 2) + 1 for h in king_wen_sequence]
    hypercube = create_hypercube_graph()
    analyze_king_wen_path(king_wen_sequence_int, hypercube)
    coords_3d, _ = project_hypercube_to_3d(hypercube)
    visualize_hypercube_and_king_wen(hypercube, king_wen_sequence_int, coords_3d)
    triads = analyze_triadic_structure_in_hypercube(king_wen_sequence_int)
    visualize_triads_in_hypercube(triads, coords_3d)
    analyze_subspace_structure(king_wen_sequence_int)
    analyze_distance_matrix_in_hypercube(king_wen_sequence_int)
    run_hypercube_analysis(king_wen_sequence_int)

    # --- Circular Analysis ---
    print("\n--- Circular Analysis ---\n")
    circular_visualization()
    circular_distance_analysis()
    circular_symmetry_analysis()
    circular_networks()
    circular_transition_entropy()
    circular_opposite_analysis()
    circular_autocorrelation()
    circular_transformation_map()
    circular_similarity_plot()
    circular_network_embedding()
    fourier_analysis()
    circular_community_analysis()
    diametric_symmetry_analysis()
    run_all_circular_analyses()

    # --- Marshall-Yong-Körtvélyessy Analysis ---
    print("\n--- Marshall-Yong-Körtvélyessy Analysis ---\n")
    analyze_trigram_polarities()
    analyze_nuclear_hexagrams()
    vector_representation_analysis()
    analyze_transformation_groups()
    compare_king_wen_to_shao_yong()
    analyze_octagonal_symmetry()

    # --- Dialectical Analysis ---
    print("\n--- Dialectical Analysis ---\n")
    triads = analyze_triadic_relationships(king_wen_sequence)
    visualize_dialectical_flow(triads)
    visualize_dialectical_spiral(king_wen_sequence)
    visualize_hexagram_evolution_network(king_wen_sequence)
    analyze_bit_contributions(king_wen_sequence)
    analyze_golden_ratio_in_triads(king_wen_sequence)

    # --- Other Analyses (already present) ---
    print("\n--- Other Analyses ---\n")
    analyze_sequence()
    analyze_patterns()
    analyze_long_range_correlations()
    analyze_map_transformations()
    analyze_cycles()
    analyze_fulcrums()
    analyze_hierarchical_structure()
    analyze_transformation_network()
    analyze_permutations()
    analyze_philosophy()
    analyze_symmetry()
    analyze_aesthetic_patterns()
    fibonacci_analysis()
    golden_ratio_analysis()
    aesthetic_proportion_analysis()
    print("\n================ ALL ANALYSES COMPLETE ================\n")

def main():
    run_all_analyses()

if __name__ == "__main__":
    main()