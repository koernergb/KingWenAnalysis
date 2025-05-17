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
    circular_autocorrelation,
    circular_transformation_map,
    circular_similarity_plot,
    circular_network_embedding,
    fourier_analysis,
    circular_community_analysis,
    diametric_symmetry_analysis
)
from aesthetic_analysis import analyze_aesthetic_patterns
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

def main():
    """Main function to run all analyses of the King Wen sequence."""
    print("Running King Wen sequence analysis...")
    
    # Basic sequence analysis
    print("\n1. Basic Sequence Analysis")
    analyze_sequence()
    
    # Pattern analysis
    print("\n2. Pattern Analysis")
    analyze_patterns()
    
    # Long-range correlation analysis
    print("\n3. Long-range Correlation Analysis")
    analyze_long_range_correlations()
    
    # Map transformation analysis
    print("\n4. Map Transformation Analysis")
    analyze_map_transformations()
    
    # Cycle analysis
    print("\n5. Cycle Analysis")
    analyze_cycles()
    
    # Fulcrum analysis
    print("\n6. Fulcrum Analysis")
    analyze_fulcrums()
    
    # Hierarchical structure analysis
    print("\n7. Hierarchical Structure Analysis")
    analyze_hierarchical_structure()
    
    # Network analysis
    print("\n8. Network Analysis")
    analyze_transformation_network()
    
    # Permutation analysis
    print("\n9. Permutation Analysis")
    analyze_permutations()
    
    # Philosophical analysis
    print("\n10. Philosophical Analysis")
    analyze_philosophy()
    
    # Symmetry analysis
    print("\n11. Symmetry Analysis")
    analyze_symmetry()
    
    # Circular analysis
    print("\n12. Circular Analysis")
    print("\n12.1 Circular Autocorrelation")
    circular_autocorrelation()
    print("\n12.2 Circular Transformation Map")
    circular_transformation_map()
    print("\n12.3 Circular Similarity Plot")
    circular_similarity_plot()
    print("\n12.4 Circular Network Embedding")
    circular_network_embedding()
    print("\n12.5 Fourier Analysis")
    fourier_analysis()
    print("\n12.6 Circular Community Analysis")
    circular_community_analysis()
    print("\n12.7 Diametric Symmetry Analysis")
    diametric_symmetry_analysis()
    
    # Aesthetic analysis
    print("\n13. Aesthetic Analysis")
    analyze_aesthetic_patterns()
    
    # Dialectical analysis
    print("\n14. Dialectical Analysis")
    try:
        print("\n14.1 Analyzing Triadic Relationships")
        triads = analyze_triadic_relationships(king_wen_sequence)
        print(f"Found {len(triads)} triadic relationships")
        
        print("\n14.2 Visualizing Dialectical Flow")
        visualize_dialectical_flow(triads)
        
        print("\n14.3 Visualizing Dialectical Spiral")
        visualize_dialectical_spiral(king_wen_sequence)
        
        print("\n14.4 Visualizing Hexagram Evolution Network")
        G = visualize_hexagram_evolution_network(king_wen_sequence)
        print(f"Created evolution network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        print("\n14.5 Analyzing Bit Contributions")
        bit_sources = analyze_bit_contributions(king_wen_sequence)
        print("Analyzed bit-level contributions to synthesis")
        
        print("\n14.6 Analyzing Golden Ratio in Triads")
        ratios, gr_proximity = analyze_golden_ratio_in_triads(king_wen_sequence)
        print(f"Average proximity to Golden Ratio: {np.mean(gr_proximity):.3f}")
        
        # Store results for comprehensive analysis
        dialectical_results = {
            "triads": triads,
            "bit_sources": bit_sources,
            "golden_ratio_analysis": {
                "ratios": ratios,
                "proximity": gr_proximity
            }
        }
        
    except Exception as e:
        print(f"Error in dialectical analysis: {str(e)}")
        dialectical_results = None

    # Hypercube analysis
    print("\n15. Hypercube Analysis")
    try:
        print("\n15.1 Running Comprehensive Hypercube Analysis")
        hypercube_results = run_hypercube_analysis(king_wen_sequence)
        
        print("\n15.2 Analyzing Subspace Structure")
        subspace_analysis = analyze_subspace_structure(king_wen_sequence)
        print("Analyzed how sequence traverses different subspaces")
        
        print("\n15.3 Analyzing Distance Matrix")
        distance_analysis = analyze_distance_matrix_in_hypercube(king_wen_sequence)
        print(f"Average Hamming distance: {distance_analysis['avg_distance']:.4f}")
        
        # Store results for comprehensive analysis
        hypercube_results.update({
            "subspace_analysis": subspace_analysis,
            "distance_analysis": distance_analysis
        })
        
    except Exception as e:
        print(f"Error in hypercube analysis: {str(e)}")
        hypercube_results = None

def comprehensive_analysis():
    """Perform a comprehensive analysis of the King Wen sequence structure."""
    
    # Run all analyses
    print("1. Analyzing cycles...")
    G, cycles = analyze_cycles()
    
    print("\n2. Analyzing hierarchical structure...")
    similarity_matrix, Z = analyze_hierarchical_structure()
    
    print("\n3. Analyzing symmetry and balance...")
    symmetry_scores, balance = analyze_symmetry()
    
    print("\n4. Analyzing special positions...")
    transition_complexity, pivot_scores, houses = analyze_special_positions()
    
    print("\n5. Analyzing philosophical patterns...")
    trends, element_transitions, phil_houses = analyze_philosophical_patterns()
    
    print("\n6. Analyzing transformation network...")
    G_net, communities = analyze_transformation_network()
    
    print("\n7. Performing permutation analysis...")
    king_wen_metrics, df_random = permutation_analysis()
    
    # Run dialectical analysis
    print("\n8. Performing dialectical analysis...")
    try:
        dialectical_results = {
            "triads": analyze_triadic_relationships(king_wen_sequence),
            "bit_sources": analyze_bit_contributions(king_wen_sequence),
            "golden_ratio_analysis": analyze_golden_ratio_in_triads(king_wen_sequence)
        }
    except Exception as e:
        print(f"Error in dialectical analysis: {str(e)}")
        dialectical_results = None

    # Run hypercube analysis
    print("\n9. Performing hypercube analysis...")
    try:
        hypercube_results = run_hypercube_analysis(king_wen_sequence)
        subspace_analysis = analyze_subspace_structure(king_wen_sequence)
        distance_analysis = analyze_distance_matrix_in_hypercube(king_wen_sequence)
        hypercube_results.update({
            "subspace_analysis": subspace_analysis,
            "distance_analysis": distance_analysis
        })
    except Exception as e:
        print(f"Error in hypercube analysis: {str(e)}")
        hypercube_results = None
    
    # Integrate findings to identify overarching structure
    print("\n========================================")
    print("INTEGRATED ANALYSIS OF KING WEN SEQUENCE")
    print("========================================")
    
    # 1. Identify the most significant structural features
    significant_features = []
    
    # Check cycle structure
    if len(cycles) > 0:
        cycle_counts = {length: len(cycles_list) for length, cycles_list in cycles.items()}
        significant_lengths = [length for length, count in cycle_counts.items() 
                              if count > 5]  # Arbitrary threshold
        if significant_lengths:
            significant_features.append(f"Strong cyclical structure with {len(significant_lengths)} " +
                                       f"prominent cycle lengths: {significant_lengths}")
    
    # Check hierarchical structure
    if hasattr(Z, 'shape') and Z.shape[0] > 0:
        from scipy.cluster.hierarchy import fcluster
        clusters = fcluster(Z, 8, criterion='maxclust')  # 8 clusters for 8 houses
        cluster_sizes = Counter(clusters)
        if max(cluster_sizes.values()) > 10:  # Large clusters exist
            significant_features.append(f"Clear hierarchical structure with {len(cluster_sizes)} major clusters")
    
    # Check symmetry
    best_pivots = sorted(symmetry_scores.items(), key=lambda x: max(x[1].values()), reverse=True)
    best_score = max(best_pivots[0][1].values()) if best_pivots else 0
    if best_score > 0.3:  # Arbitrary threshold
        significant_features.append(f"Strong symmetry around pivot position {best_pivots[0][0]+1} " +
                                   f"with score {best_score:.2f}")
    
    # Check for special positions
    mean_complexity = np.mean(transition_complexity)
    std_complexity = np.std(transition_complexity)
    special_transitions = sum(1 for c in transition_complexity if abs(c - mean_complexity) > 1.5 * std_complexity)
    if special_transitions > 5:  # Arbitrary threshold
        significant_features.append(f"Contains {special_transitions} significant transition points")
    
    # Check for philosophical patterns
    if trends and len(trends) > 2:
        significant_features.append(f"Shows {len(trends)} distinct yin-yang trend phases")
    
    # Check community structure
    if communities and len(communities) > 1:
        community_sizes = [len(c) for c in communities]
        if max(community_sizes) > 10:  # Large communities exist
            significant_features.append(f"Contains {len(communities)} transformation communities")
    
    # Add dialectical findings
    if dialectical_results:
        # Check triadic relationships
        if dialectical_results["triads"]:
            avg_thesis_contribution = np.mean([t['thesis_contribution'] for t in dialectical_results["triads"]])
            avg_antithesis_contribution = np.mean([t['antithesis_contribution'] for t in dialectical_results["triads"]])
            if abs(avg_thesis_contribution - avg_antithesis_contribution) < 1:
                significant_features.append(f"Balanced thesis-antithesis contributions in triads")
        
        # Check Golden Ratio relationships
        if dialectical_results["golden_ratio_analysis"]:
            ratios, proximity = dialectical_results["golden_ratio_analysis"]
            if np.mean(proximity) < 0.1:  # Close to Golden Ratio
                significant_features.append(f"Strong presence of Golden Ratio relationships in triads")

    # Add hypercube findings
    if hypercube_results:
        # Check hypercube path efficiency
        if 'path_analysis' in hypercube_results:
            path_analysis = hypercube_results['path_analysis']
            if path_analysis['hypercube_edge_percentage'] > 50:  # More than 50% of edges are hypercube edges
                significant_features.append(f"Efficient hypercube traversal ({path_analysis['hypercube_edge_percentage']:.1f}% hypercube edges)")

        # Check subspace coverage
        if 'subspace_analysis' in hypercube_results:
            subspace_analysis = hypercube_results['subspace_analysis']
            for dim, data in subspace_analysis.items():
                avg_coverage = sum(d['coverage'] for d in data.values()) / len(data)
                if avg_coverage > 0.7:  # More than 70% coverage
                    significant_features.append(f"High coverage of {dim}D subspaces ({avg_coverage*100:.1f}%)")

        # Check distance distribution
        if 'distance_analysis' in hypercube_results:
            distance_analysis = hypercube_results['distance_analysis']
            if distance_analysis['avg_distance'] < 3:  # Average distance less than 3
                significant_features.append(f"Compact sequence in hypercube space (avg distance: {distance_analysis['avg_distance']:.2f})")
    
    # Check overall statistical uniqueness
    num_metrics = len(king_wen_metrics)
    standardized_scores = {}
    for metric, value in king_wen_metrics.items():
        random_values = df_random[metric].values
        mean = np.mean(random_values)
        std = np.std(random_values)
        if std > 0:
            z_score = (value - mean) / std
            standardized_scores[metric] = z_score
    
    uniqueness = np.sqrt(sum(z**2 for z in standardized_scores.values()))
    if uniqueness > 3.0:  # Highly significant
        significant_features.append(f"Extremely statistically unique ({uniqueness:.2f} std devs from random)")
    
    # Print significant structural features
    print("\nSignificant Structural Features:")
    for i, feature in enumerate(significant_features):
        print(f"{i+1}. {feature}")
    
    # 2. Synthesize a unified model of the sequence structure
    print("\nSynthesized Structural Model:")
    
    # Determine primary organizing principle
    principles = {
        'cyclical': sum(len(cycles_list) for cycles_list in cycles.values()) if isinstance(cycles, dict) else 0,
        'hierarchical': len(set(fcluster(Z, 8, criterion='maxclust'))) if hasattr(Z, 'shape') else 0,
        'symmetrical': best_score * 100 if best_score else 0,
        'philosophical': len(trends) if trends else 0,
        'communal': len(communities) if communities else 0,
        'dialectical': len(dialectical_results["triads"]) if dialectical_results else 0,
        'hypercube': hypercube_results['path_analysis']['hypercube_edge_percentage'] if hypercube_results and 'path_analysis' in hypercube_results else 0
    }
    
    primary_principle = max(principles.items(), key=lambda x: x[1])[0]
    
    print(f"Primary organizing principle: {primary_principle.title()}")
    print("The King Wen sequence appears to be structured as a")
    
    if primary_principle == 'cyclical':
        print("complex system of interlocking cycles that create meaningful transitions between hexagrams.")
    elif primary_principle == 'hierarchical':
        print("multi-level hierarchy with distinct clusters of related hexagrams.")
    elif primary_principle == 'symmetrical':
        print("symmetrical arrangement with mirroring around key pivot points.")
    elif primary_principle == 'philosophical':
        print("representation of philosophical principles relating to balance and transformation.")
    elif primary_principle == 'communal':
        print("network of communities connected by specific transformation types.")
    elif primary_principle == 'dialectical':
        print("dialectical progression of thesis-antithesis-synthesis relationships.")
    elif primary_principle == 'hypercube':
        print("efficient traversal of a 6D hypercube space with balanced subspace coverage.")
    
    # Detailed description of the structure
    print("\nDetailed description:")
    
    # Based on the integration of all analyses, construct a description
    description = [
        "The King Wen sequence exhibits a sophisticated structure that appears to be deliberately designed.",
        "",
        "The sequence contains:",
        f"- {len(cycles) if isinstance(cycles, dict) else 'Several'} types of cycles",
        f"- {len(set(fcluster(Z, 8, criterion='maxclust'))) if hasattr(Z, 'shape') else 'Multiple'} hierarchical clusters",
        f"- {special_transitions} significant transition points",
        f"- {len(trends) if trends else 'Several'} yin-yang balance phases",
        f"- {len(communities) if communities else 'Multiple'} transformation communities",
    ]
    
    # Add dialectical findings
    if dialectical_results:
        description.extend([
            f"- {len(dialectical_results['triads'])} triadic relationships",
            f"- {len(dialectical_results['golden_ratio_analysis'][0])} Golden Ratio relationships"
        ])

    # Add hypercube findings
    if hypercube_results:
        description.extend([
            f"- {hypercube_results['path_analysis']['hypercube_edge_percentage']:.1f}% hypercube edge usage",
            f"- {len(hypercube_results['subspace_analysis'])} dimensional subspace traversals"
        ])
    
    description.extend([
        "",
        "Key insights:"
    ])
    
    # Add specific insights based on the analyses
    if 'inversion' in king_wen_metrics:
        expected_inversions = np.mean(df_random['inversion'].values)
        if king_wen_metrics['inversion'] > expected_inversions * 1.5:
            description.append(f"- Heavy use of inversion transformations ({king_wen_metrics['inversion']} vs {expected_inversions:.1f} expected)")
    
    if 'reversal' in king_wen_metrics:
        expected_reversals = np.mean(df_random['reversal'].values)
        if king_wen_metrics['reversal'] > expected_reversals * 1.5:
            description.append(f"- Heavy use of reversal transformations ({king_wen_metrics['reversal']} vs {expected_reversals:.1f} expected)")
    
    if 'mutual_info_1' in king_wen_metrics:
        expected_mi = np.mean(df_random['mutual_info_1'].values)
        if king_wen_metrics['mutual_info_1'] > expected_mi * 1.5:
            description.append(f"- Strong relationships between consecutive hexagrams")
    
    # Add dialectical insights
    if dialectical_results:
        ratios, proximity = dialectical_results["golden_ratio_analysis"]
        if np.mean(proximity) < 0.1:
            description.append(f"- Strong presence of Golden Ratio relationships in triads")
        
        avg_thesis = np.mean([t['thesis_contribution'] for t in dialectical_results["triads"]])
        avg_antithesis = np.mean([t['antithesis_contribution'] for t in dialectical_results["triads"]])
        if abs(avg_thesis - avg_antithesis) < 1:
            description.append(f"- Balanced thesis-antithesis contributions in triads")

    # Add hypercube insights
    if hypercube_results:
        if 'distance_analysis' in hypercube_results:
            description.append(f"- Compact sequence in hypercube space (avg distance: {hypercube_results['distance_analysis']['avg_distance']:.2f})")
        
        if 'subspace_analysis' in hypercube_results:
            for dim, data in hypercube_results['subspace_analysis'].items():
                avg_coverage = sum(d['coverage'] for d in data.values()) / len(data)
                if avg_coverage > 0.7:
                    description.append(f"- High coverage of {dim}D subspaces ({avg_coverage*100:.1f}%)")
    
    # Print the description
    for line in description:
        print(line)
    
    # Final conclusion
    p_value = 1 - stats.chi2.cdf(uniqueness**2, len(standardized_scores))
    
    print("\nConclusion:")
    print(f"The King Wen sequence is highly structured, with a probability of approximately")
    print(f"1 in {1/p_value:.1e} that such structure would occur by chance.")
    print("This strongly suggests deliberate design based on sophisticated principles of")
    print("symmetry, transformation, and balance that reflect ancient Chinese philosophical concepts.")
    
    return {
        'cycles': cycles,
        'hierarchy': Z if hasattr(Z, 'shape') else None,
        'symmetry': symmetry_scores,
        'transitions': transition_complexity,
        'philosophy': trends,
        'communities': communities,
        'metrics': king_wen_metrics,
        'random_metrics': df_random,
        'uniqueness': uniqueness,
        'p_value': p_value,
        'dialectical_analysis': dialectical_results,
        'hypercube_analysis': hypercube_results
    }

def integrated_i_ching_analysis():
    """Comprehensive analysis integrating Marshall, Shao Yong, and Körtvélyessy approaches."""
    results = {}
    
    # 1. Structural Analysis
    results["trigram_polarities"] = analyze_trigram_polarities()
    results["nuclear_hexagrams"] = analyze_nuclear_hexagrams()
    
    # 2. Circular/Octagonal Analysis
    results["octagonal_symmetry"] = analyze_octagonal_symmetry()
    results["shao_yong_comparison"] = compare_king_wen_to_shao_yong()
    
    # 3. Mathematical/Vector Analysis
    results["vector_transformations"] = vector_representation_analysis()
    results["transformation_groups"] = analyze_transformation_groups()
    
    # 4. Combined Analysis - Finding intersections between different approaches
    results["synthesis"] = {}
    
    # Compare mathematical communities with traditional houses
    results["synthesis"]["community_house_overlap"] = compare_communities_to_houses()
    
    # Look for correlations between vector transformations and nuclear relationships
    results["synthesis"]["nuclear_vector_correlation"] = correlate_nuclear_and_vector_patterns()
    
    # Analyze how circular patterns relate to vector transformations
    results["synthesis"]["circular_vector_patterns"] = analyze_circular_vector_patterns()
    
    return results

if __name__ == "__main__":
    main()