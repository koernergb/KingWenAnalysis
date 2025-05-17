import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def analyze_triadic_relationships(hexagrams):
    """Analyze each consecutive triplet of hexagrams for dialectical patterns, treating the sequence as circular."""
    triads = []
    n = len(hexagrams)
    for i in range(0, n, 3):
        thesis = hexagrams[i % n]
        antithesis = hexagrams[(i+1) % n]
        synthesis = hexagrams[(i+2) % n]
        thesis_contribution = sum(1 for a, c in zip(thesis, synthesis) if a == c)
        antithesis_contribution = sum(1 for b, c in zip(antithesis, synthesis) if b == c)
        unique_elements = sum(1 for a, b, c in zip(thesis, antithesis, synthesis) if c != a and c != b)
        triads.append({
            'positions': ((i % n)+1, ((i+1)%n)+1, ((i+2)%n)+1),
            'thesis': thesis,
            'antithesis': antithesis,
            'synthesis': synthesis,
            'thesis_contribution': thesis_contribution,
            'antithesis_contribution': antithesis_contribution,
            'unique_elements': unique_elements,
            'synthesis_ratio': thesis_contribution / antithesis_contribution if antithesis_contribution else float('inf')
        })
    return triads

def visualize_dialectical_flow(triads):
    """Create a visualization of how each triad builds upon previous ones."""
    # Extract metrics for plotting
    positions = [t['positions'][2] for t in triads]  # Position of synthesis
    thesis_contributions = [t['thesis_contribution'] for t in triads]
    antithesis_contributions = [t['antithesis_contribution'] for t in triads]
    unique_elements = [t['unique_elements'] for t in triads]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot stacked bar chart showing composition of each synthesis
    bar_width = 0.8
    bottom = np.zeros(len(triads))
    
    # Plot thesis contributions
    p1 = ax.bar(positions, thesis_contributions, bar_width, 
                label='From Thesis', color='skyblue', bottom=bottom)
    bottom += thesis_contributions
    
    # Plot antithesis contributions
    p2 = ax.bar(positions, antithesis_contributions, bar_width,
                label='From Antithesis', color='lightcoral', bottom=bottom)
    bottom += antithesis_contributions
    
    # Plot unique elements
    p3 = ax.bar(positions, unique_elements, bar_width,
                label='Unique Elements', color='lightgreen', bottom=bottom)
    
    # Customize plot
    ax.set_xlabel('Hexagram Position')
    ax.set_ylabel('Line Contributions')
    ax.set_title('Dialectical Development Throughout the King Wen Sequence')
    ax.set_xticks(positions)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

def visualize_dialectical_spiral(hexagrams):
    """Create a spiral visualization of dialectical development."""
    # Create a figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='polar')
    
    # Calculate cumulative complexity through the sequence
    complexity = []
    for i in range(len(hexagrams)):
        if i == 0:
            complexity.append(0)
        else:
            # Hamming distance from previous hexagram
            h_dist = sum(a != b for a, b in zip(hexagrams[i], hexagrams[i-1]))
            complexity.append(complexity[-1] + h_dist)
    
    # Normalize complexity for better visualization
    complexity = np.array(complexity) / max(complexity)
    
    # Create spiral coordinates
    theta = np.linspace(0, 6*np.pi, len(hexagrams))  # 3 full turns
    r = np.linspace(0.1, 1, len(hexagrams)) * (1 + complexity * 0.5)  # Radius increases with complexity
    
    # Plot points
    ax.scatter(theta, r, c=np.arange(len(hexagrams)), cmap='viridis', s=100)
    
    # Highlight triads with different colors
    for i in range(0, len(hexagrams) - 2, 3):
        ax.plot(theta[i:i+3], r[i:i+3], 'r-', linewidth=2)
        
        # Add text annotations for thesis-antithesis-synthesis
        if i % 9 == 0:  # Label every third triad for clarity
            ax.text(theta[i], r[i], f"T{i//3+1}", fontsize=10, ha='center', va='center')
            ax.text(theta[i+1], r[i+1], f"A{i//3+1}", fontsize=10, ha='center', va='center')
            ax.text(theta[i+2], r[i+2], f"S{i//3+1}", fontsize=10, ha='center', va='center')
    
    # Remove grid and axis ticks for cleaner visualization
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.title('Spiral Development of Dialectical Triads in King Wen Sequence')
    plt.tight_layout()
    plt.show()
    
    return fig

def visualize_hexagram_evolution_network(hexagrams):
    """Create a network visualization of hexagram evolution."""
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for each hexagram
    for i, hex_code in enumerate(hexagrams):
        G.add_node(i+1, code=hex_code)  # 1-indexed positions
    
    # Process triads - connect thesis and antithesis to synthesis
    for i in range(0, len(hexagrams) - 2, 3):
        thesis_pos = i + 1
        antithesis_pos = i + 2
        synthesis_pos = i + 3
        
        # Calculate contribution weights
        thesis = hexagrams[i]
        antithesis = hexagrams[i+1]
        synthesis = hexagrams[i+2]
        
        thesis_contribution = sum(1 for a, c in zip(thesis, synthesis) if a == c)
        antithesis_contribution = sum(1 for b, c in zip(antithesis, synthesis) if b == c)
        
        # Add weighted edges
        G.add_edge(thesis_pos, synthesis_pos, weight=thesis_contribution, type='thesis')
        G.add_edge(antithesis_pos, synthesis_pos, weight=antithesis_contribution, type='antithesis')
        
        # If not the first triad, connect previous synthesis to new thesis
        if i > 0:
            prev_synthesis_pos = i
            G.add_edge(prev_synthesis_pos, thesis_pos, weight=3, type='evolution')
    
    # Create layout
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Draw the network
    plt.figure(figsize=(14, 14))
    
    # Draw nodes with different sizes based on position
    node_sizes = [300 if n % 3 == 0 else 200 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color=list(G.nodes()), cmap='viridis')
    
    # Draw edges with different colors based on type
    thesis_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'thesis']
    antithesis_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'antithesis']
    evolution_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'evolution']
    
    nx.draw_networkx_edges(G, pos, edgelist=thesis_edges, edge_color='blue', width=1.5, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=antithesis_edges, edge_color='red', width=1.5, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=evolution_edges, edge_color='green', width=2, alpha=0.7)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title('Dialectical Evolution Network of the King Wen Sequence')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G

def analyze_bit_contributions(hexagrams):
    """Analyze how each bit position contributes to the dialectical synthesis."""
    # Number of triads
    n_triads = len(hexagrams) // 3
    
    # Matrix to store bit contributions (rows=triads, columns=bit positions)
    bit_sources = np.zeros((n_triads, 6, 3))  # 3 categories: from thesis, from antithesis, novel
    
    for t in range(n_triads):
        thesis = hexagrams[t*3]
        antithesis = hexagrams[t*3 + 1]
        synthesis = hexagrams[t*3 + 2]
        
        for bit_pos in range(6):
            if synthesis[bit_pos] == thesis[bit_pos]:
                bit_sources[t, bit_pos, 0] = 1  # From thesis
            elif synthesis[bit_pos] == antithesis[bit_pos]:
                bit_sources[t, bit_pos, 1] = 1  # From antithesis
            else:
                bit_sources[t, bit_pos, 2] = 1  # Novel element
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Positions for each triad
    triad_positions = np.arange(n_triads) + 1
    
    # Plot stacked bars for each bit position
    bit_names = ['Bottom', 'Second', 'Third', 'Fourth', 'Fifth', 'Top']
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    labels = ['From Thesis', 'From Antithesis', 'Novel Element']
    
    for bit_pos in range(6):
        # Calculate position offset for this bit
        offset = (bit_pos - 2.5) * 0.15
        
        # Plot contributions for this bit position
        bottom = np.zeros(n_triads)
        for source in range(3):
            values = bit_sources[:, bit_pos, source]
            plt.bar(triad_positions + offset, values, bottom=bottom, width=0.1,
                   color=colors[source], label=labels[source] if bit_pos == 0 else "")
            bottom += values
    
    # Add bit position legend
    for bit_pos in range(6):
        offset = (bit_pos - 2.5) * 0.15
        plt.scatter([], [], color='black', marker='s', 
                   label=f'Bit {bit_pos+1} ({bit_names[bit_pos]})')
    
    # Customize plot
    plt.xlabel('Triad Number')
    plt.ylabel('Bit Source')
    plt.title('Bit-Level Contributions to Synthesis in Each Triad')
    plt.xticks(triad_positions)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return bit_sources

def analyze_golden_ratio_in_triads(hexagrams):
    """Analyze if the thesis:antithesis contribution ratio approximates the Golden Ratio."""
    golden_ratio = (1 + np.sqrt(5)) / 2  # Approximately 1.618
    
    # Calculate contribution ratios for each triad
    ratios = []
    triad_positions = []
    
    for i in range(0, len(hexagrams) - 2, 3):
        thesis = hexagrams[i]
        antithesis = hexagrams[i+1]
        synthesis = hexagrams[i+2]
        
        thesis_contribution = sum(1 for a, c in zip(thesis, synthesis) if a == c)
        antithesis_contribution = sum(1 for b, c in zip(antithesis, synthesis) if b == c)
        
        # Avoid division by zero
        if antithesis_contribution > 0:
            ratio = thesis_contribution / antithesis_contribution
            ratios.append(ratio)
            triad_positions.append(i//3 + 1)  # 1-indexed triad number
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot contribution ratios
    plt.bar(triad_positions, ratios, color='skyblue', alpha=0.7)
    
    # Add horizontal line for Golden Ratio
    plt.axhline(y=golden_ratio, color='r', linestyle='--', 
               label=f'Golden Ratio (φ ≈ {golden_ratio:.3f})')
    
    # Calculate average ratio
    avg_ratio = np.mean(ratios)
    plt.axhline(y=avg_ratio, color='g', linestyle='-', 
               label=f'Average Ratio ({avg_ratio:.3f})')
    
    # Calculate proximity to Golden Ratio
    gr_proximity = [abs(r - golden_ratio) for r in ratios]
    avg_proximity = np.mean(gr_proximity)
    
    # Customize plot
    plt.xlabel('Triad Number')
    plt.ylabel('Thesis:Antithesis Contribution Ratio')
    plt.title(f'Golden Ratio in Triadic Development (Avg. Proximity: {avg_proximity:.3f})')
    plt.xticks(triad_positions)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return ratios, gr_proximity