#!/usr/bin/env python3
"""
Assignment 0: Network Analysis of Graph B (Sysmon Provenance)
Student: [Your Name]
Course: Network Analysis
Description: Characterization of a heterogeneous forensic network derived from Splunk attack_data.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# ============================== CONFIGURATION ==============================
# Percorso relativo al grafo GEXF generato dallo script v5
GRAPH_PATH = "../output/v5_graphB.gexf"
RESULTS_FILE = "results.json"
FIGURES_DIR = "figures"

# Parametri di analisi
SEED = 42
PAGE_RANK_ALPHA = 0.85
ASP_SAMPLE_SIZE = 1000  # Numero di nodi campione per stimare l'average shortest path

# Impostazioni grafici
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# ============================== DATA LOADING ==============================
def load_graph(path):
    """Carica il grafo da file GEXF."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Grafo non trovato: {path}. Assicurati di aver eseguito lo script di build.")
    
    print(f"Loading graph from {path}...")
    G = nx.read_gexf(path)
    
    # Conversione attributi numerici (GEXF li salva spesso come stringhe)
    for u, v, data in G.edges(data=True):
        if 'weight' in data:
            data['weight'] = int(data['weight'])
        # Rinomina edge_type se presente (fix dello script builder)
        if 'edge_type' in data and 'type' not in data:
            data['type'] = data['edge_type']
            
    for n, data in G.nodes(data=True):
        if 'type' not in data:
            # Fallback per nodi senza tipo (dovrebbe non succedere col builder v5)
            prefix = n.split('::')[0] if '::' in n else 'unknown'
            data['type'] = prefix
            
    print(f"Graph Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def prepare_graph_views(G_raw):
    """
    Prepara le viste del grafo:
    1. Raw Directed: Grafo originale eterogeneo.
    2. Simple Undirected: Grafo strutturale semplice, senza loop, non diretto.
    """
    views = {}
    views['raw_directed'] = G_raw
    
    # Creazione vista semplice undirected
    G_simple = nx.Graph()
    G_simple.add_nodes_from(G_raw.nodes(data=True))
    
    for u, v in G_raw.edges():
        if u != v:  # Rimuovi self-loop
            # Se ci sono multi-archi, ne teniamo uno solo (grafo semplice)
            if not G_simple.has_edge(u, v):
                G_simple.add_edge(u, v)
                
    views['simple_undirected'] = G_simple
    
    print(f"Views prepared: Directed ({G_raw.number_of_nodes()}n, {G_raw.number_of_edges()}m), "
          f"Undirected Simple ({G_simple.number_of_nodes()}n, {G_simple.number_of_edges()}m)")
    return views

# ============================== ANALYSIS FUNCTIONS ==============================

def compute_composition(G_raw, G_simple):
    """Calcola metriche di composizione base."""
    print("Computing Composition...")
    stats = {}
    
    # Node types
    type_counts = Counter(d.get('type', 'unknown') for _, d in G_raw.nodes(data=True))
    stats['node_types'] = dict(type_counts)
    stats['total_nodes'] = G_raw.number_of_nodes()
    stats['total_edges'] = G_raw.number_of_edges()
    
    # Edge types (dal grafo diretto)
    edge_types = Counter(d.get('type', 'unknown') for _, _, d in G_raw.edges(data=True))
    stats['edge_types'] = dict(edge_types)
    
    # Density
    stats['density_directed'] = round(nx.density(G_raw), 6)
    stats['density_undirected'] = round(nx.density(G_simple), 6)
    
    # Connected Components (Directed -> WCC)
    stats['num_wcc'] = nx.number_weakly_connected_components(G_raw)
    
    # Largest WCC info
    wccs = list(nx.weakly_connected_components(G_raw))
    largest_wcc = max(wccs, key=len)
    stats['largest_wcc_size'] = len(largest_wcc)
    stats['largest_wcc_fraction'] = round(len(largest_wcc) / G_raw.number_of_nodes(), 4)
    
    # Self loops count in raw
    self_loops = sum(1 for u, v in G_raw.edges() if u == v)
    stats['self_loops'] = self_loops
    
    return stats

def compute_degree_metrics(G_simple):
    """Calcola statistiche sul grado (vista undirected)."""
    print("Computing Degree Metrics...")
    degrees = [d for n, d in G_simple.degree()]
    deg_array = np.array(degrees)
    
    stats = {
        "mean_degree": round(float(np.mean(deg_array)), 6),
        "median_degree": float(np.median(deg_array)),
        "max_degree": int(np.max(deg_array)),
        "min_degree": int(np.min(deg_array)),
        "variance": round(float(np.var(deg_array)), 6),
        "std_dev": round(float(np.std(deg_array)), 6)
    }
    
    # Quantili
    stats['quantiles'] = {
        "25%": float(np.percentile(deg_array, 25)),
        "75%": float(np.percentile(deg_array, 75)),
        "90%": float(np.percentile(deg_array, 90)),
        "99%": float(np.percentile(deg_array, 99))
    }
    
    # Top Hubs
    top_hubs = sorted(G_simple.degree(), key=lambda x: x[1], reverse=True)[:10]
    # Estraiamo solo il nome pulito per il JSON
    stats['top_hubs'] = [{"node": str(n), "degree": int(d)} for n, d in top_hubs]
    
    # Distribuzione per istogramma/plot
    stats['degree_histogram'] = dict(Counter(degrees))
    
    return stats

def compute_centralities(G_raw, G_simple):
    """Calcola centralità strutturali."""
    print("Computing Centralities...")
    results = {}
    
    # 1. Degree Centrality (Undirected)
    print("  Computing Degree Centrality...")
    deg_cent = nx.degree_centrality(G_simple)
    top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:5]
    results['degree_centrality_top'] = [{"node": str(n), "score": round(v, 6)} for n, v in top_deg]
    
    # 2. PageRank (Directed) - Importante per grafi diretti
    print(f"  Computing PageRank (alpha={PAGE_RANK_ALPHA})...")
    try:
        pr = nx.pagerank(G_raw, alpha=PAGE_RANK_ALPHA, max_iter=100)
        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
        results['pagerank_top'] = [{"node": str(n), "score": round(v, 6)} for n, v in top_pr]
        
        # Nodo top assoluto
        top_node = top_pr[0][0]
        results['pagerank_top_node'] = str(top_node)
    except Exception as e:
        print(f"  Warning: PageRank failed: {e}")
        results['pagerank_top'] = []
        results['pagerank_top_node'] = None

    # 3. Betweenness (Opzionale, costoso, facciamo solo top approssimato o saltiamo se troppo lento)
    # Per A0 bastano Degree e PageRank, ma aggiungiamo un campione di Betweenness se veloce
    # Saltiamo Betweenness esatto su 49k nodi per non bloccare.
    
    return results

def compute_paths(G_simple):
    """
    Calcola metriche sui percorsi nella Largest Connected Component (LCC).
    Usa campionamento per Average Shortest Path per motivi di performance.
    """
    print("Computing Paths...")
    
    # Identifica LCC
    print("  Identifying Largest Connected Component (LCC)...")
    connected_components = list(nx.connected_components(G_simple))
    largest_cc = max(connected_components, key=len)
    subgraph = G_simple.subgraph(largest_cc).copy()
    n_nodes = subgraph.number_of_nodes()
    
    print(f"  Computing path metrics on LCC ({n_nodes} nodes) via sampling...")
    
    results = {
        "component_used": "Largest Connected Component (Simple Undirected View)",
        "component_nodes": n_nodes
    }
    
    # 1. Average Shortest Path Length (Stima campionata)
    # Il calcolo esatto O(N^2) è proibitivo per 43k nodi in Python puro senza parallelizzazione
    print(f"    Sampling {ASP_SAMPLE_SIZE} sources for ASP estimation...")
    sampled_nodes = random.sample(list(subgraph.nodes()), min(ASP_SAMPLE_SIZE, n_nodes))
    
    total_len = 0
    count = 0
    
    for source in sampled_nodes:
        lengths = nx.single_source_shortest_path_length(subgraph, source)
        total_len += sum(lengths.values())
        count += len(lengths)
        
    avg_path = total_len / count if count > 0 else 0
    results['avg_shortest_path_estimated'] = round(avg_path, 6)
    
    # 2. Diameter e Radius (Approssimati o calcolati se piccoli)
    # Per grafi sparse grandi, diameter è spesso piccolo (<20). 
    # Usiamo un approccio a due passi: lower bound da eccentricità campionate, poi verifica se necessario.
    # Per A0, una stima robusta o il calcolo esatto se rapido (grafi small-world) va bene.
    # Proviamo a calcolare le eccentricità per i nodi campionati per stimare diametro/raggio.
    
    print("    Estimating Diameter and Radius via sampling...")
    eccentricities = []
    for source in sampled_nodes:
        lengths = nx.single_source_shortest_path_length(subgraph, source)
        if len(lengths) == n_nodes: # Solo se raggiungibile tutto (dovrebbe esserlo nella LCC)
            ecc = max(lengths.values())
            eccentricities.append(ecc)
            
    if eccentricities:
        results['diameter_estimated'] = int(max(eccentricities))
        results['radius_estimated'] = int(min(eccentricities))
    else:
        results['diameter_estimated'] = -1
        results['radius_estimated'] = -1
        
    return results

def compute_clustering(G_simple):
    """Calcola coefficiente di clustering e transitivity."""
    print("Computing Clustering...")
    
    # Average Clustering Coefficient
    avg_clust = nx.average_clustering(G_simple)
    
    # Transitivity (Global Clustering Coefficient)
    transitivity = nx.transitivity(G_simple)
    
    return {
        "average_clustering_coefficient": round(avg_clust, 6),
        "transitivity": round(transitivity, 6)
    }

def compute_assortativity(G_raw, G_simple):
    """
    Calcola assortatività per grado e per categoria (tipo nodo).
    Implementa manualmente la mixing matrix per compatibilità.
    """
    print("Computing Assortativity...")
    results = {}

    # 1. Degree Assortativity (Undirected)
    try:
        deg_assort = nx.degree_assortativity_coefficient(G_simple)
        results['degree_assortativity'] = round(deg_assort, 6)
        print(f"  Degree Assortativity (Simple): {results['degree_assortativity']}")
    except Exception as e:
        results['degree_assortativity'] = None
        print(f"  Warning: Could not compute degree assortativity: {e}")

    # 2. Categorical Assortativity (Node Type)
    # Assicurati che tutti i nodi abbiano l'attributo 'type'
    for n in G_raw.nodes():
        if 'type' not in G_raw.nodes[n]:
            G_raw.nodes[n]['type'] = 'unknown'
    for n in G_simple.nodes():
        if 'type' not in G_simple.nodes[n]:
            G_simple.nodes[n]['type'] = 'unknown'
            
    # A. Directed Graph
    try:
        cat_assort_dir = nx.attribute_assortativity_coefficient(G_raw, 'type')
        results['categorical_assortativity_directed'] = round(cat_assort_dir, 6)
        print(f"  Categorical Assortativity (Directed): {results['categorical_assortativity_directed']}")
    except Exception as e:
        results['categorical_assortativity_directed'] = None
        print(f"  Warning: Could not compute directed categorical assortativity: {e}")

    # B. Undirected Graph
    try:
        cat_assort_undir = nx.attribute_assortativity_coefficient(G_simple, 'type')
        results['categorical_assortativity_undirected'] = round(cat_assort_undir, 6)
        print(f"  Categorical Assortativity (Undirected): {results['categorical_assortativity_undirected']}")
    except Exception as e:
        results['categorical_assortativity_undirected'] = None
        print(f"  Warning: Could not compute undirected categorical assortativity: {e}")

    # 3. Mixing Matrix (Manual Implementation)
    # Calcolata sul grafo Diretto per preservare la semantica Source->Target
    print("  Computing Type Mixing Matrix (Directed)...")
    
    types = sorted(list(set(d.get('type', 'unknown') for _, d in G_raw.nodes(data=True))))
    type_to_idx = {t: i for i, t in enumerate(types)}
    n_types = len(types)
    
    mix_mat = np.zeros((n_types, n_types), dtype=int)
    
    for u, v in G_raw.edges():
        t_u = G_raw.nodes[u].get('type', 'unknown')
        t_v = G_raw.nodes[v].get('type', 'unknown')
        
        i = type_to_idx[t_u]
        j = type_to_idx[t_v]
        mix_mat[i, j] += 1
        
    results['mixing_matrix_raw'] = {
        "labels": types,
        "matrix": mix_mat.tolist()
    }
    
    # Normalizzata per riga
    mix_mat_norm = mix_mat.astype(float)
    row_sums = mix_mat_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    mix_mat_norm = mix_mat_norm / row_sums
    
    results['mixing_matrix_normalized'] = {
        "labels": types,
        "matrix": np.round(mix_mat_norm, 4).tolist(),
        "interpretation": "Rows = Source Type, Cols = Target Type. Values = P(Target|Source)"
    }
    
    return results

def compute_random_reference(G_simple):
    """
    Genera un grafo Erdős–Rényi (G_nm) con stessi N ed M per confronto.
    Confronta: Mean Degree, Variance, Clustering, Avg Path.
    """
    print("Computing Random Graph Reference (Erdős–Rényi)...")
    
    n = G_simple.number_of_nodes()
    m = G_simple.number_of_edges()
    
    # Grafo ER con stesso numero di nodi e archi
    G_er = nx.gnm_random_graph(n, m, seed=SEED)
    
    er_stats = {}
    er_stats['n'] = n
    er_stats['m'] = m
    
    # Degree stats
    er_degrees = [d for _, d in G_er.degree()]
    er_stats['mean_degree'] = round(float(np.mean(er_degrees)), 6)
    er_stats['degree_variance'] = round(float(np.var(er_degrees)), 6)
    
    # Clustering
    er_stats['avg_clustering'] = round(nx.average_clustering(G_er), 6)
    
    # Path length (solo su LCC di ER, spesso è già tutto connesso se denso, qui è sparso)
    er_cc = list(nx.connected_components(G_er))
    largest_er_cc = max(er_cc, key=len)
    sub_er = G_er.subgraph(largest_er_cc)
    
    if sub_er.number_of_nodes() > 100: # Campiona anche per ER se grande
        sample_nodes = random.sample(list(sub_er.nodes()), min(500, sub_er.number_of_nodes()))
        total_len = 0
        count = 0
        for s in sample_nodes:
            l = nx.single_source_shortest_path_length(sub_er, s)
            total_len += sum(l.values())
            count += len(l)
        er_stats['avg_shortest_path_estimated'] = round(total_len / count, 6)
    else:
        er_stats['avg_shortest_path_estimated'] = round(nx.average_shortest_path_length(sub_er), 6)
        
    return er_stats

# ============================== PLOTTING ==============================

def plot_results(results, output_dir):
    """Genera tutte le figure partendo dai risultati salvati."""
    print(f"Generating plots in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Recupera dati utili dai risultati
    comp = results['composition']
    deg = results['degree_metrics']
    assort = results['assortativity']
    
    # --- Figura 1: Composizione Nodi e Archi ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie Chart Nodi
    labels = list(comp['node_types'].keys())
    sizes = list(comp['node_types'].values())
    colors = sns.color_palette("pastel", len(labels))
    axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    axes[0].set_title("Node Type Composition")
    
    # Bar Chart Archi
    edge_labels = list(comp['edge_types'].keys())
    edge_sizes = list(comp['edge_types'].values())
    axes[1].bar(range(len(edge_labels)), edge_sizes, color='steelblue')
    axes[1].set_xticks(range(len(edge_labels)))
    axes[1].set_xticklabels(edge_labels, rotation=45, ha='right')
    axes[1].set_ylabel("Count")
    axes[1].set_title("Edge Type Composition")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_composition.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "01_composition.pdf"))
    plt.close()
    print("  Saved 01_composition")

    # --- Figura 2: Degree Distribution (Log-Log) + Top Hubs ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # CCDF Log-Log
    hist = deg['degree_histogram']
    ks = sorted([int(k) for k in hist.keys()])
    # Calcolo CCDF
    total = sum(hist.values())
    ccdf = []
    current_sum = 0
    # Istogramma cumulativo inverso
    sorted_counts = sorted(hist.items(), key=lambda x: int(x[0]))
    cumulative = 0
    ccdf_vals = []
    keys_vals = []
    
    # Ricostruisci lista gradi per calcolare CCDF correttamente
    all_degs = []
    for k, v in hist.items():
        all_degs.extend([int(k)] * v)
    all_degs.sort()
    
    unique_degs = sorted(list(set(all_degs)))
    ccdf_probs = []
    for d in unique_degs:
        count_ge = sum(1 for x in all_degs if x >= d)
        ccdf_probs.append(count_ge / len(all_degs))
        
    axes[0].loglog(unique_degs, ccdf_probs, marker='.', linestyle='None', color='darkred', alpha=0.7)
    axes[0].set_xlabel("Degree k (log)")
    axes[0].set_ylabel("P(K >= k) (log)")
    axes[0].set_title("Degree Distribution (CCDF)")
    axes[0].grid(True, which="both", ls="-", alpha=0.4)
    
    # Top Hubs
    hubs = deg['top_hubs']
    hub_names = [h['node'].split('::')[-1][:20] for h in hubs] # Pulisci nome
    hub_vals = [h['degree'] for h in hubs]
    # Inverti per barh
    axes[1].barh(range(len(hubs)), hub_vals, color='coral')
    axes[1].set_yticks(range(len(hubs)))
    axes[1].set_yticklabels(hub_names, fontsize=9)
    axes[1].set_xlabel("Degree")
    axes[1].set_title("Top 10 Structural Hubs")
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_degree_structure.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "02_degree_structure.pdf"))
    plt.close()
    print("  Saved 02_degree_structure")

    # --- Figura 3: Type Mixing Heatmap ---
    if 'mixing_matrix_normalized' in assort:
        mat_data = np.array(assort['mixing_matrix_normalized']['matrix'])
        labels = assort['mixing_matrix_normalized']['labels']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(mat_data, annot=True, fmt=".2f", cmap="YlOrRd", 
                    xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Probability'})
        plt.title("Normalized Type Mixing Matrix (Source -> Target)")
        plt.xlabel("Target Type")
        plt.ylabel("Source Type")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_mixing_matrix.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "03_mixing_matrix.pdf"))
        plt.close()
        print("  Saved 03_mixing_matrix")

    # --- Figura 4: Empirical vs ER Baseline ---
    er = results['random_reference']
    emp = results['paths']
    emp_clust = results['clustering']
    
    metrics = ['Avg Path Length', 'Clustering Coeff']
    emp_vals = [emp['avg_shortest_path_estimated'], emp_clust['average_clustering_coefficient']]
    er_vals = [er['avg_shortest_path_estimated'], er['avg_clustering']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, emp_vals, width, label='Empirical (Graph B)', color='navy')
    rects2 = ax.bar(x + width/2, er_vals, width, label='Erdős–Rényi Baseline', color='orange')
    
    ax.set_ylabel('Value')
    ax.set_title('Empirical vs Random Graph Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Log scale per clustering se molto piccolo? No, meglio lineare se confrontato con ER
    # Ma il clustering è molto diverso in scala, forse due subplot o scale diverse?
    # Per semplicità A0, lasciamo così, evidenzia la differenza enorme nel clustering.
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_empirical_vs_er.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "04_empirical_vs_er.pdf"))
    plt.close()
    print("  Saved 04_empirical_vs_er")

    print("All plots generated successfully.")

# ============================== MAIN EXECUTION ==============================

def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Assignment 0 Network Analysis")
    parser.add_argument('--plot-only', action='store_true', help="Rigenera solo i grafici da results.json")
    args = parser.parse_args()
    
    if args.plot_only:
        if not os.path.exists(RESULTS_FILE):
            print(f"Errore: {RESULTS_FILE} non trovato. Esegui prima l'analisi completa.")
            sys.exit(1)
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        plot_results(results, FIGURES_DIR)
        return

    # --- Full Analysis Pipeline ---
    print("--- Starting Full Analysis ---")
    random.seed(SEED)
    np.random.seed(SEED)
    
    try:
        G_raw = load_graph(GRAPH_PATH)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
        
    views = prepare_graph_views(G_raw)
    G_simple = views['simple_undirected']
    
    results = {}
    
    # 1. Composition
    results['composition'] = compute_composition(G_raw, G_simple)
    
    # 2. Degree Metrics
    results['degree_metrics'] = compute_degree_metrics(G_simple)
    
    # 3. Centralities
    results['centralities'] = compute_centralities(G_raw, G_simple)
    
    # 4. Paths
    results['paths'] = compute_paths(G_simple)
    
    # 5. Clustering
    results['clustering'] = compute_clustering(G_simple)
    
    # 6. Assortativity
    results['assortativity'] = compute_assortativity(G_raw, G_simple)
    
    # 7. Random Reference
    results['random_reference'] = compute_random_reference(G_simple)
    
    # Save Results
    save_results(results, RESULTS_FILE)
    
    # Generate Plots
    plot_results(results, FIGURES_DIR)
    
    print("\n=== Analysis Complete ===")
    print(f"Key Findings:")
    print(f" - Nodes: {results['composition']['total_nodes']}, Edges: {results['composition']['total_edges']}")
    print(f" - Avg Degree: {results['degree_metrics']['mean_degree']}")
    print(f" - Avg Path (est): {results['paths']['avg_shortest_path_estimated']}")
    print(f" - Clustering: {results['clustering']['average_clustering_coefficient']}")
    print(f" - Degree Assortativity: {results['assortativity']['degree_assortativity']}")

if __name__ == "__main__":
    main()