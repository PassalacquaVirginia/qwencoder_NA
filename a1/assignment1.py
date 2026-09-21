#!/usr/bin/env python3
"""
Assignment 1: Community Analysis of Graph B (Sysmon Provenance)
Students: Muce + Virgi
Course: Network Analysis
Description: Community detection and comparison on heterogeneous forensic network.
"""

import os
import sys
import json
import random
import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import igraph as ig
import leidenalg

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# ============================== CONFIGURATION ==============================
GRAPH_B_PATH = "../output/v5_graphB.gexf"
GRAPH_A_PATH = "../output/v5_graphA.gexf"
RESULTS_FILE = "results.json"
FIGURES_DIR = "figures"

SEED = 42
STABILITY_SEEDS = 20

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# ============================== DATA LOADING ==============================
def load_graphs():
    """Carica Graph B e Graph A."""
    print("Loading graphs...")
    
    if not os.path.exists(GRAPH_B_PATH):
        raise FileNotFoundError(f"Graph B non trovato: {GRAPH_B_PATH}")
    if not os.path.exists(GRAPH_A_PATH):
        raise FileNotFoundError(f"Graph A non trovato: {GRAPH_A_PATH}")
        
    G_b_raw = nx.read_gexf(GRAPH_B_PATH)
    G_a_raw = nx.read_gexf(GRAPH_A_PATH)
    
    # Pulizia attributi per compatibilità
    for G in [G_b_raw, G_a_raw]:
        for u, v, d in G.edges(data=True):
            if 'weight' in d:
                try:
                    d['weight'] = int(d['weight'])
                except:
                    d['weight'] = 1
            if 'edge_type' in d and 'type' not in d:
                d['type'] = d['edge_type']
                
        for n, d in G.nodes(data=True):
            if 'type' not in d:
                prefix = n.split('::')[0] if '::' in n else 'unknown'
                d['type'] = prefix
                
    # Preparazione vista strutturale per Graph B (semplice, undirected, unweighted, loop-free)
    G_b = nx.Graph()
    G_b.add_nodes_from(G_b_raw.nodes(data=True))
    for u, v in G_b_raw.edges():
        if u != v and not G_b.has_edge(u, v):
            G_b.add_edge(u, v)
            
    # Graph A già semplice dallo script v5, ma assicuriamoci
    G_a = nx.Graph()
    G_a.add_nodes_from(G_a_raw.nodes(data=True))
    for u, v in G_a_raw.edges():
        if u != v and not G_a.has_edge(u, v):
            G_a.add_edge(u, v)
            
    print(f"Graph B loaded: {G_b.number_of_nodes()} nodes, {G_b.number_of_edges()} edges")
    print(f"Graph A loaded: {G_a.number_of_nodes()} nodes, {G_a.number_of_edges()} edges")
    
    return G_b, G_a

# ============================== METRICS ==============================
def compute_required_metrics(G_b):
    """Calcola le metriche richieste per Part 1."""
    print("Computing required graph characteristics...")
    res = {}
    
    res['domain'] = "Windows Sysmon Forensic Evidence Network"
    res['nodes'] = G_b.number_of_nodes()
    res['edges'] = G_b.number_of_edges()
    
    # Clustering medio
    print("  Computing average clustering coefficient...")
    res['avg_clustering'] = round(nx.average_clustering(G_b), 6)
    
    # Assortatività grado
    print("  Computing degree assortativity...")
    res['degree_assortativity'] = round(nx.degree_assortativity_coefficient(G_b), 6)
    
    # Assortatività categoriale
    print("  Computing categorical (type) assortativity...")
    res['categorical_assortativity'] = round(nx.attribute_assortativity_coefficient(G_b, 'type'), 6)
    
    # Diametro e avg path su LCC (stimati)
    print("  Identifying Largest Connected Component (LCC)...")
    lcc = max(nx.connected_components(G_b), key=len)
    subg = G_b.subgraph(lcc).copy()
    n_lcc = subg.number_of_nodes()
    res['lcc_nodes'] = n_lcc
    
    print(f"  Estimating path metrics on LCC ({n_lcc} nodes) via sampling...")
    sample_nodes = random.sample(list(subg.nodes()), min(1000, n_lcc))
    
    lengths = []
    for s in sample_nodes:
        sp = nx.single_source_shortest_path_length(subg, s)
        lengths.extend(sp.values())
    res['avg_shortest_path_estimated'] = round(sum(lengths) / len(lengths), 6)
    
    # Stima diametro
    eccs = []
    for s in sample_nodes:
        sp = nx.single_source_shortest_path_length(subg, s)
        if len(sp) == n_lcc:
            eccs.append(max(sp.values()))
    if eccs:
        res['diameter_estimated'] = int(max(eccs))
        res['radius_estimated'] = int(min(eccs))
    else:
        res['diameter_estimated'] = -1
        res['radius_estimated'] = -1
        
    # Distribuzione gradi per plot
    degs = [d for _, d in G_b.degree()]
    res['degree_histogram'] = dict(Counter(degs))
    
    return res

# ============================== COMMUNITY DETECTION ==============================

def run_louvain(G, seed=SEED):
    """Louvain su NetworkX."""
    print("Running Louvain...")
    start = time.time()
    partition = nx.community.louvain_communities(G, seed=seed)
    runtime = time.time() - start
    
    # Assegna etichette
    labels = {}
    for i, comm in enumerate(partition):
        for n in comm:
            labels[n] = i
            
    Q = nx.community.modularity(G, partition)
    
    return {
        'algorithm': 'Louvain',
        'num_communities': len(partition),
        'modularity': round(Q, 6),
        'runtime_sec': round(runtime, 3),
        'seed': seed,
        'labels': labels
    }

def run_leiden(G, seed=SEED):
    """Leiden su igraph con leidenalg."""
    print("Running Leiden...")
    start = time.time()
    
    # Conversione corretta NetworkX -> igraph
    G_ig = ig.Graph.from_networkx(G)
    
    # Controllo versione leidenalg per resolution_parameter
    import inspect
    sig = inspect.signature(leidenalg.ModularityVertexPartition)
    if 'resolution_parameter' in sig.parameters:
        partition = leidenalg.find_partition(
            G_ig, 
            leidenalg.ModularityVertexPartition, 
            seed=seed,
            resolution_parameter=1.0
        )
    else:
        # Versione recente senza resolution_parameter esplicito nel costruttore
        partition = leidenalg.find_partition(
            G_ig, 
            leidenalg.ModularityVertexPartition, 
            seed=seed
        )
    
    runtime = time.time() - start
    
    # Estrai etichette
    labels = {n: c for n, c in zip(G.nodes(), partition.membership)}
    
    # Modularity calculation
    Q = partition.quality()
    
    return {
        'algorithm': 'Leiden',
        'num_communities': len(set(partition.membership)),
        'modularity': round(Q, 6),
        'runtime_sec': round(runtime, 3),
        'seed': seed,
        'labels': labels
    }

def run_infomap(G):
    """
    Runs Infomap community detection on the given graph.
    Handles disconnected components and isolated nodes correctly.
    Returns a dictionary with results.
    """
    print("Running Infomap...")
    start_time = time.time()
    
    # Infomap in igraph gestisce bene i grafi grandi
    try:
        G_ig = ig.Graph.from_networkx(G)
        # Infomap in igraph: numberOfTrials per stabilità, cluster_counting per etichette
        # Nota: Infomap non ha un parametro 'seed' esplicito come Louvain/Leiden nella CLI standard,
        # ma possiamo fissare il seed globale di igraph se necessario per riproducibilità.
        ig.seed(SEED) 
        infomap_partition = G_ig.community_infomap(trials=10) # trials ~ trial_count
        
        # Estraiamo le comunità come liste di indici
        communities_indices = [list(c) for c in infomap_partition]
        
        # Mappiamo gli indici di igraph agli ID nodo di NetworkX
        node_list = list(G.nodes())
        
        # Creiamo un dizionario node -> community_id
        # Inizializziamo tutti i nodi a -1 (non assegnati) per sicurezza
        node_comm_map = {n: -1 for n in node_list}
        
        for comm_id, comm_indices in enumerate(communities_indices):
            for idx in comm_indices:
                if 0 <= idx < len(node_list):
                    nid = node_list[idx]
                    node_comm_map[nid] = comm_id
        
        # CONTROLLO CRITICO: Assicuriamoci che TUTTI i nodi siano assegnati
        # Se Infomap lascia fuori isolati, assegniamoli a comunità singole o alla più vicina
        # (In realtà Infomap assegna tutto, ma meglio prevenire)
        unassigned = [n for n, c in node_comm_map.items() if c == -1]
        if unassigned:
            print(f"  Warning: {len(unassigned)} nodes unassigned by Infomap. Assigning to singletons.")
            next_cid = max(node_comm_map.values()) + 1 if node_comm_map else 0
            for n in unassigned:
                node_comm_map[n] = next_cid
                next_cid += 1
        
        num_communities = len(set(node_comm_map.values()))
        
        # Calcolo Modularity post-hoc
        # Dobbiamo passare a networkx una lista di liste di nodi CHE COPRE TUTTO IL GRAFO
        final_communities_lists = [[] for _ in range(num_communities)]
        for n, cid in node_comm_map.items():
            final_communities_lists[cid].append(n)
            
        Q = nx.community.modularity(G, final_communities_lists)
        
        runtime = time.time() - start_time
        
        print(f"  Communities: {num_communities}, Modularity: {Q:.6f}, Time: {runtime:.2f}s")
        
        return {
            "algorithm": "Infomap",
            "num_communities": num_communities,
            "modularity": round(Q, 6),
            "runtime_seconds": round(runtime, 3),
            "labels": node_comm_map, # Dict {node: comm_id}
            "community_sizes": dict(Counter(node_comm_map.values()))
        }

    except Exception as e:
        print(f"  Infomap failed: {e}")
        # Fallback vuoto per non bloccare tutto il resto
        return {
            "algorithm": "Infomap",
            "error": str(e),
            "num_communities": 0,
            "modularity": None,
            "runtime_seconds": 0,
            "labels": {},
            "community_sizes": {}
        }

# ============================== COMPARISON ==============================

def compare_partitions(lab1, lab2):
    """Calcola NMI e ARI tra due dizionari di etichette."""
    nodes = sorted(list(set(lab1.keys()) & set(lab2.keys())))
    y1 = [lab1[n] for n in nodes]
    y2 = [lab2[n] for n in nodes]
    
    nmi = normalized_mutual_info_score(y1, y2)
    ari = adjusted_rand_score(y1, y2)
    
    return {'nmi': round(nmi, 6), 'ari': round(ari, 6)}

def run_stability(G, algorithm, n_seeds=STABILITY_SEEDS):
    """Analisi di stabilità per un algoritmo su diversi seed."""
    print(f"Running stability analysis for {algorithm} ({n_seeds} seeds)...")
    
    results = []
    canonical = None
    
    for i in range(n_seeds):
        seed = SEED + i
        if algorithm == 'Louvain':
            res = run_louvain(G, seed=seed)
        elif algorithm == 'Leiden':
            res = run_leiden(G, seed=seed)
        else:
            continue
            
        if canonical is None:
            canonical = res['labels']
            
        results.append(res)
        
    # Confronta tutti vs primo (canonical)
    stability = []
    for res in results[1:]:
        comp = compare_partitions(canonical, res['labels'])
        stability.append(comp['nmi'])
        
    return {
        'algorithm': algorithm,
        'n_seeds': n_seeds,
        'min_nmi_vs_canonical': round(min(stability) if stability else 1.0, 6),
        'mean_nmi_vs_canonical': round(np.mean(stability) if stability else 1.0, 6),
        'community_count_range': [r['num_communities'] for r in results]
    }

# ============================== TYPE PROFILES ==============================

def community_type_profiles(G, labels):
    """Calcola il profilo di composizione per tipo per ogni comunità."""
    comms = {}
    for n, c in labels.items():
        comms.setdefault(c, []).append(n)
        
    profiles = []
    for cid, nodes in sorted(comms.items()):
        types = [G.nodes[n].get('type', 'unknown') for n in nodes]
        counts = Counter(types)
        total = len(types)
        profile = {t: round(c/total, 4) for t, c in counts.items()}
        profiles.append({
            'community_id': cid,
            'size': total,
            'type_fractions': profile
        })
        
    return sorted(profiles, key=lambda x: x['size'], reverse=True)

# ============================== GRAPH B vs A COMPARISON ==============================

def compare_b_vs_a(G_b, G_a, labels_b, labels_a):
    """Confronta le comunità di B e A sui nodi processo condivisi."""
    # Trova nodi processo condivisi
    nodes_b = {n for n in G_b.nodes() if G_b.nodes[n].get('type') == 'proc'}
    nodes_a = {n for n in G_a.nodes() if G_a.nodes[n].get('type') == 'proc'}
    
    shared = sorted(nodes_b & nodes_a)
    
    if not shared:
        return {'error': 'Nessun nodo processo condiviso'}
        
    lab_b_sub = {n: labels_b[n] for n in shared}
    lab_a_sub = {n: labels_a[n] for n in shared}
    
    comp = compare_partitions(lab_b_sub, lab_a_sub)
    
    return {
        'shared_proc_nodes': len(shared),
        'nmi': comp['nmi'],
        'ari': comp['ari']
    }

# ============================== PLOTTING ==============================

def plot_results(results, G_b, output_dir):
    """Genera tutte le figure dai risultati."""
    print(f"Generating plots in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Figura 1: Degree Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    if 'metrics' in results and 'degree_histogram' in results['metrics']:
        hist = results['metrics']['degree_histogram']
        ks = sorted([int(k) for k in hist.keys()])
        vals = [hist[str(k)] if str(k) in hist else hist.get(k, 0) for k in ks]
        ax.loglog(ks, vals, marker='.', linestyle='None')
        ax.set_xlabel("Degree (log)")
        ax.set_ylabel("Frequency (log)")
        ax.set_title("Degree Distribution Graph B")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "01_degree_dist.png"), dpi=300)
        plt.close()
    
    # Figura 2: Algorithm Agreement (NMI/ARI Heatmap)
    fig, ax = plt.subplots(figsize=(8, 6))
    algos = ['Louvain', 'Leiden', 'Infomap']
    pairs = [('Louvain', 'Leiden'), ('Louvain', 'Infomap'), ('Leiden', 'Infomap')]
    
    if 'comparison' in results:
        data = []
        for a, b in pairs:
            key = f"{a}_vs_{b}"
            if key in results['comparison']:
                data.append([results['comparison'][key]['nmi'], results['comparison'][key]['ari']])
        
        if data:
            im = ax.imshow(np.array(data), cmap='YlOrRd', aspect='auto')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['NMI', 'ARI'])
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels([f"{a} vs {b}" for a, b in pairs])
            plt.colorbar(im, label='Score')
            plt.title("Algorithm Agreement (NMI/ARI)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "02_agreement.png"), dpi=300)
            plt.close()

    # Figura 3: Type Composition Top Communities
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'type_profiles' in results and len(results['type_profiles']) > 0:
        top5 = results['type_profiles'][:5]
        labels_c = [f"C{i} (n={p['size']})" for i, p in enumerate(top5)]
        
        types = sorted(set(t for p in top5 for t in p['type_fractions'].keys()))
        matrix = []
        for p in top5:
            row = [p['type_fractions'].get(t, 0) for t in types]
            matrix.append(row)
            
        bottom = np.zeros(len(top5))
        colors = sns.color_palette("pastel", len(types))
        
        for i, t in enumerate(types):
            vals = [matrix[j][i] for j in range(len(top5))]
            ax.bar(labels_c, vals, bottom=bottom, label=t, color=colors[i])
            bottom += np.array(vals)
            
        ax.set_ylabel("Fraction")
        ax.set_title("Type Composition of Top 5 Communities (Graph B)")
        ax.legend(title="Node Type")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_type_composition.png"), dpi=300)
        plt.close()

    # Figura 4: B vs A Comparison
    fig, ax = plt.subplots(figsize=(6, 4))
    if 'b_vs_a' in results:
        nmi = results['b_vs_a'].get('nmi', 0)
        ari = results['b_vs_a'].get('ari', 0)
        ax.bar(['NMI', 'ARI'], [nmi, ari], color=['navy', 'coral'])
        ax.set_ylabel("Score")
        ax.set_title("Graph B vs Graph A Community Agreement\n(Shared Process Nodes Only)")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "04_b_vs_a.png"), dpi=300)
        plt.close()
        
    print("All plots generated.")

# ============================== MAIN ==============================

def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Assignment 1 Community Analysis")
    parser.add_argument('--plot-only', action='store_true', help="Rigenera solo plot da results.json")
    args = parser.parse_args()
    
    # Caricamento grafi sempre necessario anche per plot-only (per type info)
    G_b, G_a = load_graphs()
    
    if args.plot_only:
        if not os.path.exists(RESULTS_FILE):
            print(f"Errore: {RESULTS_FILE} non trovato.")
            sys.exit(1)
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        plot_results(results, G_b, FIGURES_DIR)
        return

    print("--- Starting Full Analysis ---")
    random.seed(SEED)
    np.random.seed(SEED)
    
    results = {}
    
    # 1. Metrics
    results['metrics'] = compute_required_metrics(G_b)
    
    # 2. Community Detection
    louv_res = run_louvain(G_b, seed=SEED)
    leid_res = run_leiden(G_b, seed=SEED)
    inf_res = run_infomap(G_b)
    
    results['louvain'] = louv_res
    results['leiden'] = leid_res
    results['infomap'] = inf_res
    
    # 3. Pairwise Comparison
    results['comparison'] = {
        'Louvain_vs_Leiden': compare_partitions(louv_res['labels'], leid_res['labels']),
        'Louvain_vs_Infomap': compare_partitions(louv_res['labels'], inf_res['labels']),
        'Leiden_vs_Infomap': compare_partitions(leid_res['labels'], inf_res['labels'])
    }
    
    # 4. Stability
    results['stability_louvain'] = run_stability(G_b, 'Louvain')
    results['stability_leiden'] = run_stability(G_b, 'Leiden')
    
    # 5. Type Profiles (Louvain come riferimento)
    results['type_profiles'] = community_type_profiles(G_b, louv_res['labels'])
    
    # 6. Graph B vs A
    # Eseguiamo Louvain su A per confronto
    print("Running Louvain su Graph A...")
    louv_a = run_louvain(G_a, seed=SEED)
    results['b_vs_a'] = compare_b_vs_a(G_b, G_a, louv_res['labels'], louv_a['labels'])
    results['graph_a_communities'] = louv_a['num_communities']
    
    # Save
    save_results(results, RESULTS_FILE)
    
    # Plots
    plot_results(results, G_b, FIGURES_DIR)
    
    print("\n=== Analysis Complete ===")
    print(f"Louvain: {louv_res['num_communities']} communities, Q={louv_res['modularity']}")
    print(f"Leiden: {leid_res['num_communities']} communities, Q={leid_res['modularity']}")
    print(f"Infomap: {inf_res['num_communities']} communities")
    print(f"B vs A (Proc): NMI={results['b_vs_a']['nmi']}, ARI={results['b_vs_a']['ari']}")

if __name__ == "__main__":
    main()