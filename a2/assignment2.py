#!/usr/bin/env python3
"""
Assignment 2: Network Robustness and Recovery Analysis
Student: Muce + Virgi
Course: Network Analysis
Description: Robustness analysis of Graph A (process actor layer) under targeted/random attacks,
             cross-layer context impact using Graph B, and recovery strategies.
"""

import os
import sys
import json
import random
import argparse
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# ============================== CONFIGURATION ==============================
GRAPH_A_PATH = "../output/v5_graphA.gexf"
GRAPH_B_PATH = "../output/v5_graphB.gexf"
RESULTS_FILE = "results.json"
FIGURES_DIR = "figures"

SEED = 42
RANDOM_ATTACK_REPETITIONS = 50
RECOVERY_BUDGETS = [5, 10, 20, 40, 80]

sns.set_context("paper", font_scale=1.2)
plt.style.use('seaborn-v0_8-whitegrid')

# ============================== DATA LOADING ==============================
def load_graphs():
    """Carica Graph A e Graph B dai file GEXF."""
    print("Loading graphs...")
    G_A = nx.read_gexf(GRAPH_A_PATH)
    G_B = nx.read_gexf(GRAPH_B_PATH)

    # Pulizia attributi numerici
    for u, v, d in G_A.edges(data=True):
        if 'weight' in d:
            try:
                d['weight'] = int(d['weight'])
            except:
                pass
        if 'edge_type' in d and 'type' not in d:
            d['type'] = d['edge_type']

    for u, v, d in G_B.edges(data=True):
        if 'weight' in d:
            try:
                d['weight'] = int(d['weight'])
            except:
                pass
        if 'edge_type' in d and 'type' not in d:
            d['type'] = d['edge_type']

    for n, d in G_A.nodes(data=True):
        if 'type' not in d:
            d['type'] = 'PROC'

    print(f"Graph A loaded: {G_A.number_of_nodes()} nodes, {G_A.number_of_edges()} edges")
    print(f"Graph B loaded: {G_B.number_of_nodes()} nodes, {G_B.number_of_edges()} edges")

    return G_A, G_B

def prepare_actor_graph(G_A):
    """Prepara il grafo attore: LCC di Graph A, semplice, non diretto, senza loop."""
    # Converti in non diretto se necessario
    if G_A.is_directed():
        G_undir = nx.Graph(G_A)
    else:
        G_undir = G_A.copy()

    # Rimuovi self-loop
    G_undir.remove_edges_from(list(nx.selfloop_edges(G_undir)))

    # Trova LCC
    lcc = max(nx.connected_components(G_undir), key=len)
    G_lcc = G_undir.subgraph(lcc).copy()

    print(f"Actor graph (LCC): {G_lcc.number_of_nodes()} nodes, {G_lcc.number_of_edges()} edges")
    return G_lcc

# ============================== ATTACK FUNCTIONS ==============================
def run_random_attack(G, num_repetitions=RANDOM_ATTACK_REPETITIONS, seed=SEED):
    """Esegue attacchi casuali multipli e restituisce le curve medie."""
    print(f"Running Random Attack ({num_repetitions} repetitions)...")
    random.seed(seed)
    np.random.seed(seed)

    n = G.number_of_nodes()
    all_curves = []

    for rep in range(num_repetitions):
        G_copy = G.copy()
        nodes = list(G_copy.nodes())
        random.shuffle(nodes)

        curve = {'removed': [], 'detached': 0, 'active': n, 'fraction': 1.0}
        active_set = set(nodes)

        for i, target in enumerate(nodes):
            if target not in G_copy:
                continue

            G_copy.remove_node(target)
            active_set.discard(target)

            comps = list(nx.connected_components(G_copy))
            if not comps:
                break

            largest = max(comps, key=len)
            detached_this_step = sum(len(c) for c in comps if c != largest)

            curve['removed'].append(i + 1)
            curve['detached'] = curve.get('detached', 0) + detached_this_step
            curve['active'] = len(largest)
            curve['fraction'] = len(largest) / n

            G_copy = G_copy.subgraph(largest).copy()

            if curve['active'] == 0:
                break

        all_curves.append(curve)

    # Calcola media e deviazione standard
    max_steps = max(len(c['removed']) for c in all_curves)
    avg_curve = {'removed': list(range(1, max_steps + 1)), 'mean_fraction': [], 'std_fraction': []}

    for step in range(1, max_steps + 1):
        frags = [c['fraction'] for c in all_curves if step <= len(c['removed'])]
        if frags:
            avg_curve['mean_fraction'].append(round(np.mean(frags), 6))
            avg_curve['std_fraction'].append(round(np.std(frags), 6))
        else:
            avg_curve['mean_fraction'].append(0)
            avg_curve['std_fraction'].append(0)

    return avg_curve

def run_adaptive_attack(G, strategy='degree', seed=SEED):
    """Esegue un attacco adattativo basato sulla strategia specificata."""
    print(f"Running Adaptive Attack ({strategy})...")
    random.seed(seed)
    np.random.seed(seed)

    G_copy = G.copy()
    n = G.number_of_nodes()
    active_set = set(G_copy.nodes())

    results = {
        'strategy': strategy,
        'sequence': [],
        'removed': [],
        'detached_list': [],
        'active_list': [],
        'fraction_list': [],
        'centrality_snapshot': {}
    }

    step = 0
    while len(active_set) > 0:
        step += 1

        # Calcola centralità corrente
        if strategy == 'degree':
            centrality = nx.degree_centrality(G_copy)
        elif strategy == 'pagerank':
            try:
                centrality = nx.pagerank(G_copy, alpha=0.85, max_iter=100)
            except:
                centrality = nx.degree_centrality(G_copy)
        elif strategy == 'betweenness':
            if len(G_copy) < 500:
                centrality = nx.betweenness_centrality(G_copy)
            else:
                # Campionamento per grafi grandi
                centrality = nx.betweenness_centrality(G_copy, k=min(100, len(G_copy)))
        else:
            centrality = nx.degree_centrality(G_copy)

        if not centrality:
            break

        # Seleziona nodo con massima centralità
        target = max(centrality, key=centrality.get)
        results['sequence'].append(str(target))
        results['centrality_snapshot'][step] = {str(target): round(centrality[target], 6)}

        # Rimuovi target
        G_copy.remove_node(target)
        active_set.discard(target)

        # Calcola componenti
        comps = list(nx.connected_components(G_copy))
        if not comps:
            results['removed'].append(step)
            results['detached_list'].append(results.get('total_detached', 0))
            results['active_list'].append(0)
            results['fraction_list'].append(0.0)
            break

        largest = max(comps, key=len)
        detached_count = sum(len(c) for c in comps if c != largest)
        total_detached = results.get('total_detached', 0) + detached_count
        results['total_detached'] = total_detached

        results['removed'].append(step)
        results['detached_list'].append(total_detached)
        results['active_list'].append(len(largest))
        results['fraction_list'].append(round(len(largest) / n, 6))

        # Mantieni solo LCC
        G_copy = G_copy.subgraph(largest).copy()
        active_set = set(G_copy.nodes())

        if len(active_set) == 0:
            break

    return results

def compute_f50_auc(fractions):
    """Calcola f50 (frazione di rimozioni per raggiungere il 50% dei nodi attivi) e AUC."""
    f50 = None
    for i, f in enumerate(fractions):
        if f <= 0.5:
            f50 = (i + 1)
            break

    if f50 is None:
        f50 = len(fractions)

    # AUC approssimato con trapezi
    auc = np.trapz(fractions, dx=1.0/len(fractions)) if len(fractions) > 1 else fractions[0] if fractions else 0

    return round(f50, 6), round(float(auc), 6)

# ============================== CONTEXT RETENTION ==============================
def compute_context_retention(G_A_lcc, G_B, attack_sequence):
    """Calcola la ritenzione del contesto per ogni tipo di nodo dopo ogni rimozione."""
    print("Computing context retention...")

    # Mappa i nodi di Graph A a Graph B
    proc_nodes_A = set(G_A_lcc.nodes())

    # Trova nodi PROC equivalenti in Graph B
    proc_in_B = set()
    for n in G_B.nodes():
        if G_B.nodes[n].get('type') == 'PROC':
            # Estrai path dal nodo
            if '::' in str(n):
                path = str(n).split('::')[1]
                if any(path.endswith(p) for p in ['cmd.exe', 'PowerShell', 'explorer.exe', 'services.exe', 'svchost.exe']):
                    proc_in_B.add(n)

    # Inizializza contesto iniziale
    initial_context = {'FILE': set(), 'REG': set(), 'DLL': set(), 'IP': set()}

    for n, data in G_B.nodes(data=True):
        node_type = data.get('type', '')
        if node_type in initial_context:
            # Controlla se collegato a un processo attivo iniziale
            neighbors = list(G_B.neighbors(n))
            if any(nb in proc_in_B or any(str(p) in str(nb) for p in proc_nodes_A) for nb in neighbors):
                initial_context[node_type].add(n)

    results = {
        'initial_counts': {k: len(v) for k, v in initial_context.items()},
        'retention_over_steps': []
    }

    active_procs = set(proc_nodes_A)

    for step, removed in enumerate(attack_sequence[:5]):  # Primi 5 passi
        # Rimuovi processo rimosso
        if removed in active_procs:
            active_procs.discard(removed)

        # Calcola contesto perso
        retained = {k: set() for k in initial_context.keys()}

        for n, data in G_B.nodes(data=True):
            node_type = data.get('type', '')
            if node_type not in initial_context:
                continue
            if n not in initial_context[node_type]:
                continue

            # Controlla se ancora collegato a un processo attivo
            neighbors = list(G_B.neighbors(n))
            has_active_neighbor = False

            for nb in neighbors:
                nb_type = G_B.nodes[nb].get('type', '')
                if nb_type == 'PROC':
                    # Verifica se questo processo è ancora attivo
                    if any(str(p) in str(nb) or str(nb) in str(p) for p in active_procs):
                        has_active_neighbor = True
                        break

            if has_active_neighbor:
                retained[node_type].add(n)

        retention = {k: len(retained[k]) / max(1, len(initial_context[k])) for k in initial_context.keys()}
        results['retention_over_steps'].append({
            'step': step + 1,
            'removed': removed,
            'retention': retention
        })

    return results

# ============================== RECOVERY STRATEGIES ==============================
def build_recovery_candidates(G_A, G_B):
    """Genera candidati per il recupero basati sul contesto."""
    print("Building recovery candidates...")

    candidates = []

    # Semplificazione: cerca processi ad alto grado che potrebbero fungere da backup
    degrees = dict(G_A.degree())
    sorted_procs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    for i, (p1, d1) in enumerate(sorted_procs[:20]):
        for j, (p2, d2) in enumerate(sorted_procs[i+1:20]):
            if not G_A.has_edge(p1, p2):
                candidates.append((p1, p2, d1 + d2))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:50]

def add_recovery_edges(G, candidates, budget):
    """Aggiunge archi di recupero al grafo."""
    G_rec = G.copy()
    added = 0

    for c1, c2, _ in candidates:
        if added >= budget:
            break
        if c1 != c2 and not G_rec.has_edge(c1, c2):
            G_rec.add_edge(c1, c2, type='recovery', weight=1)
            added += 1

    return G_rec, added

def run_recovery_experiments(G_original, candidates, budgets=RECOVERY_BUDGETS):
    """Esegue esperimenti di recupero con diversi budget."""
    print("Running recovery experiments...")
    results = {}

    for budget in budgets:
        G_rec, added = add_recovery_edges(G_original, candidates, budget)

        # Esegui attacco adattativo sul grafo recuperato
        attack_res = run_adaptive_attack(G_rec, strategy='degree')
        f50, auc = compute_f50_auc(attack_res['fraction_list'])

        results[budget] = {
            'edges_added': added,
            'f50': f50,
            'auc': auc,
            'final_fraction': attack_res['fraction_list'][-1] if attack_res['fraction_list'] else 0
        }

    return results

# ============================== PLOTTING ==============================
def plot_results(results, output_dir):
    """Genera tutti i grafici dai risultati salvati."""
    print(f"Generating plots in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Figura 1: Curve di robustezza
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'random_attack' in results:
        ra = results['random_attack']
        steps = ra['removed']
        ax.plot(steps, ra['mean_fraction'], label='Random', color='gray', linestyle='--')
        ax.fill_between(steps,
                       np.array(ra['mean_fraction']) - np.array(ra['std_fraction']),
                       np.array(ra['mean_fraction']) + np.array(ra['std_fraction']),
                       alpha=0.3, color='gray')

    strategies = ['degree', 'pagerank', 'betweenness']
    colors = ['red', 'blue', 'green']

    for strat, color in zip(strategies, colors):
        key = f'adaptive_{strat}'
        if key in results:
            ar = results[key]
            ax.plot(ar['removed'], ar['fraction_list'], label=f'Adaptive {strat.capitalize()}', color=color)

    ax.set_xlabel('Fraction of Nodes Removed')
    ax.set_ylabel('Fraction of Nodes in Largest Component')
    ax.set_title('Robustness Curves: Random vs Targeted Attacks')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_robustness_curves.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '01_robustness_curves.pdf'))
    plt.close()
    print("  Saved 01_robustness_curves")

    # Figura 2: Primi 5 attacchi
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'context_retention' in results:
        cr = results['context_retention']
        steps = [r['step'] for r in cr['retention_over_steps']]

        for ctype in ['FILE', 'REG', 'DLL', 'IP']:
            retentions = [r['retention'].get(ctype, 0) for r in cr['retention_over_steps']]
            ax.plot(steps, retentions, marker='o', label=ctype)

    ax.set_xlabel('Attack Step')
    ax.set_ylabel('Context Retention Fraction')
    ax.set_title('Context Retention After First 5 Targeted Removals')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_context_retention.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '02_context_retention.pdf'))
    plt.close()
    print("  Saved 02_context_retention")

    # Figura 3: Recupero
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'recovery_experiments' in results:
        rec = results['recovery_experiments']
        budgets = list(rec.keys())
        f50_vals = [rec[b]['f50'] for b in budgets]
        auc_vals = [rec[b]['auc'] for b in budgets]

        x = np.arange(len(budgets))
        width = 0.35

        bars1 = ax.bar(x - width/2, f50_vals, width, label='f50', color='steelblue')
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, auc_vals, width, label='AUC', color='coral', alpha=0.7)

        ax.set_xlabel('Recovery Budget (Edges Added)')
        ax.set_ylabel('f50', color='steelblue')
        ax2.set_ylabel('AUC', color='coral')
        ax.set_title('Recovery Strategy Performance vs Budget')
        ax.set_xticks(x)
        ax.set_xticklabels(budgets)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_recovery_performance.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '03_recovery_performance.pdf'))
    plt.close()
    print("  Saved 03_recovery_performance")

    # Figura 4: Comparazione strategie
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Random AUC', 'Degree AUC', 'PageRank AUC', 'Betweenness AUC']
    values = []

    if 'random_attack' in results:
        ra = results['random_attack']
        random_auc = np.trapz(ra['mean_fraction'], dx=1.0/max(1, len(ra['mean_fraction'])))
        values.append(round(random_auc, 4))

    for strat in ['degree', 'pagerank', 'betweenness']:
        key = f'adaptive_{strat}'
        if key in results:
            ar = results[key]
            auc = np.trapz(ar['fraction_list'], dx=1.0/max(1, len(ar['fraction_list'])))
            values.append(round(auc, 4))

    ax.bar(metrics, values, color=['gray', 'red', 'blue', 'green'])
    ax.set_ylabel('Area Under Curve (AUC)')
    ax.set_title('Comparison of Attack Strategies')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_strategy_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, '04_strategy_comparison.pdf'))
    plt.close()
    print("  Saved 04_strategy_comparison")

    print("All plots generated successfully.")

# ============================== MAIN ==============================
def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Assignment 2 Robustness Analysis")
    parser.add_argument('--plot-only', action='store_true', help="Rigenera solo i grafici")
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(RESULTS_FILE):
            print(f"Errore: {RESULTS_FILE} non trovato.")
            sys.exit(1)
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        G_A, _ = load_graphs()
        plot_results(results, FIGURES_DIR)
        return

    print("--- Starting Full Analysis ---")
    random.seed(SEED)
    np.random.seed(SEED)

    # Carica grafi
    G_A, G_B = load_graphs()

    # Prepara grafo attore
    G_actor = prepare_actor_graph(G_A)

    results = {}

    # Attacchi
    results['random_attack'] = run_random_attack(G_actor)
    results['adaptive_degree'] = run_adaptive_attack(G_actor, strategy='degree')
    results['adaptive_pagerank'] = run_adaptive_attack(G_actor, strategy='pagerank')
    results['adaptive_betweenness'] = run_adaptive_attack(G_actor, strategy='betweenness')

    # Contesto
    results['context_retention'] = compute_context_retention(G_actor, G_B, results['adaptive_degree']['sequence'])

    # Recupero
    candidates = build_recovery_candidates(G_actor, G_B)
    results['recovery_experiments'] = run_recovery_experiments(G_actor, candidates)

    # Salva
    save_results(results, RESULTS_FILE)

    # Plot
    plot_results(results, FIGURES_DIR)

    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
