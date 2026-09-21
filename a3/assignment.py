#!/usr/bin/env python3
"""
Assignment 3: Process Compromise Cascade (Social Contagion Adaptation)
Student: Muce + Virgi
Course: Network Analysis
Description: Synthetic contagion experiment on Graph A (Process Layer) with Graph B context analysis.
Optimized for Python 3.10+ and NetworkX 3.x
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

SEED_BASE = 42
MAX_ITERATIONS = 100

# Stati
STATE_TRUSTED = "TRUSTED"
STATE_ALTERED = "ALTERED"
STATE_COMPROMISED = "COMPROMISED"
STATE_SUSCEPTIBLE = "SUSCEPTIBLE"

# Ruoli
ROLE_VERIFIER = "VERIFIER"
ROLE_UNINTENTIONAL = "UNINTENTIONAL"
ROLE_BOT = "BOT"
ROLE_ORDINARY = "ORDINARY"

ANCHOR_CONFIG = {"V": 22, "U": 22, "B": 2, "theta": 0.40}

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.1)

# ============================== DATA LOADING ==============================

def load_graphs():
    if not os.path.exists(GRAPH_A_PATH) or not os.path.exists(GRAPH_B_PATH):
        print(f"Errore: Grafi non trovati.")
        sys.exit(1)

    print("Loading graphs...")
    
    G_a_raw = nx.read_gexf(GRAPH_A_PATH)
    G_a = nx.Graph()
    G_a.add_nodes_from(G_a_raw.nodes(data=True))
    for u, v in G_a_raw.edges():
        if u != v:
            G_a.add_edge(u, v)
    
    if not nx.is_connected(G_a):
        lcc = max(nx.connected_components(G_a), key=len)
        G_a = G_a.subgraph(lcc).copy()
        print(f"  Extracted LCC for Graph A: {G_a.number_of_nodes()} nodes")
    else:
        print(f"  Graph A is connected: {G_a.number_of_nodes()} nodes")

    G_b = nx.read_gexf(GRAPH_B_PATH)
    for u, v, d in G_b.edges(data=True):
        if 'edge_type' in d and 'type' not in d:
            d['type'] = d['edge_type']
        if 'weight' in d:
            try:
                d['weight'] = int(d['weight'])
            except:
                d['weight'] = 1
    
    print(f"Graph A loaded: {G_a.number_of_nodes()} nodes, {G_a.number_of_edges()} edges (LCC)")
    print(f"Graph B loaded: {G_b.number_of_nodes()} nodes, {G_b.number_of_edges()} edges")
    
    return G_a, G_b

# ============================== CASCADE MODEL ==============================

def assign_roles(nodes, n_verifiers, n_unintentional, n_bots, seed=None, placement='random', cent_map=None):
    if seed is not None:
        random.seed(seed)
    
    if placement == 'random':
        shuffled = nodes[:]
        random.shuffle(shuffled)
        ordered_nodes = shuffled
    else:
        # Ordinamento per centralità decrescente
        if cent_map is None:
            ordered_nodes = nodes[:] # Fallback a random se manca mappa
            random.shuffle(ordered_nodes)
        else:
            ordered_nodes = sorted(nodes, key=lambda k: cent_map.get(k, 0), reverse=True)
    
    roles = {n: ROLE_ORDINARY for n in nodes}
    idx = 0
    
    # Priorità: Bot -> Unintentional -> Verifier
    for _ in range(n_bots):
        if idx < len(ordered_nodes):
            roles[ordered_nodes[idx]] = ROLE_BOT
            idx += 1
    for _ in range(n_unintentional):
        if idx < len(ordered_nodes):
            roles[ordered_nodes[idx]] = ROLE_UNINTENTIONAL
            idx += 1
    for _ in range(n_verifiers):
        if idx < len(ordered_nodes):
            roles[ordered_nodes[idx]] = ROLE_VERIFIER
            idx += 1
            
    return roles

def initialize_states(roles):
    states = {}
    mutated_flags = {}
    for node, role in roles.items():
        if role == ROLE_VERIFIER:
            states[node] = STATE_TRUSTED
        elif role == ROLE_BOT:
            states[node] = STATE_COMPROMISED
        elif role == ROLE_UNINTENTIONAL:
            states[node] = STATE_TRUSTED
            mutated_flags[node] = False
        else:
            states[node] = STATE_SUSCEPTIBLE
            mutated_flags[node] = False
    return states, mutated_flags

def run_cascade_step(G, roles, states, mutated_flags, theta, neighbors_map):
    new_states = states.copy()
    changes_made = False
    
    # Lista nodi suscettibili
    susceptible_nodes = [n for n, s in states.items() if s == STATE_SUSCEPTIBLE]
    
    for node in susceptible_nodes:
        neighs = neighbors_map[node]
        if not neighs:
            continue
            
        active_neighs = [n for n in neighs if states[n] != STATE_SUSCEPTIBLE]
        if not active_neighs:
            continue
            
        frac_active = len(active_neighs) / len(neighs)
        
        if frac_active < theta:
            continue
        
        # Calcolo pluralità
        neighbor_states = [states[n] for n in active_neighs]
        counts = Counter(neighbor_states)
        max_count = max(counts.values())
        top_states = [s for s, c in counts.items() if c == max_count]
        
        if len(top_states) > 1:
            continue # Tie
        
        adopted_state = top_states[0]
        role = roles[node]
        
        if role == ROLE_UNINTENTIONAL:
            new_states[node] = STATE_ALTERED
            mutated_flags[node] = True
            changes_made = True
        elif role == ROLE_BOT:
            new_states[node] = STATE_COMPROMISED
            changes_made = True
        else:
            new_states[node] = adopted_state
            changes_made = True
            
    return new_states, changes_made

def run_cascade(G, roles, states, mutated_flags, theta, neighbors_map):
    iteration = 0
    converged = False
    
    while not converged and iteration < MAX_ITERATIONS:
        iteration += 1
        states, changes_made = run_cascade_step(G, roles, states, mutated_flags, theta, neighbors_map)
        if not changes_made:
            converged = True
            
    return states, iteration

def run_repetition_optimized(nodes, roles_init, theta, neighbors_map, max_iter=MAX_ITERATIONS):
    """Versione ultra-ottimizzata per loop massivi."""
    # Copia rapida
    states = roles_init['states'].copy()
    roles = roles_init['roles']
    mutated = roles_init['mutated'].copy()
    
    iteration = 0
    changed = True
    
    # Pre-calcolo suscettibili iniziali per velocizzare
    # Ma poiché lo stato cambia, dobbiamo ricontrollare o mantenere una lista dinamica
    # Per semplicità e corretteza, iteriamo su tutti i nodi controllando lo stato
    
    while changed and iteration < max_iter:
        iteration += 1
        changed = False
        new_states = states.copy()
        
        for node in nodes:
            if states[node] != STATE_SUSCEPTIBLE:
                continue
            
            neighs = neighbors_map[node]
            if not neighs:
                continue
                
            # Conteggio attivi
            active_count = 0
            state_counts = Counter()
            
            for n in neighs:
                s = states[n]
                if s != STATE_SUSCEPTIBLE:
                    active_count += 1
                    state_counts[s] += 1
            
            if active_count == 0:
                continue
                
            if active_count / len(neighs) < theta:
                continue
                
            max_c = max(state_counts.values())
            tops = [s for s, c in state_counts.items() if c == max_c]
            
            if len(tops) > 1:
                continue
                
            adopted = tops[0]
            role = roles[node]
            
            if role == ROLE_UNINTENTIONAL:
                new_states[node] = STATE_ALTERED
                mutated[node] = True
                changed = True
            elif role == ROLE_BOT:
                new_states[node] = STATE_COMPROMISED
                changed = True
            else:
                new_states[node] = adopted
                changed = True
        
        states = new_states
        
    counts = Counter(states.values())
    total = len(nodes)
    
    return {
        "compromised_count": counts.get(STATE_COMPROMISED, 0),
        "altered_count": counts.get(STATE_ALTERED, 0),
        "trusted_count": counts.get(STATE_TRUSTED, 0),
        "susceptible_count": counts.get(STATE_SUSCEPTIBLE, 0),
        "iterations": iteration,
        "majority_compromised": counts.get(STATE_COMPROMISED, 0) > total / 2,
        "majority_altered": counts.get(STATE_ALTERED, 0) > total / 2,
        "final_states": states # Necessario per footprint
    }

# ============================== EXPERIMENTS ==============================

def run_convergence_study(G, config, centralities, neighbors_map, max_seeds=200):
    print("Running convergence study (optimized)...")
    steps = [25, 50, 100, 200]
    steps = [s for s in steps if s <= max_seeds]
    
    nodes = list(G.nodes())
    total_nodes = len(nodes)
    
    configs_to_test = ['random', 'degree', 'betweenness']
    results = {k: {} for k in configs_to_test}
    
    # Cache per ruoli generati (per non ricalcarli se non serve, ma qui li rigeneriamo per seed)
    
    for p_name in configs_to_test:
        cent_map = None
        if p_name == 'degree':
            cent_map = centralities['deg']
        elif p_name == 'betweenness':
            cent_map = centralities['bet']
            
        cumulative_comp = []
        cumulative_maj = []
        
        # Eseguiamo tutte le run sequenzialmente
        current_states_acc = []
        current_maj_acc = []
        
        for i in range(max_seeds):
            seed = SEED_BASE + i
            roles_dict = assign_roles(nodes, config['V'], config['U'], config['B'], seed=seed, placement=p_name, cent_map=cent_map)
            states_dict, mutated_dict = initialize_states(roles_dict)
            
            roles_init = {'roles': roles_dict, 'states': states_dict, 'mutated': mutated_dict}
            
            res = run_repetition_optimized(nodes, roles_init, config['theta'], neighbors_map)
            
            cum_frac = res['compromised_count'] / total_nodes
            maj_flag = 1.0 if res['majority_compromised'] else 0.0
            
            cumulative_comp.append(cum_frac)
            cumulative_maj.append(maj_flag)
            
            n_curr = i + 1
            if n_curr in steps:
                mean_c = np.mean(cumulative_comp)
                freq_m = np.mean(cumulative_maj)
                results[p_name][n_curr] = {
                    "mean_compromised_fraction": round(float(mean_c), 4),
                    "majority_frequency": round(float(freq_m), 4),
                    "runs": n_curr
                }
                
    return results

def run_theta_sweep(G, config, centralities, neighbors_map, thetas, n_runs, placement='random'):
    print(f"Running Theta Sweep ({placement})...")
    nodes = list(G.nodes())
    total = len(nodes)
    results = []
    
    cent_map = None
    if placement == 'degree': cent_map = centralities['deg']
    elif placement == 'betweenness': cent_map = centralities['bet']
    
    for theta in thetas:
        maj_count = 0
        comp_fracs = []
        
        for i in range(n_runs):
            seed = SEED_BASE + i
            roles_d = assign_roles(nodes, config['V'], config['U'], config['B'], seed=seed, placement=placement, cent_map=cent_map)
            st_d, mut_d = initialize_states(roles_d)
            init = {'roles': roles_d, 'states': st_d, 'mutated': mut_d}
            
            res = run_repetition_optimized(nodes, init, theta, neighbors_map)
            if res['majority_compromised']:
                maj_count += 1
            comp_fracs.append(res['compromised_count'] / total)
            
        results.append({
            "theta": theta,
            "majority_freq": maj_count / n_runs,
            "mean_comp_frac": float(np.mean(comp_fracs)),
            "std_comp_frac": float(np.std(comp_fracs))
        })
    return results

def run_bot_sweep(G, config, centralities, neighbors_map, bot_counts, placements, n_runs, fixed_theta=None):
    print("Running Bot Sweep...")
    nodes = list(G.nodes())
    total = len(nodes)
    results = {p: [] for p in placements}
    
    theta_val = fixed_theta if fixed_theta else config['theta']
    
    for b_count in bot_counts:
        for p in placements:
            cent_map = None
            if p == 'degree': cent_map = centralities['deg']
            elif p == 'betweenness': cent_map = centralities['bet']
            
            maj_count = 0
            comp_fracs = []
            
            for i in range(n_runs):
                seed = SEED_BASE + i
                roles_d = assign_roles(nodes, config['V'], config['U'], b_count, seed=seed, placement=p, cent_map=cent_map)
                st_d, mut_d = initialize_states(roles_d)
                init = {'roles': roles_d, 'states': st_d, 'mutated': mut_d}
                
                res = run_repetition_optimized(nodes, init, theta_val, neighbors_map)
                if res['majority_compromised']:
                    maj_count += 1
                comp_fracs.append(res['compromised_count'] / total)
            
            results[p].append({
                "bots": b_count,
                "majority_freq": maj_count / n_runs,
                "mean_comp_frac": float(np.mean(comp_fracs)),
                "std_comp_frac": float(np.std(comp_fracs))
            })
    return results

def run_u_sweep_transition(G, config, centralities, neighbors_map, u_counts, theta, n_runs, placement='random'):
    print(f"Running Unintentional Sweep (Theta={theta})...")
    nodes = list(G.nodes())
    total = len(nodes)
    results = []
    
    cent_map = None
    if placement == 'degree': cent_map = centralities['deg']
    elif placement == 'betweenness': cent_map = centralities['bet']
    
    for u_count in u_counts:
        maj_alt_count = 0
        alt_fracs = []
        
        for i in range(n_runs):
            seed = SEED_BASE + i
            roles_d = assign_roles(nodes, config['V'], u_count, config['B'], seed=seed, placement=placement, cent_map=cent_map)
            st_d, mut_d = initialize_states(roles_d)
            init = {'roles': roles_d, 'states': st_d, 'mutated': mut_d}
            
            res = run_repetition_optimized(nodes, init, theta, neighbors_map)
            if res['majority_altered']:
                maj_alt_count += 1
            alt_fracs.append(res['altered_count'] / total)
            
        results.append({
            "unintentional": u_count,
            "majority_altered_freq": maj_alt_count / n_runs,
            "mean_altered_frac": float(np.mean(alt_fracs)),
            "std_altered_frac": float(np.std(alt_fracs))
        })
    return results

# ============================== LOLBAS & CONTEXT ==============================

def get_lolbas_list():
    return [
        "powershell.exe", "cmd.exe", "wscript.exe", "cscript.exe",
        "mshta.exe", "rundll32.exe", "regsvr32.exe", "certutil.exe",
        "bitsadmin.exe", "wmic.exe", "bash.exe", "sh.exe",
        "grep.exe", "findstr.exe", "curl.exe", "wget.exe",
        "tar.exe", "expand.exe", "esentutl.exe", "diskshadow.exe",
        "pcalua.exe", "odbcconf.exe", "msbuild.exe", "installutil.exe"
    ]

def analyze_lolbins(G_a, G_b):
    print("Analyzing LOLBins...")
    lolbins = get_lolbas_list()
    lolbin_nodes = []
    
    for n in G_a.nodes():
        parts = n.split('::')
        if len(parts) >= 2:
            path = parts[1].lower()
            basename = os.path.basename(path)
            if basename in lolbins:
                lolbin_nodes.append(n)
                
    stats = {"total_lolbin_nodes": len(lolbin_nodes), "nodes": lolbin_nodes, "centralities": {}}
    
    if lolbin_nodes:
        deg = dict(G_a.degree())
        # Bet e PR già calcolati globalmente, ma qui li ricalcoliamo locali per semplicità o prendiamo da globali se passati
        # Per ora ricalcoliamo solo se necessario, o meglio, passiamoli come argomento se servono
        # Per questa funzione demo, lasciamo vuoto o calcoliamo quick
        for n in lolbin_nodes:
            stats["centralities"][n] = {"degree": deg[n]} 
            
    return stats

def compute_forensic_footprint(G_a, G_b, final_states):
    compromised_nodes = {n for n, s in final_states.items() if s == STATE_COMPROMISED}
    altered_nodes = {n for n, s in final_states.items() if s == STATE_ALTERED}
    affected_nodes = compromised_nodes | altered_nodes
    
    footprint = {"FILE": 0, "REG": 0, "DLL": 0, "IP": 0, "Total_Context": 0}
    counted_context = set()
    
    for p_node in affected_nodes:
        if p_node not in G_b:
            continue
        for neighbor in G_b.neighbors(p_node):
            if neighbor not in counted_context:
                n_type = G_b.nodes[neighbor].get('type', 'unknown')
                if n_type in footprint:
                    footprint[n_type] += 1
                    footprint["Total_Context"] += 1
                    counted_context.add(neighbor)
    return footprint

# ============================== PLOTTING ==============================

def plot_results(results, output_dir):
    print(f"Generating plots in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    if 'convergence' in results:
        fig, ax = plt.subplots(figsize=(8, 6))
        conv_data = results['convergence']
        for p_name, steps_data in conv_data.items():
            # steps_data è un dict {n_runs: stats}
            runs = sorted(steps_data.keys())
            freqs = [steps_data[r]['majority_frequency'] for r in runs]
            ax.plot(runs, freqs, marker='o', label=p_name.capitalize())
        ax.set_xlabel("Number of Runs")
        ax.set_ylabel("Majority Compromised Frequency")
        ax.set_title("Convergence Study")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "01_convergence.png"), dpi=300)
        plt.close()

    if 'theta_sweep' in results:
        fig, ax = plt.subplots(figsize=(8, 6))
        data = results['theta_sweep']
        if isinstance(data, list) and len(data) > 0:
            thetas = [e['theta'] for e in data]
            freqs = [e['majority_freq'] for e in data]
            ax.plot(thetas, freqs, marker='s', color='red')
            ax.set_xlabel("Threshold (Theta)")
            ax.set_ylabel("Majority Compromised Frequency")
            ax.set_title("Effect of Threshold on Spread")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "02_theta_sweep.png"), dpi=300)
            plt.close()

    if 'bot_sweep' in results:
        fig, ax = plt.subplots(figsize=(8, 6))
        data_dict = results['bot_sweep']
        if isinstance(data_dict, dict):
            for p_name, data_list in data_dict.items():
                bots = [d['bots'] for d in data_list]
                freqs = [d['majority_freq'] for d in data_list]
                ax.plot(bots, freqs, marker='o', label=p_name.capitalize())
            ax.set_xlabel("Number of Bots")
            ax.set_ylabel("Majority Compromised Frequency")
            ax.set_title("Bot Budget vs Impact")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "03_bot_sweep.png"), dpi=300)
            plt.close()

    if 'u_sweep' in results:
        fig, ax = plt.subplots(figsize=(8, 6))
        data = results['u_sweep']
        if isinstance(data, list) and len(data) > 0:
            u_counts = [e['unintentional'] for e in data]
            freqs = [e['majority_altered_freq'] for e in data]
            ax.fill_between(u_counts, freqs, alpha=0.3)
            ax.plot(u_counts, freqs, marker='x', color='purple')
            ax.set_xlabel("Number of Unintentional Nodes")
            ax.set_ylabel("Majority Altered Frequency")
            ax.set_title(f"Phase Transition: Unintentional Nodes")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "04_unintentional_transition.png"), dpi=300)
            plt.close()
            
    print("Plots generated.")

# ============================== MAIN ==============================

def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Assignment 3 Cascade")
    parser.add_argument('--plot-only', action='store_true', help="Solo plot")
    args = parser.parse_args()
    
    if args.plot_only:
        if not os.path.exists(RESULTS_FILE):
            print(f"Errore: {RESULTS_FILE} non trovato.")
            sys.exit(1)
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        plot_results(results, FIGURES_DIR)
        return

    print("--- Starting Full Analysis ---")
    random.seed(SEED_BASE)
    np.random.seed(SEED_BASE)
    
    G_a, G_b = load_graphs()
    
    # Pre-computing
    print("Pre-computing centralities (Degree, Betweenness, PageRank)...")
    t0 = time.time()
    deg_cent = dict(G_a.degree())
    bet_cent = nx.betweenness_centrality(G_a)
    pr_cent = nx.pagerank(G_a)
    neighbors_map = {n: list(G_a.neighbors(n)) for n in G_a.nodes()}
    print(f"  Centralities computed in {time.time()-t0:.2f}s")
    
    centralities = {
        'deg': deg_cent,
        'bet': bet_cent,
        'pr': pr_cent
    }
    
    results = {}
    
    # 1. Convergence
    print("Running Convergence Study...")
    conv_res = run_convergence_study(G_a, ANCHOR_CONFIG, centralities, neighbors_map, max_seeds=200)
    results['convergence'] = conv_res
    
    N_RUNS = 200
    
    # 2. Theta Sweep
    print("Running Theta Sweep...")
    thetas = [0.2, 0.3, 0.4, 0.5, 0.6]
    results['theta_sweep'] = run_theta_sweep(G_a, ANCHOR_CONFIG, centralities, neighbors_map, thetas, N_RUNS, placement='random')
    
    # 3. Bot Sweep
    print("Running Bot Sweep...")
    bot_counts = [0, 1, 2, 3, 5, 10]
    placements = ['random', 'degree', 'betweenness']
    results['bot_sweep'] = run_bot_sweep(G_a, ANCHOR_CONFIG, centralities, neighbors_map, bot_counts, placements, N_RUNS, fixed_theta=ANCHOR_CONFIG['theta'])
    
    # 4. Unintentional Sweep
    print("Running Unintentional Sweep...")
    u_counts = [0, 5, 10, 20, 30, 40, 50]
    results['u_sweep'] = run_u_sweep_transition(G_a, ANCHOR_CONFIG, centralities, neighbors_map, u_counts, ANCHOR_CONFIG['theta'], N_RUNS, placement='random')
    
    # 5. LOLBas
    lolbas_res = analyze_lolbins(G_a, G_b)
    results['lolbas'] = lolbas_res
    
    # 6. Footprint (Single Run Example)
    print("Computing Forensic Footprint (Example Run)...")
    nodes = list(G_a.nodes())
    roles_d = assign_roles(nodes, ANCHOR_CONFIG['V'], ANCHOR_CONFIG['U'], ANCHOR_CONFIG['B'], seed=SEED_BASE, placement='random', cent_map=None)
    st_d, mut_d = initialize_states(roles_d)
    init = {'roles': roles_d, 'states': st_d, 'mutated': mut_d}
    res_ex = run_repetition_optimized(nodes, init, ANCHOR_CONFIG['theta'], neighbors_map)
    footprint = compute_forensic_footprint(G_a, G_b, res_ex['final_states'])
    results['footprint_example'] = footprint
    
    save_results(results, RESULTS_FILE)
    plot_results(results, FIGURES_DIR)
    
    print("\n=== Analysis Complete ===")
    print(f"LOLBAS found: {lolbas_res['total_lolbin_nodes']} nodes")
    print(f"Footprint example: {footprint['Total_Context']} context nodes affected")

if __name__ == "__main__":
    main()