#!/usr/bin/env python3
"""
Process Provenance Graph Builder — Splunk Attack Data (v5 — NO FILTERS + QA)

Grafo eterogeneo multi-layer COMPLETO, senza filtri topologici.
Novità v5 (QA):
 - Fix GEXF: l'attributo arco 'type' viene rinominato 'edge_type' nell'export
   (GEXF riserva 'type' per directed/undirected: altrimenti sparisce in Gephi)
 - Conteggio parse error (filtro implicito quantificato e dichiarato)
 - Event 3: direzione normalizzata processo->IP, ma 'Initiated' salvato
   come attributi init_out/init_in
 - Attributo nodo 'scen' (scenario T-ID/strumento) per controlli A1

Output:
 - Grafo A (baseline omogenea): solo processi, archi 'spawned'
 - Grafo B (provenance completo): tutti i layer + archi 'executed_as'

Selezione dati: solo datasets/attack_techniques/** con T-ID nel path,
solo windows-sysmon.log, Event ID 1/3/7/11/13.

Dipendenze: pip install networkx pandas matplotlib numpy
Opzionale:  pip install powerlaw
"""

import os
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ============================== CONFIG ==============================
INCLUDE_NETWORK = True    # Event 3: tutte le connessioni (in+out)
MAX_COMMANDS    = 5       # max command-line salvate per arco (solo memoria)

MITRE_RE = re.compile(r'(T\d{4}(?:\.\d{3})?)')

# QA: quantifica le perdite implicite (il principio "zero filtri" va misurato)
PARSE_STATS = {'event_blocks': 0, 'parse_errors': 0}


# ============================== UTILS ==============================
def normalize_path(p):
    """Identità del path: minuscolo + slash uniformi. NON è un filtro."""
    if not p:
        return None
    return p.lower().replace('\\', '/').strip()


def make_unique_labels(paths, max_len=42):
    """Etichette non ambigue per i plot."""
    labels = {}
    for k in range(1, 8):
        pending = [n for n in paths if n not in labels]
        if not pending:
            break
        counts = Counter('/'.join(n.split('/')[-k:]) for n in pending)
        for n in pending:
            lab = '/'.join(n.split('/')[-k:])
            if counts[lab] == 1:
                labels[n] = lab
    for n in paths:
        labels.setdefault(n, n)
    return {n: (l if len(l) <= max_len else '…' + l[-(max_len - 1):])
            for n, l in labels.items()}


# ============================== PARSING ==============================
def iter_events_xml(filepath):
    """Estrae & Yield (event_id, fields) da un file Sysmon XML. Conta blocchi visti e parse error (QA)."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"❌[iter_events_xml] {filepath}: {e}")
        return

    starts = [m.start() for m in re.finditer(r'<Event\b', content)]
    for i, s in enumerate(starts):
        PARSE_STATS['event_blocks'] += 1
        e = starts[i + 1] if i + 1 < len(starts) else len(content)
        xml_str = content[s:e]
        cut = xml_str.rfind('</Event>')
        if cut != -1:
            xml_str = xml_str[:cut + len('</Event>')]
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError:
            PARSE_STATS['parse_errors'] += 1
            continue
        eid = root.find('.//{*}System/{*}EventID')
        if eid is None:
            continue
        ev_data = root.find('.//{*}EventData')
        if ev_data is None:
            continue
        fields = {d.get('Name'): (d.text or '') for d in ev_data.findall('{*}Data')}
        yield eid.text, fields


def parse_file(filepath):
    """Estrae TUTTI gli eventi Sysmon 1/3/7/11/13, senza filtri."""
    events = []
    m = MITRE_RE.search(str(filepath))
    mitre = m.group(1) if m else 'unknown'
    parts = Path(filepath).parts
    scen = '/'.join(parts[-3:-1]) if len(parts) >= 3 else 'unknown'

    for eid, f in iter_events_xml(filepath):
        if eid == '1':
            p, c = normalize_path(f.get('ParentImage')), normalize_path(f.get('Image'))
            if p and c:
                events.append(dict(kind='spawn', src=p, dst=c,
                                   cmd=f.get('CommandLine') or '',
                                   mitre=mitre, scen=scen))
        elif eid == '7':
            p, d = normalize_path(f.get('Image')), normalize_path(f.get('ImageLoaded'))
            if p and d:
                events.append(dict(kind='load', src=p, dst=d, mitre=mitre, scen=scen))
        elif eid == '11':
            p, d = normalize_path(f.get('Image')), normalize_path(f.get('TargetFilename'))
            if p and d:
                events.append(dict(kind='write', src=p, dst=d, mitre=mitre, scen=scen))
        elif eid == '13':
            p, d = normalize_path(f.get('Image')), normalize_path(f.get('TargetObject'))
            if p and d:
                events.append(dict(kind='reg', src=p, dst=d, mitre=mitre, scen=scen))
        elif eid == '3' and INCLUDE_NETWORK:
            p = normalize_path(f.get('Image'))
            ip = (f.get('DestinationIp') or '').strip()
            if p and ip:
                events.append(dict(kind='net', src=p, dst=ip,
                                   port=f.get('DestinationPort') or '',
                                   init=(f.get('Initiated') or '').lower(),
                                   mitre=mitre, scen=scen))
    return events


def find_logs(base_dir):
    """Solo scenari single-tecnica: sotto attack_techniques/ con T-ID nel path."""
    out = []
    for root, _d, files in os.walk(base_dir):
        if 'attack_techniques' not in Path(root).parts:
            continue
        if not MITRE_RE.search(root):
            continue
        for fn in files:
            if fn == 'windows-sysmon.log':
                out.append(os.path.join(root, fn))
    return sorted(out)


# ============================== BUILD ==============================
DST_PREFIX = {'load': 'dll', 'write': 'file', 'reg': 'reg', 'net': 'ip'}

# ex
def node_id(kind, role, path):
    """Sorgente sempre processo; destinazione prende il tipo del layer."""
    if role == 'src' or kind == 'spawn':
        return f"proc::{path}"
    return f"{DST_PREFIX[kind]}::{path}"


def build_graphs(events):
    edge_acc = {}          # (u,v,etype) -> {weight, commands, n_out, n_in}
    node_mitre_sets = {}
    node_scen_sets = {}
    executed_images = set()

    for ev in events:
        k = ev['kind']
        src = node_id(k, 'src', ev['src'])
        dst = node_id(k, 'dst', ev['dst'])

        if k == 'spawn':
            executed_images.add(ev['dst'])
            executed_images.add(ev['src'])
            etype = 'spawned'
        elif k == 'load':
            etype = 'loaded'
        elif k == 'write':
            etype = 'wrote'
        elif k == 'reg':
            etype = 'set_value'
        else:
            etype = 'connected'

        key = (src, dst, etype)
        if key not in edge_acc:
            edge_acc[key] = {'weight': 0, 'commands': [], 'n_out': 0, 'n_in': 0}
        d = edge_acc[key]
        d['weight'] += 1
        if k == 'spawn' and len(d['commands']) < MAX_COMMANDS:
            d['commands'].append(ev.get('cmd', ''))
        if k == 'net':
            if ev.get('init') == 'true':
                d['n_out'] += 1
            elif ev.get('init') == 'false':
                d['n_in'] += 1
        for nid in (src, dst):
            node_mitre_sets.setdefault(nid, set()).add(ev['mitre'])
            node_scen_sets.setdefault(nid, set()).add(ev['scen'])

    # --- Grafo B: completo + archi derivati executed_as
    B = nx.DiGraph()
    for (u, v, etype), d in edge_acc.items():
        attrs = dict(type=etype, weight=d['weight'],
                     commands=' | '.join(x for x in d['commands'] if x))
        if etype == 'connected':
            attrs['init_out'] = d['n_out']
            attrs['init_in'] = d['n_in']
        B.add_edge(u, v, **attrs)
    for fpath in {k[1][6:] for k in edge_acc if k[2] == 'wrote'}:
        if fpath in executed_images:
            B.add_edge(f"file::{fpath}", f"proc::{fpath}",
                       type='executed_as', weight=1, commands='')

    # --- Grafo A: baseline omogenea (solo spawned)
    A = nx.DiGraph()
    for (u, v, etype), d in edge_acc.items():
        if etype == 'spawned':
            A.add_edge(u, v, type=etype, weight=d['weight'],
                       commands=' | '.join(x for x in d['commands'] if x))

    # --- attributi nodo: tipo + path + ground truth MITRE + scenario
    for G in (A, B):
        for n in G.nodes():
            t, path = n.split('::', 1)
            G.nodes[n]['type'] = t
            G.nodes[n]['path'] = path
            G.nodes[n]['mitre'] = ';'.join(sorted(node_mitre_sets.get(n, set())))
            G.nodes[n]['scen'] = ';'.join(sorted(node_scen_sets.get(n, set())))
    return A, B


# ============================== ANALISI / OUTPUT ==============================
def compute_statistics(G, name):
    s = {'graph': name, 'num_nodes': G.number_of_nodes(), 'num_edges': G.number_of_edges(),
         'density': nx.density(G),
         'num_wcc': nx.number_weakly_connected_components(G)}
    if G.number_of_nodes():
        lw = max(nx.weakly_connected_components(G), key=len)
        s['largest_wcc_fraction'] = round(len(lw) / G.number_of_nodes(), 4)
        s['node_types'] = dict(Counter(G.nodes[n]['type'] for n in G.nodes()))
        s['edge_types'] = dict(Counter(G[u][v]['type'] for u, v in G.edges()))
        od = dict(G.out_degree())
        s['top_out'] = sorted(od.items(), key=lambda x: x[1], reverse=True)[:10]
    return s


def fit_power_law(G):
    try:
        import powerlaw
    except ImportError:
        return None
    degs = np.array([d for _, d in G.degree() if d > 0])
    if len(degs) < 50:
        return None
    fit = powerlaw.Fit(degs, discrete=True, verbose=False)
    res = {'alpha': float(fit.alpha), 'xmin': float(fit.xmin)}
    for other in ('exponential', 'lognormal'):
        try:
            R, p = fit.distribution_compare('power_law', other, normalized_ratio=True)
            res[f'vs_{other}'] = {'R': float(R), 'p': float(p)}
        except Exception:
            pass
    return res


def ccdf_on_ax(G, ax, label, color):
    degs = np.array([d for _, d in G.degree() if d > 0])
    if len(degs) == 0:
        return
    degs.sort()
    ccdf = np.arange(len(degs), 0, -1) / len(degs)
    ax.step(degs, ccdf, where='post', label=label, color=color)


def save_plots(A, B, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ccdf_on_ax(A, ax, 'A: processi-only', 'tab:blue')
    ccdf_on_ax(B, ax, 'B: provenance completo', 'tab:red')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('k (log)'); ax.set_ylabel('P(K ≥ k) (log)')
    ax.set_title('CCDF degree totale: A vs B'); ax.legend(); ax.grid(alpha=0.3, which='both')

    ax = axes[0, 1]
    top = sorted(B.out_degree(), key=lambda x: x[1], reverse=True)[:15]
    if top:
        nodes, degs = zip(*top)
        types = [n.split('::', 1)[0] for n in nodes]
        paths = [n.split('::', 1)[1] for n in nodes]
        labs = make_unique_labels(list(paths))
        ax.barh(range(len(nodes)), degs, color='steelblue')
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels([f"[{t}] {labs[p]}" for t, p in zip(types, paths)], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel('Out-degree')
        ax.set_title('Top 15 hub grafo B (zero filtri)')

    ax = axes[1, 0]
    comp = Counter(B.nodes[n]['type'] for n in B.nodes())
    ax.pie(comp.values(), labels=comp.keys(), autopct='%1.1f%%')
    ax.set_title('Composizione nodi grafo B')

    ax = axes[1, 1]
    wcc = Counter(len(c) for c in nx.weakly_connected_components(B))
    ax.bar(wcc.keys(), wcc.values(), color='coral')
    ax.set_yscale('log'); ax.set_xlabel('Dim. componente')
    ax.set_ylabel('N. componenti (log)'); ax.set_title('WCC grafo B'); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(outdir) / 'v5_plots.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def export(G, path_csv, path_gexf):
    rows = [{'Source': u, 'Target': v, 'Type': d['type'], 'Weight': d['weight'],
             'Commands': d.get('commands', '')} for u, v, d in G.edges(data=True)]
    pd.DataFrame(rows).to_csv(path_csv, index=False)
    Gx = G.copy()
    for _u, _v, d in Gx.edges(data=True):
        d['edge_type'] = d.pop('type')   # evita collisione col 'type' riservato GEXF
    nx.write_gexf(Gx, path_gexf)


# ============================== MAIN ==============================
def main():
    print("Process Provenance Graph Builder (v5 — NO FILTERS + QA)")
    print("=" * 60)
    OUT = Path('output'); OUT.mkdir(exist_ok=True)

    logs = find_logs('datasets')
    print(f"{len(logs)} log sysmon (attack_techniques + T-ID)")
    if not logs:
        print("❌ Nessun log: git lfs pull?"); return

    events = []
    for i, fp in enumerate(logs, 1):
        if i % 25 == 0 or i == len(logs):
            print(f"  parsing [{i}/{len(logs)}]…")
        events.extend(parse_file(fp))
    tot = PARSE_STATS['event_blocks']
    err = PARSE_STATS['parse_errors']
    print(f"✅ Eventi totali (1/3/7/11/13): {len(events)}")
    print(f"🧾 QA parsing: blocchi={tot} | parse errors={err} "
          f"({err / max(1, tot):.2%})")
    if not events:
        print("❌ Nessun evento."); return

    A, B = build_graphs(events)
    print(f" Grafo A (baseline): {A.number_of_nodes()} nodi, {A.number_of_edges()} archi")
    print(f" Grafo B (completo): {B.number_of_nodes()} nodi, {B.number_of_edges()} archi")

    export(A, OUT / 'v5_graphA_edgelist.csv', OUT / 'v5_graphA.gexf')
    export(B, OUT / 'v5_graphB_edgelist.csv', OUT / 'v5_graphB.gexf')

    stats = {'graphA': compute_statistics(A, 'A'),
             'graphB': compute_statistics(B, 'B'),
             'parse_stats': dict(PARSE_STATS)}
    print("⏳ Fit power-law…")
    stats['graphA']['power_law_fit'] = fit_power_law(A)
    stats['graphB']['power_law_fit'] = fit_power_law(B)
    with open(OUT / 'v5_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)

    for g in ('graphA', 'graphB'):
        s = stats[g]
        print("\n" + "=" * 60)
        print(f"📊 {s['graph']}  |  nodi={s['num_nodes']}  archi={s['num_edges']}  "
              f"WCCmax={s.get('largest_wcc_fraction')}")
        print("   tipi nodo:", s.get('node_types'))
        print("   tipi arco:", s.get('edge_types'))
        print("   top out-degree:")
        for n, d in s.get('top_out', []):
            print(f"     {d:>6}  {n}")
        pl = s.get('power_law_fit')
        if pl:
            print(f"   power-law: alpha={pl['alpha']:.3f} xmin={pl['xmin']} "
                  f"| vs_exp R={pl.get('vs_exponential', {}).get('R', float('nan')):.2f}")
    print("=" * 60)

    save_plots(A, B, OUT)
    print("\n🎉 v5 completata! File in output/: v5_graphA*, v5_graphB*, "
          "v5_statistics.json, v5_plots.png")


if __name__ == '__main__':
    main()