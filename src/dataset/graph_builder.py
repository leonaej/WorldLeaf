import pandas as pd
import networkx as nx
import pickle
import os

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
def load_nodes() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "species_with_qids_filtered.csv")
    df = pd.read_csv(path)
    df = df[df['wikidata_qid'].notna()]
    print(f"Loaded {len(df)} nodes")
    return df

def load_edges() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "edges_all_filtered.csv")
    df = pd.read_csv(path)
    df = df[
        df['subject_id'].notna() & (df['subject_id'] != '') &
        df['object_id'].notna() & (df['object_id'] != '')
    ]
    print(f"Loaded {len(df)} edges")
    return df

# ── Build Graph ───────────────────────────────────────────────────────────────
def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        qid = str(row['wikidata_qid']).strip()
        G.add_node(qid,
            name=str(row.get('name', '')),
            common_name=str(row.get('common_name', '')),
            rank=str(row.get('rank', '')),
            iconic_taxon=str(row.get('iconic_taxon', ''))
        )

    print(f"Added {G.number_of_nodes()} nodes from species list")

    # Add edges
    edges_added = 0
    edges_skipped = 0
    new_nodes = 0

    for _, row in edges_df.iterrows():
        subj = str(row['subject_id']).strip()
        obj = str(row['object_id']).strip()
        rel = str(row['relation']).strip()
        subj_label = str(row.get('subject_label', '')).strip()
        obj_label = str(row.get('object_label', '')).strip()

        # Add node if not already in graph (taxonomy nodes etc)
        if subj not in G:
            G.add_node(subj, name=subj_label, common_name='', rank='', iconic_taxon='unknown')
            new_nodes += 1
        if obj not in G:
            G.add_node(obj, name=obj_label, common_name='', rank='', iconic_taxon='unknown')
            new_nodes += 1

        G.add_edge(subj, obj, relation=rel,
                   subject_label=subj_label,
                   object_label=obj_label)
        edges_added += 1

    print(f"Added {edges_added} edges")
    print(f"Added {new_nodes} extra nodes from edge targets (taxonomy etc)")
    print(f"Total nodes in graph: {G.number_of_nodes()}")

    return G

# ── Stats ─────────────────────────────────────────────────────────────────────
def print_stats(G: nx.DiGraph, edges_df: pd.DataFrame):
    print("\n── Graph Statistics ──────────────────────────────")
    print(f"Nodes:          {G.number_of_nodes()}")
    print(f"Edges:          {G.number_of_edges()}")
    print(f"Is connected:   {nx.is_weakly_connected(G)}")
    print(f"Density:        {nx.density(G):.6f}")

    print("\nEdges by relation:")
    print(edges_df['relation'].value_counts().to_string())

    # Isolated nodes
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    print(f"\nIsolated nodes (degree 0): {len(isolated)}")

    # Top 10 most connected nodes
    degree_dict = dict(G.degree())
    top10 = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 most connected nodes:")
    for qid, deg in top10:
        name = G.nodes[qid].get('name', qid)
        print(f"  {name} ({qid}): degree {deg}")

# ── Save ──────────────────────────────────────────────────────────────────────
def save_outputs(G: nx.DiGraph):
    # Save nodes CSV
    node_rows = []
    for qid, attrs in G.nodes(data=True):
        node_rows.append({
            'node_id': qid,
            'name': attrs.get('name', ''),
            'common_name': attrs.get('common_name', ''),
            'rank': attrs.get('rank', ''),
            'iconic_taxon': attrs.get('iconic_taxon', '')
        })
    nodes_df = pd.DataFrame(node_rows)
    nodes_path = os.path.join(PROCESSED_DIR, "nodes.csv")
    nodes_df.to_csv(nodes_path, index=False)
    print(f"\nSaved nodes → {nodes_path}")

    # Save edges CSV
    edge_rows = []
    for u, v, attrs in G.edges(data=True):
        edge_rows.append({
            'subject_id': u,
            'object_id': v,
            'relation': attrs.get('relation', ''),
            'subject_label': attrs.get('subject_label', ''),
            'object_label': attrs.get('object_label', '')
        })
    edges_df = pd.DataFrame(edge_rows)
    edges_path = os.path.join(PROCESSED_DIR, "edges.csv")
    edges_df.to_csv(edges_path, index=False)
    print(f"Saved edges → {edges_path}")

    # Save graph pickle
    pickle_path = os.path.join(PROCESSED_DIR, "graph.gpickle")
    with open(pickle_path, 'wb') as f:
        pickle.dump(G, f)
    print(f"Saved graph → {pickle_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    nodes_df = load_nodes()
    edges_df = load_edges()
    G = build_graph(nodes_df, edges_df)
    print_stats(G, edges_df)
    save_outputs(G)
    print("\nDone!")