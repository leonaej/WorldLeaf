import json
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
NODES_PATH     = "data/processed/nodes.csv"
EDGES_PATH     = "data/processed/edges.csv"

NODE_EMB_NPY   = "Baseline/node_embeddings.npy"
NODE_EMB_META  = "Baseline/node_embeddings_meta.json"

EDGE_EMB_NPY   = "proposed_solution/edge_embeddings2/edge_embeddings.npy"
EDGE_EMB_META  = "proposed_solution/edge_embeddings2/edge_embeddings_meta.json"

QUERY_EMB_NPY  = "proposed_solution/RL_agent/query_embeddings.npy"
QUERY_EMB_META = "proposed_solution/RL_agent/query_embeddings_meta.json"

SINGLE_HOP_TRAIN = "QA/single_hop_qa.json"
MULTI_HOP_TRAIN  = "QA/multi_hop_qa.json"

SINGLE_HOP_EVAL  = "QA/single_hop_qa_fixed.json"
MULTI_HOP_EVAL   = "QA/multi_hop_qa_fixed.json"


# ── graph ──────────────────────────────────────────────────────────────────
def load_graph():
    """
    Returns:
        adjacency: dict {node_id: [(neighbor_id, relation, edge_key), ...]}
        node_info: dict {node_id: {name, common_name, rank, iconic_taxon}}
    """
    nodes_df = pd.read_csv(NODES_PATH)
    edges_df = pd.read_csv(EDGES_PATH)

    # node info lookup
    node_info = {}
    for _, row in nodes_df.iterrows():
        node_info[row['node_id']] = {
            "name":         row['name']         if pd.notna(row['name'])         else None,
            "common_name":  row['common_name']  if pd.notna(row['common_name'])  else None,
            "rank":         row['rank']         if pd.notna(row['rank'])         else None,
            "iconic_taxon": row['iconic_taxon'] if pd.notna(row['iconic_taxon']) else None
        }

    # adjacency list
    adjacency = {nid: [] for nid in node_info}
    for _, row in edges_df.iterrows():
        h = row['subject_id']
        t = row['object_id']
        r = row['relation']
        edge_key = f"{h}__{t}"
        if h in adjacency:
            adjacency[h].append((t, r, edge_key))

    print(f"Graph loaded: {len(node_info)} nodes, {len(edges_df)} edges")
    return adjacency, node_info


# ── node embeddings (wikipedia, 3072d) ────────────────────────────────────
def load_node_embeddings():
    """
    Returns:
        matrix:      np.ndarray (1423, 3072)
        node_to_idx: dict {node_id: row_index}
        idx_to_node: dict {row_index: node_id}
    """
    matrix = np.load(NODE_EMB_NPY)

    with open(NODE_EMB_META, 'r') as f:
        meta = json.load(f)

    node_order = meta['node_order']  # list of node_ids in row order
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_order)}
    idx_to_node = {idx: node_id for idx, node_id in enumerate(node_order)}

    print(f"Node embeddings loaded: {matrix.shape}")
    return matrix, node_to_idx, idx_to_node


# ── edge embeddings (openai, 3072d) ───────────────────────────────────────
def load_edge_embeddings():
    """
    Returns:
        matrix:      np.ndarray (1519, 3072)
        edge_to_idx: dict {head__tail: row_index}
        idx_to_edge: dict {row_index: {head, tail, relations, text}}
    """
    matrix = np.load(EDGE_EMB_NPY)

    with open(EDGE_EMB_META, 'r') as f:
        meta = json.load(f)  # list of entries

    edge_to_idx = {}
    idx_to_edge = {}
    for entry in meta:
        key = f"{entry['head']}__{entry['tail']}"
        idx = entry['index']
        edge_to_idx[key] = idx
        idx_to_edge[idx] = {
            "head":      entry['head'],
            "tail":      entry['tail'],
            "relations": entry['relations'],
            "text":      entry['text']
        }

    print(f"Edge embeddings loaded: {matrix.shape}")
    return matrix, edge_to_idx, idx_to_edge


# ── query embeddings (3072d) ──────────────────────────────────────────────
def load_query_embeddings():
    """
    Returns:
        matrix:       np.ndarray (1753, 3072)
        query_lookup: dict {question_text: {index, answer_nodes, hop_type}}
    """
    matrix = np.load(QUERY_EMB_NPY)

    with open(QUERY_EMB_META, 'r') as f:
        meta = json.load(f)  # list of entries

    query_lookup = {}
    for entry in meta:
        query_lookup[entry['question']] = {
            "index":        entry['index'],
            "answer_nodes": entry['answer_nodes'],
            "hop_type":     entry['hop_type']
        }

    print(f"Query embeddings loaded: {matrix.shape}")
    return matrix, query_lookup


# ── training data (10k, one answer per row) ───────────────────────────────
def load_training_data():
    """
    Returns:
        list of {question, answer_node_id, answer_node_name, hop_type}
    """
    training = []

    with open(SINGLE_HOP_TRAIN, 'r') as f:
        single = json.load(f)
    for qa in single:
        training.append({
            "question":         qa['question'],
            "answer_node_id":   qa['answer_node_id'],
            "answer_node_name": qa.get('answer_name', ''),
            "hop_type":         "single"
        })

    with open(MULTI_HOP_TRAIN, 'r') as f:
        multi = json.load(f)
    for qa in multi:
        training.append({
            "question":         qa['question'],
            "answer_node_id":   qa['answer_node_id'],
            "answer_node_name": qa.get('answer_name', ''),
            "hop_type":         "multi"
        })

    print(f"Training data loaded: {len(training)} episodes")
    return training


# ── eval data (grouped answers) ───────────────────────────────────────────
def load_eval_data():
    """
    Returns:
        list of {question, answer_nodes, hop_type}
        answer_nodes is a list of {id, name}
    """
    eval_data = []

    with open(SINGLE_HOP_EVAL, 'r') as f:
        single = json.load(f)
    for qa in single:
        eval_data.append({
            "question":     qa['question'],
            "answer_nodes": qa['answer_nodes'],
            "hop_type":     "single"
        })

    with open(MULTI_HOP_EVAL, 'r') as f:
        multi = json.load(f)
    for qa in multi:
        eval_data.append({
            "question":     qa['question'],
            "answer_nodes": qa['answer_nodes'],
            "hop_type":     "multi"
        })

    print(f"Eval data loaded: {len(eval_data)} unique questions")
    return eval_data


# ── cosine similarity ─────────────────────────────────────────────────────
def cosine_similarity(vec, matrix):
    """
    Cosine similarity between one vector and all rows in a matrix.

    Args:
        vec:    np.ndarray (d,)
        matrix: np.ndarray (N, d)
    Returns:
        similarities: np.ndarray (N,)
    """
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return mat_norm @ vec_norm