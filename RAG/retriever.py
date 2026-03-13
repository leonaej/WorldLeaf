import numpy as np
import json
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "baseline/node_embeddings.npy")
WIKI_METADATA_PATH   = os.path.join(BASE_DIR, "baseline/node_embeddings_meta.json")
EDGE_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "baseline/node_edge_embeddings.npy")
EDGE_METADATA_PATH   = os.path.join(BASE_DIR, "baseline/node_edge_embeddings_meta.json")
NODE_TEXTS_PATH      = os.path.join(BASE_DIR, "baseline/node_texts.json")
EDGE_TEXTS_PATH      = os.path.join(BASE_DIR, "baseline/node_edge_texts.json")
MODEL                = "text-embedding-3-large"
TOP_K                = 3

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Load embeddings and build index ──────────────────────────────────────────
def load_index():
    # Load Wikipedia embeddings
    wiki_matrix = np.load(WIKI_EMBEDDINGS_PATH)
    with open(WIKI_METADATA_PATH, "r") as f:
        wiki_meta = json.load(f)
    wiki_node_order = wiki_meta["node_order"]

    # Load edge embeddings
    edge_matrix = np.load(EDGE_EMBEDDINGS_PATH)
    with open(EDGE_METADATA_PATH, "r") as f:
        edge_meta = json.load(f)
    edge_node_order = edge_meta["node_order"]

    # Make sure node orders match
    assert wiki_node_order == edge_node_order, "Node orders don't match!"

    # Concatenate: (1423 x 3072) + (1423 x 3072) = (1423 x 6144)
    combined_matrix = np.concatenate([wiki_matrix, edge_matrix], axis=1)

    # Load node texts
    with open(NODE_TEXTS_PATH, "r", encoding="utf-8") as f:
        node_texts = json.load(f)
    with open(EDGE_TEXTS_PATH, "r", encoding="utf-8") as f:
        edge_texts = json.load(f)

    print(f"Index loaded: {combined_matrix.shape[0]} nodes x {combined_matrix.shape[1]} dimensions")
    return combined_matrix, wiki_node_order, node_texts, edge_texts

# ── Embed a query ─────────────────────────────────────────────────────────────
def embed_query(query: str) -> np.ndarray:
    try:
        response = client.embeddings.create(model=MODEL, input=[query])
        query_vec = np.array(response.data[0].embedding)  # 3072d
        return np.concatenate([query_vec, query_vec])      # 6144d
    except Exception as e:
        print(f"[ERROR] Query embedding failed: {e}")
        return None

# ── Retrieve top-k nodes ──────────────────────────────────────────────────────
def retrieve(query: str, combined_matrix: np.ndarray, node_order: list,
             node_texts: dict, edge_texts: dict, top_k: int = TOP_K) -> list:
    query_vec = embed_query(query)
    if query_vec is None:
        return []

    scores = cosine_similarity(query_vec.reshape(1, -1), combined_matrix)[0]
    top_k_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_k_indices:
        node_id = node_order[idx]
        results.append({
            "node_id":   node_id,
            "name":      node_texts.get(node_id, {}).get("name", node_id),
            "score":     float(scores[idx]),
            "wiki_text": node_texts.get(node_id, {}).get("text", ""),
            "edge_text": edge_texts.get(node_id, {}).get("text", "")
        })

    return results