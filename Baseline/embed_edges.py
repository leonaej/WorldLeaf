import pandas as pd
import json
import os
import time
import sys
import numpy as np
from openai import OpenAI
import tiktoken

# ── Config ──────────────────────────────────────────────────────────────────
EDGES_PATH       = "data/processed/edges.csv"
NODES_PATH       = "data/processed/nodes.csv"
EMBEDDINGS_PATH  = "baseline/node_edge_embeddings.npy"
METADATA_PATH    = "baseline/node_edge_embeddings_meta.json"
MODEL            = "text-embedding-3-large"
MAX_TOKENS       = 8000
CHECKPOINT_EVERY = 50

client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
encoder = tiktoken.get_encoding("cl100k_base")

# ── Truncate text ─────────────────────────────────────────────────────────────
def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return encoder.decode(tokens)
    return text

# ── Embed a single text ───────────────────────────────────────────────────────
def embed_one(text: str) -> list:
    try:
        truncated = truncate_text(text)
        response  = client.embeddings.create(model=MODEL, input=[truncated])
        return response.data[0].embedding
    except Exception as e:
        print(f"  [ERROR] Embedding failed: {e}")
        return None

# ── Build edge text for a node ────────────────────────────────────────────────
def build_edge_text(node_id: str, node_name: str, outgoing: dict, incoming: dict) -> str:
    if not outgoing and not incoming:
        return node_name

    lines = [f"Species: {node_name}"]

    relation_labels = {
        "eats":               "Eats",
        "preys_on":           "Preys on",
        "parasitizes":        "Parasitizes",
        "pollinates":         "Pollinates",
        "parent_taxon":       "Parent taxon",
        "scavenges_from":     "Scavenges from",
        "migrates_with":      "Migrates with",
        "disperses_seeds_of": "Disperses seeds of",
        "symbiotic_with":     "Symbiotic with"
    }

    inverse_labels = {
        "eats":               "Eaten by",
        "preys_on":           "Preyed on by",
        "parasitizes":        "Parasitized by",
        "pollinates":         "Pollinated by",
        "parent_taxon":       "Parent taxon of",
        "scavenges_from":     "Scavenged from by",
        "migrates_with":      "Migrates with",
        "disperses_seeds_of": "Seeds dispersed by",
        "symbiotic_with":     "Symbiotic with"
    }

    for rel, objects in outgoing.items():
        label = relation_labels.get(rel, rel)
        lines.append(f"{label}: {', '.join(objects)}")

    for rel, subjects in incoming.items():
        label = inverse_labels.get(rel, rel)
        lines.append(f"{label}: {', '.join(subjects)}")

    return "\n".join(lines)

# ── Build lookups ─────────────────────────────────────────────────────────────
def build_lookups():
    edges_df = pd.read_csv(EDGES_PATH)
    nodes_df = pd.read_csv(NODES_PATH)

    node_lookup = {}
    for _, row in nodes_df.iterrows():
        node_id = str(row["node_id"])
        name = row["common_name"] if pd.notna(row["common_name"]) and str(row["common_name"]).strip() not in ("", "nan") else row["name"]
        node_lookup[node_id] = str(name).strip()

    outgoing = {nid: {} for nid in node_lookup}
    incoming = {nid: {} for nid in node_lookup}

    for _, edge in edges_df.iterrows():
        subj      = str(edge["subject_id"]).strip()
        obj       = str(edge["object_id"]).strip()
        rel       = str(edge["relation"]).strip()
        obj_name  = node_lookup.get(obj, str(edge["object_label"]).strip())
        subj_name = node_lookup.get(subj, str(edge["subject_label"]).strip())

        if subj in outgoing:
            if rel not in outgoing[subj]:
                outgoing[subj][rel] = []
            outgoing[subj][rel].append(obj_name)

        if obj in incoming:
            if rel not in incoming[obj]:
                incoming[obj][rel] = []
            incoming[obj][rel].append(subj_name)

    return edges_df, node_lookup, outgoing, incoming

# ── Save checkpoint ───────────────────────────────────────────────────────────
def save_checkpoint(embeddings_dict: dict, node_order: list):
    matrix = np.array([embeddings_dict[nid] for nid in node_order if nid in embeddings_dict])
    np.save(EMBEDDINGS_PATH, matrix)
    meta = {
        "model":          MODEL,
        "dimensions":     3072,
        "node_order":     [nid for nid in node_order if nid in embeddings_dict],
        "total_embedded": len(embeddings_dict)
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [CHECKPOINT] Saved {len(embeddings_dict)} embeddings to {EMBEDDINGS_PATH}")

# ── Preview mode ──────────────────────────────────────────────────────────────
def preview():
    _, node_lookup, outgoing, incoming = build_lookups()
    print("── Preview: first 5 node edge texts ────────────────")
    for nid in list(node_lookup.keys())[:5]:
        name  = node_lookup[nid]
        text  = build_edge_text(nid, name, outgoing[nid], incoming[nid])
        tokens = len(encoder.encode(text))
        print(f"\n--- {name} ({tokens} tokens) ---")
        print(text)
    print("\n────────────────────────────────────────────────────")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    edges_df, node_lookup, outgoing, incoming = build_lookups()
    all_node_ids = list(node_lookup.keys())
    print(f"Loaded {len(all_node_ids)} nodes and {len(edges_df)} edges")

    # Resume support
    embeddings_dict = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
        already_done = set(meta.get("node_order", []))
        if os.path.exists(EMBEDDINGS_PATH):
            matrix = np.load(EMBEDDINGS_PATH)
            for i, nid in enumerate(meta["node_order"]):
                embeddings_dict[nid] = matrix[i].tolist()
        print(f"Resuming — {len(embeddings_dict)} nodes already embedded, skipping those.")
    else:
        already_done = set()

    remaining = [nid for nid in all_node_ids if nid not in already_done]
    print(f"Remaining to embed: {len(remaining)}\n")

    for i, nid in enumerate(remaining):
        name       = node_lookup[nid]
        edge_text  = build_edge_text(nid, name, outgoing[nid], incoming[nid])
        token_count = len(encoder.encode(edge_text))

        print(f"[{i+1}/{len(remaining)}] {name} ({token_count} tokens)")

        emb = embed_one(edge_text)
        embeddings_dict[nid] = emb if emb is not None else [0.0] * 3072

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(embeddings_dict, all_node_ids)

        time.sleep(0.1)

    save_checkpoint(embeddings_dict, all_node_ids)

    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"Total nodes embedded: {len(embeddings_dict)}")
    print(f"Embedding matrix:     {len(embeddings_dict)} x 3072")
    print(f"Saved to:             {EMBEDDINGS_PATH}")
    print(f"Metadata saved to:    {METADATA_PATH}")
    print(f"────────────────────────────────────────────────────")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        preview()
    else:
        main()