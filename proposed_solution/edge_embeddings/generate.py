import os
import json
import numpy as np
import pandas as pd
import torch
from model import EdgeEmbeddingModel

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# paths
NODE_EMB_PATH = "baseline/node_embeddings.npy"
NODE_EDGE_EMB_PATH = "baseline/node_edge_embeddings.npy"
NODE_EMB_META_PATH = "baseline/node_embeddings_meta.json"
EDGES_PATH = "data/processed/edges.csv"
MODEL_PATH = "proposed_solution/edge_embeddings/trained_model.pt"
OUTPUT_PATH = "proposed_solution/edge_embeddings/edge_embeddings.json"

RELATION_TYPES = [
    'parent_taxon', 'eats', 'preys_on', 'scavenges_from',
    'disperses_seeds_of', 'symbiotic_with', 'migrates_with',
    'parasitizes', 'pollinates'
]


def load_node_embeddings():
    wiki_emb = np.load(NODE_EMB_PATH)
    edge_emb = np.load(NODE_EDGE_EMB_PATH)
    combined = np.concatenate([wiki_emb, edge_emb], axis=1)

    with open(NODE_EMB_META_PATH, 'r') as f:
        meta = json.load(f)

    
    qid_to_idx = {qid: i for i, qid in enumerate(meta['node_order'])}
    return combined, qid_to_idx


def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model
    model = EdgeEmbeddingModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # load node embeddings
    combined_emb, qid_to_idx = load_node_embeddings()

    # load edges and group by unique node pair
    edges_df = pd.read_csv(EDGES_PATH)

    pair_to_relations = {}
    for _, row in edges_df.iterrows():
        h = row['subject_id']
        t = row['object_id']
        r = row['relation']
        

        if h not in qid_to_idx or t not in qid_to_idx:
            continue

        key = (h, t)
        if key not in pair_to_relations:
            pair_to_relations[key] = []
        pair_to_relations[key].append(r)

    print(f"Total unique node pairs: {len(pair_to_relations)}")

    # generate edge embeddings
    edge_embeddings = {}

    for (h, t), relations in pair_to_relations.items():
        h_emb = combined_emb[qid_to_idx[h]]
        t_emb = combined_emb[qid_to_idx[t]]

        pair_input = torch.tensor(
            np.concatenate([h_emb, t_emb]),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        embedding = model.get_embedding(pair_input)
        embedding_list = embedding.squeeze(0).cpu().numpy().tolist()

        # store with all metadata
        edge_key = f"{h}__{t}"
        edge_embeddings[edge_key] = {
            "head": h,
            "tail": t,
            "relations": relations,
            "embedding": embedding_list
        }

    # save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(edge_embeddings, f)

    print(f"Saved {len(edge_embeddings)} edge embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()


