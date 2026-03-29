import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import EdgeEmbeddingModel

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# paths
NODE_EMB_PATH = "baseline/node_embeddings.npy"
NODE_EDGE_EMB_PATH = "baseline/node_edge_embeddings.npy"
NODE_EMB_META_PATH = "baseline/node_embeddings_meta.json"
EDGES_PATH = "data/processed/edges.csv"
SAVE_PATH = "proposed_solution/edge_embeddings/trained_model.pt"

RELATION_TYPES = [
    'parent_taxon', 'eats', 'preys_on', 'scavenges_from',
    'disperses_seeds_of', 'symbiotic_with', 'migrates_with',
    'parasitizes', 'pollinates'
]

# hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NEGATIVE_SAMPLES_PER_POSITIVE = 5


class EdgeDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = torch.tensor(pairs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


def load_node_embeddings():
    wiki_emb = np.load(NODE_EMB_PATH)
    edge_emb = np.load(NODE_EDGE_EMB_PATH)
    combined = np.concatenate([wiki_emb, edge_emb], axis=1)

    with open(NODE_EMB_META_PATH, 'r') as f:
        meta = json.load(f)

    qid_to_idx = {qid: i for i, qid in enumerate(meta['node_order'])}
    return combined, qid_to_idx


def build_dataset(combined_emb, qid_to_idx):
    edges_df = pd.read_csv(EDGES_PATH)

    # group edges by unique node pair → multi-label vector
    pair_to_label = {}
    for _, row in edges_df.iterrows():
        h = row['subject_id']
        t = row['object_id']
        r = row['relation']

        if h not in qid_to_idx or t not in qid_to_idx:
            continue

        key = (h, t)
        if key not in pair_to_label:
            pair_to_label[key] = [0] * len(RELATION_TYPES)

        if r in RELATION_TYPES:
            pair_to_label[key][RELATION_TYPES.index(r)] = 1

    # build positive examples
    positive_pairs = []
    positive_labels = []
    connected_set = set(pair_to_label.keys())

    for (h, t), label in pair_to_label.items():
        h_emb = combined_emb[qid_to_idx[h]]
        t_emb = combined_emb[qid_to_idx[t]]
        positive_pairs.append(np.concatenate([h_emb, t_emb]))
        positive_labels.append(label)

    # build negative examples
    all_qids = list(qid_to_idx.keys())
    negative_pairs = []
    negative_labels = []
    rng = np.random.default_rng(42)

    num_negatives = len(positive_pairs) * NEGATIVE_SAMPLES_PER_POSITIVE
    attempts = 0
    while len(negative_pairs) < num_negatives and attempts < num_negatives * 10:
        h = all_qids[rng.integers(len(all_qids))]
        t = all_qids[rng.integers(len(all_qids))]
        attempts += 1
        if h == t or (h, t) in connected_set:
            continue
        h_emb = combined_emb[qid_to_idx[h]]
        t_emb = combined_emb[qid_to_idx[t]]
        negative_pairs.append(np.concatenate([h_emb, t_emb]))
        negative_labels.append([0] * len(RELATION_TYPES))

    all_pairs = np.array(positive_pairs + negative_pairs)
    all_labels = np.array(positive_labels + negative_labels)

    print(f"Positive pairs: {len(positive_pairs)}")
    print(f"Negative pairs: {len(negative_pairs)}")
    print(f"Total training examples: {len(all_pairs)}")

    return all_pairs, all_labels


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    combined_emb, qid_to_idx = load_node_embeddings()
    all_pairs, all_labels = build_dataset(combined_emb, qid_to_idx)

    dataset = EdgeDataset(all_pairs, all_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EdgeEmbeddingModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for pairs, labels in dataloader:
            pairs = pairs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            _, predictions = model(pairs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    train()