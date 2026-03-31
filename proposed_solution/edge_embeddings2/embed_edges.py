import os
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# paths
EDGES_PATH = "data/processed/edges.csv"
NODES_PATH = "data/processed/nodes.csv"
OUTPUT_NPY = "proposed_solution/edge_embeddings2/edge_embeddings.npy"
OUTPUT_META = "proposed_solution/edge_embeddings2/edge_embeddings_meta.json"
CHECKPOINT_PATH = "proposed_solution/edge_embeddings2/checkpoint.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

RELATION_TO_TEXT = {
    'parent_taxon':       'belongs to the taxonomic group',
    'eats':               'eats',
    'preys_on':           'preys on',
    'scavenges_from':     'scavenges from',
    'disperses_seeds_of': 'disperses seeds of',
    'symbiotic_with':     'is symbiotic with',
    'migrates_with':      'migrates with',
    'parasitizes':        'parasitizes',
    'pollinates':         'pollinates'
}


def build_name_lookup(nodes_df):
    lookup = {}
    for _, row in nodes_df.iterrows():
        qid = row['node_id']
        scientific = row['name'] if pd.notna(row['name']) and str(row['name']).strip() else None
        common = row['common_name'] if pd.notna(row['common_name']) and str(row['common_name']).strip() else None
        lookup[qid] = {'scientific': scientific, 'common': common}
    return lookup


def get_display_name(qid, lookup, prefer='common'):
    if qid not in lookup:
        return qid
    names = lookup[qid]
    if prefer == 'common':
        return names['common'] or names['scientific'] or qid
    else:
        return names['scientific'] or names['common'] or qid


def build_edge_text(h_qid, t_qid, relations, lookup):
    sentences = []
    for relation in relations:
        relation_text = RELATION_TO_TEXT.get(relation, relation)

        h_common = get_display_name(h_qid, lookup, prefer='common')
        t_common = get_display_name(t_qid, lookup, prefer='common')
        sentence1 = f"{h_common} {relation_text} {t_common}."

        h_scientific = get_display_name(h_qid, lookup, prefer='scientific')
        t_scientific = get_display_name(t_qid, lookup, prefer='scientific')

        if h_scientific != h_common or t_scientific != t_common:
            sentence2 = f"{h_scientific} {relation_text} {t_scientific}."
            sentences.append(f"{sentence1} {sentence2}")
        else:
            sentences.append(sentence1)

    return " ".join(sentences)


def embed_text(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            return json.load(f)
    return {"embeddings": [], "meta": []}


def save_checkpoint(data):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(data, f)


def main():
    # load data
    edges_df = pd.read_csv(EDGES_PATH)
    nodes_df = pd.read_csv(NODES_PATH)
    lookup = build_name_lookup(nodes_df)

    # group edges by unique node pair
    pair_to_relations = {}
    for _, row in edges_df.iterrows():
        h = row['subject_id']
        t = row['object_id']
        r = row['relation']
        key = (h, t)
        if key not in pair_to_relations:
            pair_to_relations[key] = []
        if r not in pair_to_relations[key]:
            pair_to_relations[key].append(r)

    print(f"Total unique node pairs: {len(pair_to_relations)}")

    # load checkpoint
    checkpoint = load_checkpoint()
    embeddings = checkpoint["embeddings"]
    meta = checkpoint["meta"]
    done_keys = {f"{m['head']}__{m['tail']}" for m in meta}
    print(f"Resuming from checkpoint: {len(meta)} pairs already embedded")

    # embed each pair
    pairs = list(pair_to_relations.items())
    for i, ((h, t), relations) in enumerate(tqdm(pairs)):
        edge_key = f"{h}__{t}"

        if edge_key in done_keys:
            continue

        text = build_edge_text(h, t, relations, lookup)
        embedding = embed_text(text)

        embeddings.append(embedding)
        meta.append({
            "index": len(meta),
            "head": h,
            "tail": t,
            "relations": relations,
            "text": text
        })

        # checkpoint every 50 pairs
        if (i + 1) % 50 == 0:
            save_checkpoint({"embeddings": embeddings, "meta": meta})
            print(f"Checkpoint saved at {i+1} pairs")

    # save final outputs
    embedding_matrix = np.array(embeddings)
    np.save(OUTPUT_NPY, embedding_matrix)
    print(f"Saved embedding matrix: {embedding_matrix.shape}")

    with open(OUTPUT_META, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {len(meta)} entries")

    # cleanup checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"Done! Edge embeddings saved to {OUTPUT_NPY} and {OUTPUT_META}")


if __name__ == "__main__":
    main()