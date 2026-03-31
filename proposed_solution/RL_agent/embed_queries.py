import os
import json
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# paths
SINGLE_HOP_PATH = "QA/single_hop_qa_fixed.json"
MULTI_HOP_PATH = "QA/multi_hop_qa_fixed.json"
OUTPUT_NPY = "proposed_solution/RL_agent/query_embeddings.npy"
OUTPUT_META = "proposed_solution/RL_agent/query_embeddings_meta.json"
CHECKPOINT_PATH = "proposed_solution/RL_agent/query_embeddings_checkpoint.json"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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
    # load all unique questions from both files
    with open(SINGLE_HOP_PATH, 'r') as f:
        single_hop = json.load(f)
    with open(MULTI_HOP_PATH, 'r') as f:
        multi_hop = json.load(f)

    # collect unique questions — only what we need
    all_questions = []
    seen = set()

    for qa in single_hop:
        q = qa['question']
        if q not in seen:
            seen.add(q)
            all_questions.append({
                "question": q,
                "hop_type": "single",
                "answer_nodes": qa['answer_nodes']
            })

    for qa in multi_hop:
        q = qa['question']
        if q not in seen:
            seen.add(q)
            all_questions.append({
                "question": q,
                "hop_type": "multi",
                "answer_nodes": qa['answer_nodes']
            })

    print(f"Total unique questions: {len(all_questions)}")

    # load checkpoint
    checkpoint = load_checkpoint()
    embeddings = checkpoint["embeddings"]
    meta = checkpoint["meta"]
    done_questions = {m['question'] for m in meta}
    print(f"Resuming from checkpoint: {len(meta)} already embedded")

    # embed each question
    for i, qa in enumerate(tqdm(all_questions)):
        if qa['question'] in done_questions:
            continue

        embedding = embed_text(qa['question'])
        embeddings.append(embedding)
        meta.append({
            "index": len(meta),
            "question": qa['question'],
            "hop_type": qa['hop_type'],
            "answer_nodes": qa['answer_nodes']
        })

        # checkpoint every 50
        if (i + 1) % 50 == 0:
            save_checkpoint({"embeddings": embeddings, "meta": meta})
            print(f"Checkpoint saved at {i+1}")

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

    print("Done!")


if __name__ == "__main__":
    main()