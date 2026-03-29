import json
import os
import time
import numpy as np
from openai import OpenAI
import tiktoken

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_PATH       = "baseline/node_texts.json"
EMBEDDINGS_PATH  = "baseline/node_embeddings.npy"
METADATA_PATH    = "baseline/node_embeddings_meta.json"
MODEL            = "text-embedding-3-large"
MAX_TOKENS       = 8000   # model limit is 8192, keep buffer
CHECKPOINT_EVERY = 50     # save every 50 nodes

client   = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
encoder  = tiktoken.get_encoding("cl100k_base")  # encoding used by text-embedding-3

# ── Truncate text to max tokens using tiktoken ────────────────────────────────
def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Accurately truncate text to stay within token limit using tiktoken."""
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return encoder.decode(tokens)
    return text

# ── Embed a single text ───────────────────────────────────────────────────────
def embed_one(text: str) -> list:
    """Embed a single text, truncating if needed."""
    try:
        truncated = truncate_text(text)
        response  = client.embeddings.create(model=MODEL, input=[truncated])
        return response.data[0].embedding
    except Exception as e:
        print(f"  [ERROR] Embedding failed: {e}")
        return None

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

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Run fetch_wikipedia.py first!")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        node_texts = json.load(f)

    print(f"Loaded {len(node_texts)} nodes from {INPUT_PATH}")

    all_nodes      = list(node_texts.keys())
    nodes_no_text  = [nid for nid in all_nodes if not node_texts[nid]["has_text"]]
    nodes_to_embed = [nid for nid in all_nodes if node_texts[nid]["has_text"]]

    print(f"Nodes with text to embed:           {len(nodes_to_embed)}")
    print(f"Nodes without text (zero vector):   {len(nodes_no_text)}")

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

    nodes_to_embed = [nid for nid in nodes_to_embed if nid not in already_done]
    print(f"Remaining to embed: {len(nodes_to_embed)}\n")

    # ── Embed one by one ──────────────────────────────────────────────────────
    total = len(nodes_to_embed)
    for i, nid in enumerate(nodes_to_embed):
        name = node_texts[nid]["name"]
        text = node_texts[nid]["text"]

        token_count = len(encoder.encode(text))
        print(f"[{i+1}/{total}] {name} ({token_count} tokens)")

        emb = embed_one(text)
        embeddings_dict[nid] = emb if emb is not None else [0.0] * 3072

        # Checkpoint every N nodes
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(embeddings_dict, all_nodes)

        time.sleep(0.1)  # avoid rate limiting

    # ── Zero vectors for nodes without text ───────────────────────────────────
    print(f"\nAssigning zero vectors to {len(nodes_no_text)} nodes without text...")
    for nid in nodes_no_text:
        embeddings_dict[nid] = [0.0] * 3072

    # Final save
    save_checkpoint(embeddings_dict, all_nodes)

    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"Total nodes:          {len(all_nodes)}")
    print(f"Nodes embedded:       {total}")
    print(f"Nodes zero vector:    {len(nodes_no_text)}")
    print(f"Embedding matrix:     {len(embeddings_dict)} x 3072")
    print(f"Saved to:             {EMBEDDINGS_PATH}")
    print(f"Metadata saved to:    {METADATA_PATH}")
    print(f"────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()