import pandas as pd
import json
import os

# ── Config ──────────────────────────────────────────────────────────────────
EDGES_PATH  = "data/processed/edges.csv"
NODES_PATH  = "data/processed/nodes.csv"
OUTPUT_PATH = "qa/single_hop_qa.json"

# ── Question templates (forward and backward) ─────────────────────────────────
# Note: parent_taxon only has forward direction (backward is unnatural)
TEMPLATES = {
    "eats": {
        "forward":  "What does {subject} eat?",
        "backward": "What is {object} eaten by?"
    },
    "preys_on": {
        "forward":  "What does {subject} prey on?",
        "backward": "What is {object} preyed on by?"
    },
    "parasitizes": {
        "forward":  "What does {subject} parasitize?",
        "backward": "What is {object} parasitized by?"
    },
    "pollinates": {
        "forward":  "What does {subject} pollinate?",
        "backward": "What pollinates {object}?"
    },
    "parent_taxon": {
        "forward":  "What is the parent taxon of {subject}?",
        "backward": None   # dropped — unnatural question
    },
    "scavenges_from": {
        "forward":  "What does {subject} scavenge from?",
        "backward": "What scavenges from {object}?"
    },
    "migrates_with": {
        "forward":  "Who does {subject} migrate with?",
        "backward": "Who does {object} migrate with?"
    },
    "disperses_seeds_of": {
        "forward":  "Which plants does {subject} disperse seeds for?",
        "backward": "Which species disperses seeds for {object}?"
    },
    "symbiotic_with": {
        "forward":  "What is {subject} in a symbiotic relationship with?",
        "backward": "What is {object} in a symbiotic relationship with?"
    }
}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    edges_df = pd.read_csv(EDGES_PATH)
    nodes_df = pd.read_csv(NODES_PATH)

    # Build node lookup: node_id -> common_name or scientific name
    node_lookup = {}
    for _, row in nodes_df.iterrows():
        node_id = str(row["node_id"])
        name = row["common_name"] if pd.notna(row["common_name"]) and str(row["common_name"]).strip() != "" else row["name"]
        node_lookup[node_id] = str(name).strip()

    print(f"Loaded {len(edges_df)} edges and {len(nodes_df)} nodes")

    qa_pairs = []
    skipped  = 0

    for _, edge in edges_df.iterrows():
        relation  = str(edge["relation"]).strip()
        subj_id   = str(edge["subject_id"]).strip()
        obj_id    = str(edge["object_id"]).strip()
        subj_name = str(edge["subject_label"]).strip()
        obj_name  = str(edge["object_label"]).strip()

        # Skip if relation not in our templates
        if relation not in TEMPLATES:
            skipped += 1
            continue

        # ── Forward question ──────────────────────────────────────────────────
        forward_q = TEMPLATES[relation]["forward"].format(
            subject=subj_name,
            object=obj_name
        )
        qa_pairs.append({
            "question":       forward_q,
            "answer_node_id": obj_id,
            "answer_name":    obj_name,
            "relation":       relation,
            "direction":      "forward",
            "subject_id":     subj_id,
            "subject_name":   subj_name,
            "object_id":      obj_id,
            "object_name":    obj_name,
            "hop":            1
        })

        # ── Backward question (skip if None) ──────────────────────────────────
        if TEMPLATES[relation]["backward"] is None:
            continue

        backward_q = TEMPLATES[relation]["backward"].format(
            subject=subj_name,
            object=obj_name
        )
        qa_pairs.append({
            "question":       backward_q,
            "answer_node_id": subj_id,
            "answer_name":    subj_name,
            "relation":       relation,
            "direction":      "backward",
            "subject_id":     subj_id,
            "subject_name":   subj_name,
            "object_id":      obj_id,
            "object_name":    obj_name,
            "hop":            1
        })

    # Save output
    os.makedirs("qa", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"Total edges processed:    {len(edges_df)}")
    print(f"Skipped edges:            {skipped}")
    print(f"Total QA pairs generated: {len(qa_pairs)}")
    print(f"  Forward questions:      {sum(1 for q in qa_pairs if q['direction'] == 'forward')}")
    print(f"  Backward questions:     {sum(1 for q in qa_pairs if q['direction'] == 'backward')}")
    print(f"\nBy relation type:")
    for rel in TEMPLATES:
        count = sum(1 for q in qa_pairs if q["relation"] == rel)
        print(f"  {rel:20s}: {count} questions")
    print(f"────────────────────────────────────────────────────")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()