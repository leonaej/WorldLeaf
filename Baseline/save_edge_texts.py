import pandas as pd
import json
import os

# ── Config ──────────────────────────────────────────────────────────────────
EDGES_PATH  = "data/processed/edges.csv"
NODES_PATH  = "data/processed/nodes.csv"
OUTPUT_PATH = "baseline/node_edge_texts.json"

# ── EXACT same build_edge_text as embed_edges.py ─────────────────────────────
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

# ── EXACT same build_lookups as embed_edges.py ───────────────────────────────
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

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    edges_df, node_lookup, outgoing, incoming = build_lookups()
    print(f"Loaded {len(node_lookup)} nodes and {len(edges_df)} edges")

    node_edge_texts = {}
    for nid, name in node_lookup.items():
        text     = build_edge_text(nid, name, outgoing[nid], incoming[nid])
        has_edges = bool(outgoing[nid] or incoming[nid])

        node_edge_texts[nid] = {
            "node_id":   nid,
            "name":      name,
            "text":      text,
            "has_edges": has_edges
        }

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(node_edge_texts, f, indent=2, ensure_ascii=False)

    # ── Stats ─────────────────────────────────────────────────────────────────
    with_edges    = sum(1 for v in node_edge_texts.values() if v["has_edges"])
    without_edges = len(node_edge_texts) - with_edges

    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"Total nodes:          {len(node_edge_texts)}")
    print(f"Nodes with edges:     {with_edges}")
    print(f"Nodes without edges:  {without_edges}")
    print(f"Saved to:             {OUTPUT_PATH}")
    print(f"────────────────────────────────────────────────────")

    # Print a few examples
    print(f"\nExample edge texts:")
    for nid in list(node_lookup.keys())[:3]:
        print(f"\n--- {node_lookup[nid]} ---")
        print(node_edge_texts[nid]["text"])

if __name__ == "__main__":
    main()