import pandas as pd
import json
import os
import random

# ── Config ──────────────────────────────────────────────────────────────────
EDGES_PATH  = "data/processed/edges.csv"
NODES_PATH  = "data/processed/nodes.csv"
OUTPUT_PATH = "qa/multi_hop_chains.json"

ECOLOGICAL_RELATIONS = {
    "eats", "preys_on", "parasitizes", "pollinates",
    "scavenges_from", "migrates_with", "disperses_seeds_of", "symbiotic_with"
}

# ── Chain validity rules ──────────────────────────────────────────────────────
def is_valid_chain(relations: list, nodes: list) -> bool:
    # No repeated nodes (no circular chains)
    if len(set(nodes)) != len(nodes):
        return False

    # Max 1 parent_taxon per chain
    if relations.count("parent_taxon") > 1:
        return False

    # parent_taxon cannot be first relation
    if relations[0] == "parent_taxon":
        return False

    # Must have at least one ecological relation
    if not any(r in ECOLOGICAL_RELATIONS for r in relations):
        return False

    # No same relation twice in a row
    for i in range(len(relations) - 1):
        if relations[i] == relations[i+1]:
            return False

    return True

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    edges_df = pd.read_csv(EDGES_PATH)
    nodes_df = pd.read_csv(NODES_PATH)

    # Build node lookup: node_id -> common_name or scientific name
    node_lookup = {}
    for _, row in nodes_df.iterrows():
        node_id = str(row["node_id"])
        name = row["common_name"] if pd.notna(row["common_name"]) and str(row["common_name"]).strip() not in ("", "nan") else row["name"]
        node_lookup[node_id] = str(name).strip()

    # Build adjacency list: node_id -> list of (relation, object_id)
    adjacency = {}
    for _, edge in edges_df.iterrows():
        subj = str(edge["subject_id"]).strip()
        obj  = str(edge["object_id"]).strip()
        rel  = str(edge["relation"]).strip()

        if subj not in adjacency:
            adjacency[subj] = []
        adjacency[subj].append({
            "relation":  rel,
            "object_id": obj,
        })

    print(f"Built adjacency list for {len(adjacency)} nodes")

    # ── Build 2-hop chains ────────────────────────────────────────────────────
    two_hop_chains = []
    for node_a, edges_a in adjacency.items():
        name_a = node_lookup.get(node_a, node_a)
        for edge_ab in edges_a:
            node_b = edge_ab["object_id"]
            rel_ab = edge_ab["relation"]
            name_b = node_lookup.get(node_b, node_b)

            if node_b not in adjacency:
                continue

            for edge_bc in adjacency[node_b]:
                node_c = edge_bc["object_id"]
                rel_bc = edge_bc["relation"]
                name_c = node_lookup.get(node_c, node_c)

                relations = [rel_ab, rel_bc]
                nodes     = [node_a, node_b, node_c]

                if not is_valid_chain(relations, nodes):
                    continue

                two_hop_chains.append({
                    "chain_text": f"{name_a} → {rel_ab} → {name_b} → {rel_bc} → {name_c}",
                    "nodes":      [node_a, node_b, node_c],
                    "node_names": [name_a, name_b, name_c],
                    "relations":  relations,
                    "answer_node_id":  node_c,
                    "answer_name":     name_c,
                    "start_node_id":   node_a,
                    "start_node_name": name_a,
                    "hop": 2
                })

    print(f"Found {len(two_hop_chains)} valid 2-hop chains")

    # ── Build 3-hop chains ────────────────────────────────────────────────────
    three_hop_chains = []
    for node_a, edges_a in adjacency.items():
        name_a = node_lookup.get(node_a, node_a)
        for edge_ab in edges_a:
            node_b = edge_ab["object_id"]
            rel_ab = edge_ab["relation"]
            name_b = node_lookup.get(node_b, node_b)

            if node_b not in adjacency:
                continue

            for edge_bc in adjacency[node_b]:
                node_c = edge_bc["object_id"]
                rel_bc = edge_bc["relation"]
                name_c = node_lookup.get(node_c, node_c)

                if node_c not in adjacency:
                    continue

                for edge_cd in adjacency[node_c]:
                    node_d = edge_cd["object_id"]
                    rel_cd = edge_cd["relation"]
                    name_d = node_lookup.get(node_d, node_d)

                    relations = [rel_ab, rel_bc, rel_cd]
                    nodes     = [node_a, node_b, node_c, node_d]

                    if not is_valid_chain(relations, nodes):
                        continue

                    three_hop_chains.append({
                        "chain_text": f"{name_a} → {rel_ab} → {name_b} → {rel_bc} → {name_c} → {rel_cd} → {name_d}",
                        "nodes":      [node_a, node_b, node_c, node_d],
                        "node_names": [name_a, name_b, name_c, name_d],
                        "relations":  relations,
                        "answer_node_id":  node_d,
                        "answer_name":     name_d,
                        "start_node_id":   node_a,
                        "start_node_name": name_a,
                        "hop": 3
                    })

    print(f"Found {len(three_hop_chains)} valid 3-hop chains")

    # ── Combine all chains ────────────────────────────────────────────────────
    all_chains = two_hop_chains + three_hop_chains
    random.seed(42)
    random.shuffle(all_chains)

    # Save
    os.makedirs("qa", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chains, f, indent=2, ensure_ascii=False)

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"Total 2-hop chains:   {len(two_hop_chains)}")
    print(f"Total 3-hop chains:   {len(three_hop_chains)}")
    print(f"Total chains saved:   {len(all_chains)}")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"────────────────────────────────────────────────────")

    print(f"\nExample chains:")
    for chain in all_chains[:5]:
        print(f"  [{chain['hop']}-hop] {chain['chain_text']}")

if __name__ == "__main__":
    main()