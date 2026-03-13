import requests
import pandas as pd
import time
import os
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
GLOBI_API = "https://api.globalbioticinteractions.org/interaction"
RAW_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")

# Relations we want from GloBI
TARGET_RELATIONS = {
    "pollinates":   "pollinates",
    "parasiteOf":   "parasitizes",
    "preysOn":      "preys_on"
}

# ── Load our 767 species list ─────────────────────────────────────────────────
def load_serengeti_species() -> set:
    path = os.path.join(RAW_DIR, "inaturalist_species_filtered.csv")
    df = pd.read_csv(path)
    # Return set of scientific names lowercased for matching
    return set(df["name"].str.lower().tolist())

# ── Query GloBI for one species ───────────────────────────────────────────────
def query_globi(species_name: str, interaction_type: str) -> list:
    params = {
        "sourceTaxon": species_name,
        "interactionType": interaction_type,
        "type": "json"
    }
    try:
        r = requests.get(GLOBI_API, params=params, timeout=30,
                        headers={"User-Agent": "WorldLeafBot/1.0"})
        data = r.json()
        columns = data.get("columns", [])
        rows = data.get("data", [])
        
        # Find indices we need
        src_name_idx = columns.index("source_taxon_name")
        tgt_name_idx = columns.index("target_taxon_name")
        int_type_idx = columns.index("interaction_type")
        
        results = []
        for row in rows:
            results.append({
                "source_name": row[src_name_idx],
                "interaction": row[int_type_idx],
                "target_name": row[tgt_name_idx]
            })
        return results
    except Exception as e:
        print(f"  Failed for {species_name} / {interaction_type}: {e}")
        return []

# ── Main fetch ────────────────────────────────────────────────────────────────
def fetch_globi_edges() -> pd.DataFrame:
    serengeti_species = load_serengeti_species()
    print(f"Loaded {len(serengeti_species)} Serengeti species for filtering")

    species_df = pd.read_csv(os.path.join(RAW_DIR, "inaturalist_species_filtered.csv"))
    species_list = species_df["name"].tolist()

    all_edges = []

    for globi_type, our_name in TARGET_RELATIONS.items():
        print(f"\nFetching '{our_name}' edges from GloBI...")
        relation_edges = []

        for species in tqdm(species_list):
            results = query_globi(species, globi_type)
            for r in results:
                target = r["target_name"].lower() if r["target_name"] else ""
                # Only keep if target is in our Serengeti species list
                if target in serengeti_species:
                    relation_edges.append({
                        "subject_label": r["source_name"],
                        "relation":      our_name,
                        "object_label":  r["target_name"],
                        "subject_id":    "",  # no QID from GloBI
                        "object_id":     ""
                    })
            time.sleep(0.5)

        df = pd.DataFrame(relation_edges).drop_duplicates()
        path = os.path.join(RAW_DIR, f"edges_globi_{our_name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} edges → {path}")
        all_edges.append(df)

    return pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()

# ── Merge with existing edges ─────────────────────────────────────────────────
def merge_with_existing(globi_edges: pd.DataFrame):
    existing = pd.read_csv(os.path.join(RAW_DIR, "edges_all_filtered.csv"))
    print(f"\nExisting edges: {len(existing)}")
    print(f"New GloBI edges: {len(globi_edges)}")

    combined = pd.concat([existing, globi_edges], ignore_index=True).drop_duplicates()
    path = os.path.join(RAW_DIR, "edges_all_filtered.csv")
    combined.to_csv(path, index=False)
    print(f"Combined edges saved: {len(combined)} → {path}")
    print(combined["relation"].value_counts())

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    globi_edges = fetch_globi_edges()
    if not globi_edges.empty:
        merge_with_existing(globi_edges)
    else:
        print("No GloBI edges found.")