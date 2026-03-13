import requests
import pandas as pd
import time
import os
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
INATURALIST_PLACE_ID = 69054
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_HEADERS = {
    "User-Agent": "WorldLeafBot/1.0 (research project; your@email.com)",
    "Accept": "application/sparql-results+json"
}
RAW_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ── Step 1: Get species list from iNaturalist ─────────────────────────────────
def fetch_inaturalist_species() -> pd.DataFrame:
    print("Fetching species from iNaturalist Serengeti...")
    all_species = []
    page = 1
    per_page = 200

    while True:
        url = "https://api.inaturalist.org/v1/observations/species_counts"
        params = {
            "place_id": INATURALIST_PLACE_ID,
            "per_page": per_page,
            "page": page,
            "verifiable": "true"
        }
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        results = data.get("results", [])
        if not results:
            break

        for item in results:
            taxon = item.get("taxon", {})
            all_species.append({
                "inaturalist_id": taxon.get("id"),
                "name":           taxon.get("name"),
                "common_name":    taxon.get("preferred_common_name", ""),
                "rank":           taxon.get("rank"),
                "iconic_taxon":   taxon.get("iconic_taxon_name"),
                "count":          item.get("count")
            })

        print(f"  Page {page}: got {len(results)} species (total so far: {len(all_species)})")
        if len(all_species) >= data.get("total_results", 0):
            break
        page += 1
        time.sleep(1)

    df = pd.DataFrame(all_species)
    path = os.path.join(RAW_DIR, "inaturalist_species.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} species → {path}")
    return df

# ── Step 2: Look up each species on Wikidata by name ─────────────────────────
def get_wikidata_qid(species_name: str) -> str | None:
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": species_name,
        "language": "en",
        "type": "item",
        "format": "json",
        "limit": 1
    }
    try:
        r = requests.get(url, params=params, timeout=10,
                         headers={"User-Agent": "WorldLeafBot/1.0"})
        results = r.json().get("search", [])
        if results:
            return results[0]["id"]
    except Exception as e:
        print(f"  QID lookup failed for {species_name}: {e}")
    return None

def fetch_wikidata_qids(species_df: pd.DataFrame) -> pd.DataFrame:
    print("\nLooking up Wikidata QIDs for each species...")
    qids = []
    for _, row in tqdm(species_df.iterrows(), total=len(species_df)):
        qid = get_wikidata_qid(row["name"])
        qids.append(qid)
        time.sleep(0.3)  # be polite
    species_df["wikidata_qid"] = qids
    path = os.path.join(RAW_DIR, "species_with_qids.csv")
    species_df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    print(f"  QIDs found: {species_df['wikidata_qid'].notna().sum()} / {len(species_df)}")
    return species_df

# ── Step 3: Fetch edges from Wikidata for known QIDs ─────────────────────────
def fetch_edges_for_species(species_df: pd.DataFrame) -> pd.DataFrame:
    print("\nFetching edges from Wikidata...")
    qids = species_df["wikidata_qid"].dropna().tolist()
    
    # Build VALUES clause in chunks of 50
    all_edges = []
    chunk_size = 50
    chunks = [qids[i:i+chunk_size] for i in range(0, len(qids), chunk_size)]

    relations = {
        "parent_taxon":   "P171",
        "eats":           "P1034",
        "interacts_with": "P2579",
        "main_food":      "P1566",  # fallback food source
    }

    for rel_name, prop in relations.items():
        print(f"  Fetching '{rel_name}' edges...")
        rel_edges = []
        for chunk in tqdm(chunks):
            values = " ".join([f"wd:{q}" for q in chunk])
            query = f"""
            SELECT DISTINCT ?subject ?subjectLabel ?object ?objectLabel WHERE {{
              VALUES ?subject {{ {values} }}
              ?subject wdt:{prop} ?object .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            """
            try:
                r = requests.get(
                    WIKIDATA_ENDPOINT,
                    params={"query": query, "format": "json"},
                    headers=WIKIDATA_HEADERS,
                    timeout=60
                )
                bindings = r.json()["results"]["bindings"]
                for b in bindings:
                    rel_edges.append({
                        "subject_id":    b["subject"]["value"].split("/")[-1],
                        "subject_label": b.get("subjectLabel", {}).get("value", ""),
                        "relation":      rel_name,
                        "object_id":     b["object"]["value"].split("/")[-1],
                        "object_label":  b.get("objectLabel", {}).get("value", "")
                    })
            except Exception as e:
                print(f"    Chunk failed: {e}")
            time.sleep(1)

        df = pd.DataFrame(rel_edges)
        path = os.path.join(RAW_DIR, f"edges_{rel_name}.csv")
        df.to_csv(path, index=False)
        print(f"    Saved {len(df)} edges → {path}")
        all_edges.append(df)
        time.sleep(2)

    combined = pd.concat(all_edges, ignore_index=True)
    path = os.path.join(RAW_DIR, "edges_all.csv")
    combined.to_csv(path, index=False)
    print(f"\nAll edges combined: {len(combined)} rows → {path}")
    return combined

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1 - get species from iNaturalist
    species_df = fetch_inaturalist_species()

    # Step 2 - get Wikidata QIDs
    species_df = fetch_wikidata_qids(species_df)

    # Step 3 - get edges from Wikidata
    edges = fetch_edges_for_species(species_df)

    print("\nDone!")
    print(f"  Species: {len(species_df)}")
    print(f"  Edges:   {len(edges)}")