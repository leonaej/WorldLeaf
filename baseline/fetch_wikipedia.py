import pandas as pd
import requests
import json
import os
import time
import re

# ── Config ──────────────────────────────────────────────────────────────────
NODES_PATH  = "data/processed/nodes.csv"
OUTPUT_PATH = "baseline/node_texts.json"
CHECKPOINT_EVERY = 10

HEADERS = {
    "User-Agent": "WorldLeaf/1.0 (thesis project; contact@example.com) python-requests"
}

# Keywords to match against section titles (lowercase)
RELEVANT_KEYWORDS = [
    "diet", "food", "feed", "ecolog", "behav", "habitat",
    "prey", "predat", "distribut", "biology", "description",
    "taxonomy", "conservation", "reproduction", "lifestyle"
]

# ── Get Wikipedia title from QID ─────────────────────────────────────────────
def get_wiki_title(qid: str) -> str:
    try:
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "sitelinks",
            "sitefilter": "enwiki",
            "format": "json"
        }
        r = requests.get(
            "https://www.wikidata.org/w/api.php",
            params=params,
            headers=HEADERS,
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        sitelinks = data.get("entities", {}).get(qid, {}).get("sitelinks", {})
        if "enwiki" not in sitelinks:
            return ""
        return sitelinks["enwiki"]["title"]
    except Exception as e:
        print(f"  [ERROR getting title] {qid}: {e}")
        return ""

# ── Clean wikitext markup ─────────────────────────────────────────────────────
def clean_wikitext(raw: str) -> str:
    clean = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', raw)  # [[link|text]] -> text
    clean = re.sub(r'\{\{[^}]+\}\}', '', clean)                    # remove {{templates}}
    clean = re.sub(r"'{2,3}", '', clean)                            # remove bold/italic
    clean = re.sub(r'<[^>]+>', '', clean)                           # remove HTML tags
    clean = re.sub(r'==+[^=]+=+', '', clean)                        # remove section headers
    clean = re.sub(r'\n{3,}', '\n\n', clean).strip()
    return clean

# ── Fetch full intro + relevant sections ─────────────────────────────────────
def get_wikipedia_content(wiki_title: str) -> str:
    try:
        # Step 1: Get full intro section
        intro_params = {
            "action": "query",
            "titles": wiki_title,
            "prop": "extracts",
            "exsectionformat": "plain",
            "exintro": True,
            "explaintext": True,
            "format": "json"
        }
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=intro_params,
            headers=HEADERS,
            timeout=15
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        intro_text = ""
        for page in pages.values():
            intro_text = page.get("extract", "").strip()

        # Step 2: Get list of all sections
        sections_params = {
            "action": "parse",
            "page": wiki_title,
            "prop": "sections",
            "format": "json"
        }
        r2 = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=sections_params,
            headers=HEADERS,
            timeout=15
        )
        r2.raise_for_status()
        data = r2.json()

        if "error" in data:
            # Article exists but parse failed, return just intro
            return f"=== Introduction ===\n{intro_text}" if intro_text else ""

        sections = data.get("parse", {}).get("sections", [])

        # Step 3: Filter relevant sections by keyword
        relevant_sections = []
        for sec in sections:
            title_lower = sec.get("line", "").lower()
            if any(kw in title_lower for kw in RELEVANT_KEYWORDS):
                relevant_sections.append({
                    "index": sec["index"],
                    "title": sec["line"]
                })

        # Step 4: Fetch content of each relevant section
        section_texts = []
        for sec in relevant_sections:
            sec_params = {
                "action": "parse",
                "page": wiki_title,
                "section": sec["index"],
                "prop": "wikitext",
                "format": "json"
            }
            r3 = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=sec_params,
                headers=HEADERS,
                timeout=15
            )
            r3.raise_for_status()
            raw = r3.json().get("parse", {}).get("wikitext", {}).get("*", "")
            clean = clean_wikitext(raw)
            if clean:
                section_texts.append(f"=== {sec['title']} ===\n{clean}")
            time.sleep(0.2)

        # Step 5: Combine
        all_parts = []
        if intro_text:
            all_parts.append(f"=== Introduction ===\n{intro_text}")
        all_parts.extend(section_texts)

        return "\n\n".join(all_parts)

    except Exception as e:
        print(f"  [ERROR fetching content] {wiki_title}: {e}")
        return ""

# ── Checkpoint save ───────────────────────────────────────────────────────────
def save_checkpoint(data: dict):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [CHECKPOINT] Saved {len(data)} entries to {OUTPUT_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(NODES_PATH)
    total = len(df)
    print(f"Loaded {total} nodes from {NODES_PATH}")

    # Resume support
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            node_texts = json.load(f)
        print(f"Resuming — {len(node_texts)} nodes already fetched, skipping those.")
    else:
        node_texts = {}

    fetched_count = 0
    for i, row in df.iterrows():
        node_id = str(row["node_id"])
        name    = str(row["name"])

        if node_id in node_texts:
            continue

        print(f"[{i+1}/{total}] Fetching: {name} ({node_id})")

        wiki_title = get_wiki_title(node_id)
        if not wiki_title:
            print(f"  [NO WIKI] {name} — no English Wikipedia article")
            node_texts[node_id] = {
                "node_id":    node_id,
                "name":       name,
                "wiki_title": "",
                "text":       "",
                "has_text":   False
            }
        else:
            time.sleep(0.3)
            content = get_wikipedia_content(wiki_title)
            node_texts[node_id] = {
                "node_id":    node_id,
                "name":       name,
                "wiki_title": wiki_title,
                "text":       content,
                "has_text":   bool(content)
            }
            if content:
                word_count = len(content.split())
                print(f"  [OK] {wiki_title} — {word_count} words")
            else:
                print(f"  [EMPTY] {wiki_title} — no content extracted")

        fetched_count += 1
        time.sleep(0.3)

        if fetched_count % CHECKPOINT_EVERY == 0:
            save_checkpoint(node_texts)

    # Final save
    save_checkpoint(node_texts)

    # ── Stats ─────────────────────────────────────────────────────────────────
    total_with_text  = sum(1 for v in node_texts.values() if v["has_text"])
    total_without    = total - total_with_text
    all_text         = " ".join(v["text"] for v in node_texts.values() if v["text"])
    total_words      = len(all_text.split())
    estimated_tokens = int(total_words * 1.33)

    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"Total nodes:          {total}")
    print(f"Nodes with text:      {total_with_text}")
    print(f"Nodes without text:   {total_without}")
    print(f"Total words:          {total_words:,}")
    print(f"Estimated tokens:     {estimated_tokens:,}")
    print(f"────────────────────────────────────────────────────")
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()