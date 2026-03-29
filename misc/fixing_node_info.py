import os
import re
import csv
import json
import time
from openai import OpenAI

# --- CONFIG ---
INPUT_FILE = "data/processed/nodes.csv"
OUTPUT_FILE = "data/raw/taxonomy_nodes_temp_3.csv"
BATCH_SIZE = 5
MODEL = "gpt-4o-mini"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- HELPERS ---
def is_missing(row):
    return (
        row["common_name"] == "" or
        row["rank"] in ["", "unknown"] or
        row["iconic_taxon"] in ["", "unknown"]
    )

def build_prompt(names):
    entity_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(names)])

    return f"""
You are given a list of biological entities by their scientific names.

For each entity, fill in the following fields:
- common_name (if unknown, return null)
- rank (e.g., species, genus, family, etc.; if unknown, return null)
- iconic_taxon (choose from broad groups like Mammalia, Aves, Reptilia, Amphibia, Insecta, Plantae, Fungi, etc.; if unknown, return null)

Rules:
- Do NOT guess. If unsure, return null.
- Be concise and consistent.
- Use lowercase for rank.
- Use standard biological naming conventions.
- Ensure all fields are present for every entity.
- Return ONLY a valid JSON array with no extra text, no markdown, no code fences.

Return ONLY valid JSON in the following format:
[
  {{
    "name": "<scientific_name>",
    "common_name": "<common_name or null>",
    "rank": "<rank or null>",
    "iconic_taxon": "<iconic_taxon or null>"
  }}
]

Entities:
{entity_list}
"""

def call_model(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if model wraps output anyway
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            parsed = json.loads(content)

            # Normalize null values to empty string for CSV output
            for row in parsed:
                for key in ["common_name", "rank", "iconic_taxon"]:
                    if row.get(key) is None or str(row.get(key, "")).upper() == "NULL":
                        row[key] = "NULL"

            return parsed

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse failed on attempt {attempt+1}. Raw output:")
            print(content)
            print("Error:", e)
        except Exception as e:
            print(f"⚠️ API call failed on attempt {attempt+1}: {e}")

        if attempt < retries - 1:
            wait = 2 ** attempt
            print(f"Retrying in {wait}s...")
            time.sleep(wait)

    print("❌ All retries failed for this batch. Skipping.")
    return None

def load_already_processed(output_file):
    """Return a set of names already written to the output file."""
    processed = set()
    if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
        with open(output_file, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["name"])
    return processed

# --- MAIN ---
def main():
    with open(INPUT_FILE, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    missing_rows = [row for row in rows if is_missing(row)]

    print(f"Total rows: {len(rows)}")
    print(f"Missing rows: {len(missing_rows)}")

    already_processed = load_already_processed(OUTPUT_FILE)
    missing_rows = [row for row in missing_rows if row["name"] not in already_processed]
    print(f"Rows to process (after skipping already done): {len(missing_rows)}")

    if not missing_rows:
        print("Nothing left to process!")
        return

    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline='', encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out, fieldnames=["node_id", "name", "common_name", "rank", "iconic_taxon"]  # ← added node_id
        )
        if not file_exists or os.stat(OUTPUT_FILE).st_size == 0:
            writer.writeheader()

        for i in range(0, len(missing_rows), BATCH_SIZE):
            batch = missing_rows[i:i+BATCH_SIZE]
            names = [row["name"] for row in batch]

            # Build a lookup for node_id by name for this batch
            node_id_lookup = {row["name"]: row.get("node_id", "") for row in batch}  # ← added

            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(missing_rows) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"Processing batch {batch_num}/{total_batches}: {names}")

            prompt = build_prompt(names)
            data = call_model(prompt)

            if data:
                for row in data:
                    row["node_id"] = node_id_lookup.get(row["name"], "")  # ← added
                    writer.writerow(row)
                f_out.flush()
                os.fsync(f_out.fileno())
                print(f"  ✓ Wrote {len(data)} rows")
            else:
                print(f"  ✗ Skipped batch {batch_num} after all retries")

            time.sleep(1)

    print(f"\nDone. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()