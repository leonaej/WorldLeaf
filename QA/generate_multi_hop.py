import json
import os
import time
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
CHAINS_PATH  = "qa/multi_hop_chains.json"
OUTPUT_PATH  = "qa/multi_hop_qa.json"
MODEL        = "gpt-4o-mini"
BATCH_SIZE   = 20
CHECKPOINT_EVERY = 5  # save every 5 batches (250 questions)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an ecology expert specializing in African Savanna ecosystems.

Given a chain of ecological relationships between species, generate a single natural sounding question in English where:
- The answer is ALWAYS the LAST species/taxon in the chain
- The question reflects the full chain of reasoning
- The question sounds like something a student or researcher would ask
- Use common names where provided

Relation meanings:
- eats: species A consumes species B as food
- preys_on: species A actively hunts species B
- scavenges_from: species A scavenges food from kills made by species B
- parasitizes: species A is a parasite of species B
- pollinates: species A pollinates species B
- migrates_with: species A migrates alongside species B
- disperses_seeds_of: species A disperses seeds of species B
- symbiotic_with: species A has a mutualistic relationship with species B
- parent_taxon: species A belongs to taxon B

Examples:
Chain: Lion → preys_on → Zebra → eats → Rhodes Grass
Question: What do the animals that lions prey on eat?

Chain: Cheetah → preys_on → Thomson's Gazelle → parent_taxon → Gazella
Question: What is the parent taxon of the animals that cheetahs prey on?

Chain: Black-backed Jackal → scavenges_from → Lion → preys_on → Zebra → eats → Grass
Question: What do the animals preyed on by the species that black-backed jackals scavenge from eat?

CRITICAL RULES:
- Return ONLY a JSON array of questions in the same order as the input chains
- The array MUST have exactly the same number of elements as the number of input chains
- If a chain is confusing or unclear, still return a best attempt question for it
- No explanation, no markdown, just the JSON array
Example output: ["Question 1?", "Question 2?", "Question 3?"]"""

# ── Generate questions for a batch of chains ──────────────────────────────────
def generate_questions(chains: list) -> list:
    # Build the user message with numbered chains
    chain_lines = []
    for i, chain in enumerate(chains):
        chain_lines.append(f"{i+1}. Chain: {chain['chain_text']}")

    user_message = "Generate a question for each of these chains:\n\n" + "\n".join(chain_lines)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.3  # low temperature for consistent output
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown if present
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        questions = json.loads(raw)

        if len(questions) != len(chains):
            print(f"  [WARNING] Expected {len(chains)} questions, got {len(questions)}")
            # Pad with empty strings if mismatch
            while len(questions) < len(chains):
                questions.append("")

        return questions

    except Exception as e:
        print(f"  [ERROR] Batch failed: {e}")
        return [""] * len(chains)

# ── Save checkpoint ───────────────────────────────────────────────────────────
def save_checkpoint(qa_pairs: list):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"  [CHECKPOINT] Saved {len(qa_pairs)} QA pairs to {OUTPUT_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load chains
    with open(CHAINS_PATH, "r", encoding="utf-8") as f:
        all_chains = json.load(f)

    print(f"Loaded {len(all_chains)} chains from {CHAINS_PATH}")

    # Resume support
    qa_pairs = []
    already_done = 0
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        already_done = len(qa_pairs)
        print(f"Resuming — {already_done} questions already generated, skipping those.")

    # Skip already done chains
    remaining_chains = all_chains[already_done:]
    print(f"Remaining chains to process: {len(remaining_chains)}\n")

    # ── Process in batches ────────────────────────────────────────────────────
    total_batches = (len(remaining_chains) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        batch = remaining_chains[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
        print(f"[Batch {batch_idx+1}/{total_batches}] Generating {len(batch)} questions...")

        questions = generate_questions(batch)

        for chain, question in zip(batch, questions):
            qa_pairs.append({
                "question":        question,
                "answer_node_id":  chain["answer_node_id"],
                "answer_name":     chain["answer_name"],
                "chain":           chain["nodes"],
                "chain_text":      chain["chain_text"],
                "node_names":      chain["node_names"],
                "relations":       chain["relations"],
                "start_node_id":   chain["start_node_id"],
                "start_node_name": chain["start_node_name"],
                "hop":             chain["hop"]
            })

        # Checkpoint every N batches
        if (batch_idx + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(qa_pairs)

        time.sleep(0.5)

    # Final save
    save_checkpoint(qa_pairs)

    # ── Stats ─────────────────────────────────────────────────────────────────
    two_hop   = sum(1 for q in qa_pairs if q["hop"] == 2)
    three_hop = sum(1 for q in qa_pairs if q["hop"] == 3)
    empty     = sum(1 for q in qa_pairs if not q["question"])

    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"Total QA pairs:       {len(qa_pairs)}")
    print(f"  2-hop questions:    {two_hop}")
    print(f"  3-hop questions:    {three_hop}")
    print(f"  Empty/failed:       {empty}")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"────────────────────────────────────────────────────")

    # Print examples
    print(f"\nExample QA pairs:")
    for qa in qa_pairs[:5]:
        print(f"  [{qa['hop']}-hop] Q: {qa['question']}")
        print(f"           A: {qa['answer_name']}")
        print(f"           Chain: {qa['chain_text']}\n")

if __name__ == "__main__":
    main()