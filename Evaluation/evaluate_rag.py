import os
import sys
import json
import time

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../rag"))

from openai import OpenAI
from retriever import load_index, retrieve

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINGLE_HOP_PATH  = os.path.join(BASE_DIR, "qa/single_hop_qa.json")
MULTI_HOP_PATH   = os.path.join(BASE_DIR, "qa/multi_hop_qa.json")
RESULTS_DIR      = os.path.join(BASE_DIR, "evaluation/results")
MODEL            = "gpt-4o-mini"
TOP_K            = 3
CHECKPOINT_EVERY = 100

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── LLM judge prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are helping evaluate a knowledge graph retrieval system for African Savanna ecology.
I will give you a question and some context about species retrieved from a database.

Your job is to:
1. Look ONLY at the provided context — do NOT use your own knowledge at all
2. Reply with ONLY one of the following — nothing else, no explanation:

   - The species or taxon name(s) if you found the answer in the context
   - "TYPE1" if the context is completely irrelevant to the question — the species or topic being asked about is not present in the context at all
   - "TYPE2" if the relevant species IS in the context but there is NO information about the specific relationship asked in the question
   - "TYPE3" if the relevant species IS in the context and there IS information about the relationship, but the information is wrong or contradicts the question

Examples:
Question: What does a lion prey on?
Context has no lion information at all → TYPE1

Question: What does a lion prey on?
Context has lion node but only describes lion appearance and habitat, no prey info → TYPE2

Question: What does a lion prey on?
Context has lion node saying lion preys on grass (incorrect) → TYPE3

Question: What does a lion prey on?
Context has lion node saying lion preys on zebra and wildebeest → zebra, wildebeest"""

# ── Build context from retrieved nodes ───────────────────────────────────────
def build_context(retrieved: list) -> str:
    context_parts = []
    for i, node in enumerate(retrieved):
        parts = [f"--- Node {i+1}: {node['name']} ---"]
        if node["wiki_text"]:
            parts.append("Wikipedia info:\n" + node["wiki_text"])
        if node["edge_text"]:
            parts.append("Relationships:\n" + node["edge_text"])
        context_parts.append("\n".join(parts))
    return "\n\n".join(context_parts)

# ── Ask LLM judge ─────────────────────────────────────────────────────────────
def ask_llm(question: str, context: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        return "TYPE1"

# ── Evaluate one question ─────────────────────────────────────────────────────
def evaluate_one(qa: dict, combined_matrix, node_order: list,
                 node_texts: dict, edge_texts: dict) -> dict:

    question        = qa["question"]
    correct_name    = qa["answer_name"]
    correct_node_id = qa["answer_node_id"]
    hop             = qa.get("hop", 1)

    # Retrieve top-3 nodes
    retrieved       = retrieve(question, combined_matrix, node_order, node_texts, edge_texts, top_k=TOP_K)
    retrieved_ids   = [r["node_id"] for r in retrieved]
    retrieved_names = [r["name"] for r in retrieved]

    # Ask LLM to classify
    context      = build_context(retrieved)
    llm_response = ask_llm(question, context)

    # Simple classification — LLM decides everything
    if llm_response in ("TYPE1", "TYPE2", "TYPE3"):
        result = llm_response
        hit    = False
    else:
        # LLM returned a name or list of names = HIT
        result = "HIT"
        hit    = True

    return {
        "question":        question,
        "expected":        correct_name,
        "expected_id":     correct_node_id,
        "retrieved_ids":   retrieved_ids,
        "retrieved_names": retrieved_names,
        "llm_response":    llm_response,
        "result":          result,
        "hit":             hit,
        "hop":             hop
    }

# ── Save results ──────────────────────────────────────────────────────────────
def save_results(results: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# ── Print stats ───────────────────────────────────────────────────────────────
def print_stats(results: list, label: str):
    total = len(results)
    hits  = sum(1 for r in results if r["result"] == "HIT")
    type1 = sum(1 for r in results if r["result"] == "TYPE1")
    type2 = sum(1 for r in results if r["result"] == "TYPE2")
    type3 = sum(1 for r in results if r["result"] == "TYPE3")

    print(f"\n── {label} ─────────────────────────────────────────")
    print(f"Total questions:              {total}")
    print(f"✅ Hit:                        {hits}  ({hits/total*100:.1f}%)")
    print(f"❌ Type 1 (retrieval miss):    {type1}  ({type1/total*100:.1f}%)")
    print(f"⚠️  Type 2 (data gap):          {type2}  ({type2/total*100:.1f}%)")
    print(f"🔴 Type 3 (wrong info):         {type3}  ({type3/total*100:.1f}%)")
    print(f"Hit@1: {hits/total*100:.2f}%")
    print(f"────────────────────────────────────────────────────")

# ── Evaluate a full QA set ────────────────────────────────────────────────────
def evaluate_set(qa_pairs: list, combined_matrix, node_order: list,
                 node_texts: dict, edge_texts: dict,
                 output_path: str, label: str):

    # Resume support
    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming {label} — {len(results)} already done")

    remaining = qa_pairs[len(results):]
    print(f"Remaining: {len(remaining)} questions\n")

    for i, qa in enumerate(remaining):
        print(f"[{len(results)+1}/{len(qa_pairs)}] {qa['question'][:80]}...")
        result = evaluate_one(qa, combined_matrix, node_order, node_texts, edge_texts)
        results.append(result)
        print(f"  → {result['result']} | Expected: {result['expected']} | LLM: {result['llm_response'][:60] if result['llm_response'] else None}")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_results(results, output_path)
            print(f"  [CHECKPOINT] Saved {len(results)} results")

        time.sleep(0.3)

    save_results(results, output_path)
    print_stats(results, label)
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading index...")
    combined_matrix, node_order, node_texts, edge_texts = load_index()

    with open(SINGLE_HOP_PATH, "r", encoding="utf-8") as f:
        single_hop_qa = json.load(f)
    with open(MULTI_HOP_PATH, "r", encoding="utf-8") as f:
        multi_hop_qa = json.load(f)

    print(f"Loaded {len(single_hop_qa)} single-hop and {len(multi_hop_qa)} multi-hop questions")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Evaluate single-hop
    print(f"\n{'='*60}")
    print("Evaluating single-hop QA...")
    single_results = evaluate_set(
        single_hop_qa, combined_matrix, node_order, node_texts, edge_texts,
        output_path=os.path.join(RESULTS_DIR, "single_hop_results.json"),
        label="Single-hop Results"
    )

    # Evaluate multi-hop
    print(f"\n{'='*60}")
    print("Evaluating multi-hop QA...")
    multi_results = evaluate_set(
        multi_hop_qa, combined_matrix, node_order, node_texts, edge_texts,
        output_path=os.path.join(RESULTS_DIR, "multi_hop_results.json"),
        label="Multi-hop Results"
    )

    # Combined score
    all_results = single_results + multi_results
    save_results(all_results, os.path.join(RESULTS_DIR, "all_results.json"))
    print_stats(all_results, "Overall Combined Results")

if __name__ == "__main__":
    main()