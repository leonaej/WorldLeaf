import os
import sys
import json

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from retriever import load_index, retrieve

# ── Config ──────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
TOP_K = 3

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Build context from retrieved nodes ───────────────────────────────────────
def build_context(retrieved_nodes: list) -> str:
    context_parts = []
    for i, node in enumerate(retrieved_nodes):
        parts = [f"--- Node {i+1}: {node['name']} ---"]
        if node["wiki_text"]:
            parts.append("Wikipedia info:\n" + node["wiki_text"])
        if node["edge_text"]:
            parts.append("Relationships:\n" + node["edge_text"])
        context_parts.append("\n".join(parts))
    return "\n\n".join(context_parts)

# ── Generate answer ───────────────────────────────────────────────────────────
def generate_answer(query: str, context: str) -> str:
    system_prompt = """You are an ecology expert specializing in the African Savanna ecosystem.
You will be given context about species and their relationships, and a question.
Answer the question based ONLY on the provided context.
Be concise — answer in 1-2 sentences maximum.
If the context does not contain enough information to answer, say "I don't know"."""

    user_message = f"""Context:
{context}

Question: {query}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        return ""

# ── Full RAG pipeline ─────────────────────────────────────────────────────────
def rag_answer(query: str, combined_matrix, node_order: list,
               node_texts: dict, edge_texts: dict) -> dict:
    retrieved = retrieve(query, combined_matrix, node_order, node_texts, edge_texts, top_k=TOP_K)

    if not retrieved:
        return {
            "query":           query,
            "retrieved_nodes": [],
            "context":         "",
            "answer":          "Retrieval failed"
        }

    context = build_context(retrieved)
    answer  = generate_answer(query, context)

    return {
        "query":           query,
        "retrieved_nodes": [{"node_id": r["node_id"], "name": r["name"], "score": r["score"]} for r in retrieved],
        "context":         context,
        "answer":          answer
    }

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading index...")
    combined_matrix, node_order, node_texts, edge_texts = load_index()

    test_queries = [
        "What does a lion eat?",
        "What is the parent taxon of the plains zebra?",
        "What does a cheetah prey on?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        result = rag_answer(query, combined_matrix, node_order, node_texts, edge_texts)
        print(f"Retrieved: {[r['name'] for r in result['retrieved_nodes']]}")
        print(f"Answer: {result['answer']}")