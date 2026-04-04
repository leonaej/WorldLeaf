import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI

sys.path.append("proposed_solution/RL_agent")
from utils import (load_graph, load_node_embeddings, load_edge_embeddings,
                   load_query_embeddings)
from environment import Environment
from policy import PolicyNetwork
from train import run_trajectory, CONFIG

# ── paths ──────────────────────────────────────────────────────────────────
BEST_MODEL_PATH  = "proposed_solution/RL_agent/checkpoints/best_model.pt"
TEST_DATA_PATH   = "proposed_solution/RL_agent/test_data.json"
NODE_TEXTS_PATH  = "baseline/node_texts.json"
EDGE_TEXTS_PATH  = "baseline/node_edge_texts.json"
RESULTS_PATH     = "proposed_solution/RL_agent/eval_llm_results.json"
CHECKPOINT_PATH  = "proposed_solution/RL_agent/eval_llm_checkpoint.json"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── build context from candidate nodes ────────────────────────────────────
def build_context(candidate_nodes, node_texts, edge_texts, node_info):
    """
    Build context string from all candidate nodes the agent visited.
    """
    context_blocks = []

    for node_id in candidate_nodes:
        # get node name
        info = node_info.get(node_id, {})
        name = info.get('common_name') or info.get('name') or node_id

        # get wikipedia text
        wiki_text = ""
        if node_id in node_texts:
            entry = node_texts[node_id]
            if entry.get('intro'):
                wiki_text = entry['intro'][:1000]  # cap at 1000 chars
            elif entry.get('sections'):
                first_section = list(entry['sections'].values())[0]
                wiki_text = first_section[:1000]

        # get edge text
        edge_text = ""
        if node_id in edge_texts:
            edge_text = edge_texts[node_id].get('text', '')

        if not wiki_text and not edge_text:
            continue

        block = f"--- Node: {name} ---\n"
        if wiki_text:
            block += f"Wikipedia: {wiki_text}\n"
        if edge_text:
            block += f"Relationships: {edge_text}\n"

        context_blocks.append(block)

    return "\n".join(context_blocks) if context_blocks else ""


# ── gpt judge ─────────────────────────────────────────────────────────────
def gpt_judge(question, context, valid_answers):
    """
    Send question + context to GPT-4o mini for evaluation.

    Returns:
        result: str — species name, TYPE1, TYPE2, or TYPE3
    """
    valid_answer_str = ", ".join(valid_answers)

    if not context:
        return "TYPE1"

    system_prompt = """You are an ecology expert evaluating whether a question about the Serengeti ecosystem can be answered from provided context.

You will be given:
1. A question about species relationships
2. Valid answers to the question
3. Context retrieved from a knowledge graph

Your task: determine if the context contains enough information to answer the question.

Respond with EXACTLY one of:
- The species name if you can find the answer in the context (must match one of the valid answers)
- TYPE1 if the context is completely irrelevant to the question
- TYPE2 if relevant species are in the context but the specific relationship information is missing
- TYPE3 if relevant species are in the context but the information is incorrect

Rules:
- Use ONLY the provided context, not your own knowledge
- Respond with just the species name or TYPE1/TYPE2/TYPE3, nothing else"""

    user_prompt = f"""Question: {question}

Valid answers: [{valid_answer_str}]

Context:
{context}

What is your evaluation?"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT error: {e}")
        return "TYPE1"


# ── load checkpoint ────────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            return json.load(f)
    return {"results": [], "done_questions": []}


def save_checkpoint(data):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(data, f)


# ── main evaluation ────────────────────────────────────────────────────────
def evaluate_llm(debug=False):

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load everything
    adjacency, node_info                   = load_graph()
    node_matrix, node_to_idx, idx_to_node  = load_node_embeddings()
    edge_matrix, edge_to_idx, _            = load_edge_embeddings()
    query_matrix, query_lookup             = load_query_embeddings()




    # load node texts
    with open(NODE_TEXTS_PATH, 'r', encoding='utf-8') as f:
        node_texts = json.load(f)
    print(f"Node texts loaded: {len(node_texts)} entries")

    with open(EDGE_TEXTS_PATH, 'r', encoding='utf-8') as f:
        edge_texts = json.load(f)
    print(f"Edge texts loaded: {len(edge_texts)} entries")

    # load test data
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)




    if debug:
        test_data = test_data[:10]
        print(f"DEBUG MODE: {len(test_data)} questions")

    print(f"Test questions: {len(test_data)}")

    # environment + model
    env = Environment(adjacency, node_info, node_matrix,
                      node_to_idx, idx_to_node, edge_matrix, edge_to_idx)

    policy = PolicyNetwork().to(device)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    policy.load_state_dict(checkpoint['model_state'])
    policy.eval()
    print(f"Loaded best model from epoch {checkpoint['epoch']} "
          f"(val Hit@1: {checkpoint['val_hit1']:.4f})")

    max_hops   = 3
    beam_width = CONFIG["beam_width"]
    top_k      = CONFIG["top_k_start"]

    # load checkpoint
    ckpt = load_checkpoint()
    results = ckpt["results"]
    done_questions = set(ckpt["done_questions"])
    print(f"Resuming from checkpoint: {len(results)} already evaluated")

    # ── evaluate each question ─────────────────────────────────────────────
    for qa in tqdm(test_data, desc="Evaluating with GPT judge"):
        question   = qa['question']
        answer_ids = {a['id'] for a in qa['answer_nodes']}
        answer_names = [a['name'] for a in qa['answer_nodes']]
        hop_type   = qa['hop_type']

        if question in done_questions:
            continue

        if question not in query_lookup:
            results.append({
                "question":     question,
                "hop_type":     hop_type,
                "answer_nodes": qa['answer_nodes'],
                "candidates":   [],
                "context":      "",
                "gpt_response": "TYPE1",
                "result":       "TYPE1",
                "hit":          False,
                "skipped":      True
            })
            done_questions.add(question)
            continue

        # run RL agent
        query_emb   = query_matrix[query_lookup[question]['index']]
        start_nodes = env.get_start_nodes(query_emb, top_k=top_k)

        all_candidates = set()
        for start_node in start_nodes:
            _, _, _, final_nodes = run_trajectory(
                start_node, query_emb, answer_ids,
                env, policy, device, max_hops, beam_width
            )
            all_candidates.update(final_nodes)

        # build context from candidate nodes
        context = build_context(
            list(all_candidates), node_texts, edge_texts, node_info)

        # gpt judge
        gpt_response = gpt_judge(question, context, answer_names)

        # classify result
        gpt_clean = gpt_response.strip().upper()
        if gpt_clean in ["TYPE1", "TYPE2", "TYPE3"]:
            result = gpt_clean
            hit    = False
        else:
            # gpt returned a species name — check if it matches valid answers
            hit    = any(
                gpt_response.lower() in name.lower() or
                name.lower() in gpt_response.lower()
                for name in answer_names
            )
            result = "HIT" if hit else "TYPE1"

        results.append({
            "question":     question,
            "hop_type":     hop_type,
            "answer_nodes": qa['answer_nodes'],
            "candidates":   list(all_candidates),
            "context_length": len(context),
            "gpt_response": gpt_response,
            "result":       result,
            "hit":          hit,
            "skipped":      False
        })
        done_questions.add(question)

        # checkpoint every 20 questions
        if len(results) % 20 == 0:
            save_checkpoint({
                "results":        results,
                "done_questions": list(done_questions)
            })

    # ── compute metrics ────────────────────────────────────────────────────
    total      = len(results)
    hits       = sum(1 for r in results if r['hit'])
    type1      = sum(1 for r in results if r['result'] == 'TYPE1')
    type2      = sum(1 for r in results if r['result'] == 'TYPE2')
    type3      = sum(1 for r in results if r['result'] == 'TYPE3')

    single     = [r for r in results if r['hop_type'] == 'single']
    multi      = [r for r in results if r['hop_type'] == 'multi']
    single_hits = sum(1 for r in single if r['hit'])
    multi_hits  = sum(1 for r in multi  if r['hit'])

    overall_hit1    = hits / total            if total        > 0 else 0

    single_hop_hit1 = single_hits / len(single) if len(single) > 0 else 0
    multi_hop_hit1  = multi_hits  / len(multi)  if len(multi)  > 0 else 0

    # ── print results ──────────────────────────────────────────────────────
    print(f"\n── RL Agent LLM Evaluation Results ─────────────────────────")
    print(f"Total questions:          {total}")
    print(f"HIT:                   {hits}  ({overall_hit1:.1%})")
    print(f"TYPE1 (traversal miss): {type1}  ({type1/total:.1%})")
    print(f"TYPE2 (data gap):      {type2}  ({type2/total:.1%})")
    print(f"TYPE3 (wrong info):     {type3}  ({type3/total:.1%})")
    print(f"\n── By Hop Type ──────────────────────────────────────────────")
    print(f"Single-hop Hit@1:         {single_hop_hit1:.4f}  "
          f"({single_hits}/{len(single)})")
    print(f"Multi-hop Hit@1:          {multi_hop_hit1:.4f}  "
          f"({multi_hits}/{len(multi)})")
    print(f"\n── Baseline Comparison ──────────────────────────────────────")
    print(f"{'Metric':<25} {'RAG':>10} {'RL Agent':>10} {'Delta':>10}")
    print(f"{'─'*55}")
    print(f"{'Overall Hit@1':<25} {0.4912:>10.4f} "
          f"{overall_hit1:>10.4f} {overall_hit1-0.4912:>+10.4f}")
    print(f"{'Single-hop Hit@1':<25} {0.8960:>10.4f} "
          f"{single_hop_hit1:>10.4f} {single_hop_hit1-0.8960:>+10.4f}")
    print(f"{'Multi-hop Hit@1':<25} {0.3816:>10.4f} "
          f"{multi_hop_hit1:>10.4f} {multi_hop_hit1-0.3816:>+10.4f}")
    print(f"{'TYPE1 rate':<25} {0.1050:>10.4f} "
          f"{type1/total:>10.4f} {type1/total-0.1050:>+10.4f}")
    print(f"{'TYPE2 rate':<25} {0.4030:>10.4f} "
          f"{type2/total:>10.4f} {type2/total-0.4030:>+10.4f}")
    print(f"{'─'*55}")

    # ── save final results ─────────────────────────────────────────────────
    output = {
        "model_checkpoint":   BEST_MODEL_PATH,
        "best_model_epoch":   checkpoint['epoch'],
        "val_hit1":           checkpoint['val_hit1'],
        "metrics": {
            "overall_hit1":    overall_hit1,
            "single_hop_hit1": single_hop_hit1,
            "multi_hop_hit1":  multi_hop_hit1,
            "type1_rate":      type1 / total,
            "type2_rate":      type2 / total,
            "type3_rate":      type3 / total,
            "total":           total,
            "hits":            hits,
            "type1":           type1,
            "type2":           type2,
            "type3":           type3
        },
        "baseline_comparison": {
            "rag_overall":    0.4912,
            "rag_single":     0.8960,
            "rag_multi":      0.3816,
            "rag_type1":      0.1050,
            "rag_type2":      0.4030,
            "delta_overall":  overall_hit1 - 0.4912,
            "delta_single":   single_hop_hit1 - 0.8960,
            "delta_multi":    multi_hop_hit1 - 0.3816,
        },
        "per_question_results": results
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # cleanup checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Run on 10 questions only")
    args = parser.parse_args()
    evaluate_llm(debug=args.debug)