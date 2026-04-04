import os
import sys
import json
import numpy as np
import torch

sys.path.append("proposed_solution/RL_agent")
from utils import (load_graph, load_node_embeddings, load_edge_embeddings,
                   load_query_embeddings)
from environment import Environment
from policy import PolicyNetwork
from train import CONFIG
from openai import OpenAI

# ── paths ──────────────────────────────────────────────────────────────────
BEST_MODEL_PATH = "proposed_solution/RL_agent/checkpoints/best_model.pt"
TEST_DATA_PATH  = "proposed_solution/RL_agent/test_data.json"
NODE_TEXTS_PATH = "baseline/node_texts.json"
EDGE_TEXTS_PATH = "baseline/node_edge_texts.json"
DEMO_OUTPUT     = "proposed_solution/RL_agent/demo_results.json"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ── verbose trajectory ─────────────────────────────────────────────────────
def run_trajectory_verbose(start_node, query_emb, answer_node_ids,
                           env, policy, device, max_hops, beam_width,
                           node_info):
    """
    Same as run_trajectory but records the full path for display.
    Returns list of completed paths, each with step-by-step trace.
    """

    def get_name(node_id):
        info = node_info.get(node_id, {})
        return info.get('common_name') or info.get('name') or node_id

    # beam: (node_id, hops, path_trace, log_probs_list)
    beam = [(start_node, 0, [{"node": start_node,
                               "name": get_name(start_node),
                               "action": "START",
                               "relation": None}], [])]
    completed = []

    for hop in range(max_hops):
        if not beam:
            break

        next_beam = []

        for (current_node, hops, path_trace, log_probs_list) in beam:

            # dead end
            if env.is_dead_end(current_node):
                completed.append({
                    "final_node":  current_node,
                    "final_name":  get_name(current_node),
                    "hops":        hops,
                    "path":        path_trace,
                    "stopped_by":  "dead_end",
                    "is_answer":   current_node in answer_node_ids
                })
                continue

            # get actions
            actions = env.get_actions(current_node, query_emb)

            # score all actions
            query_tensor = torch.tensor(
                query_emb, dtype=torch.float32).to(device)
            action_embeddings = torch.tensor(
                np.array([a['embedding'] for a in actions]),
                dtype=torch.float32).to(device)
            cosine_sims = torch.tensor(
                np.array([a['cosine_sim'] for a in actions]),
                dtype=torch.float32).to(device)

            with torch.no_grad():
                probs, _ = policy.forward(
                    query_tensor, action_embeddings, cosine_sims)

            probs_np = probs.detach().cpu().numpy()
            top_indices = np.argsort(probs_np)[::-1][:beam_width]

            for idx in top_indices:
                action = actions[idx]
                prob = float(probs_np[idx])

                if action["type"] == "stop":
                    completed.append({
                        "final_node":  current_node,
                        "final_name":  get_name(current_node),
                        "hops":        hops,
                        "path":        path_trace + [{
                            "node":     current_node,
                            "name":     get_name(current_node),
                            "action":   "STOP",
                            "relation": None,
                            "prob":     prob
                        }],
                        "stopped_by":  "agent",
                        "is_answer":   current_node in answer_node_ids
                    })
                else:
                    neighbor = action["neighbor_id"]
                    next_beam.append((
                        neighbor,
                        hops + 1,
                        path_trace + [{
                            "node":     neighbor,
                            "name":     get_name(neighbor),
                            "action":   "FOLLOW",
                            "relation": action["relation"],
                            "prob":     prob
                        }],
                        log_probs_list
                    ))

        # force stop at max hops
        if hop == max_hops - 1:
            for (current_node, hops, path_trace, _) in next_beam:
                completed.append({
                    "final_node":  current_node,
                    "final_name":  get_name(current_node),
                    "hops":        hops,
                    "path":        path_trace,
                    "stopped_by":  "max_hops",
                    "is_answer":   current_node in answer_node_ids
                })
            next_beam = []

        beam = next_beam

    return completed


# ── build context ──────────────────────────────────────────────────────────
def build_context(candidate_nodes, node_texts, edge_texts, node_info):
    context_blocks = []
    for node_id in candidate_nodes:
        info = node_info.get(node_id, {})
        name = info.get('common_name') or info.get('name') or node_id

        wiki_text = ""
        if node_id in node_texts:
            entry = node_texts[node_id]
            if entry.get('intro'):
                wiki_text = entry['intro'][:1000]
            elif entry.get('sections'):
                first_section = list(entry['sections'].values())[0]
                wiki_text = first_section[:1000]

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
    valid_answer_str = ", ".join(valid_answers)
    if not context:
        return "TYPE1"

    system_prompt = """You are an ecology expert evaluating whether a question about the Serengeti ecosystem can be answered from provided context.

Respond with EXACTLY one of:
- The species name if you can find the answer in the context
- TYPE1 if the context is completely irrelevant
- TYPE2 if relevant species present but missing relationship info
- TYPE3 if relevant species present but wrong info

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
            temperature=0.0,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT error: {e}")
        return "TYPE1"


# ── format path for display ────────────────────────────────────────────────
def format_path(path):
    parts = []
    for step in path:
        if step["action"] == "START":
            parts.append(step["name"])
        elif step["action"] == "FOLLOW":
            parts.append(f"--[{step['relation']}]--> {step['name']}")
        elif step["action"] == "STOP":
            parts.append("--> STOP")
    return " ".join(parts)


# ── main demo ──────────────────────────────────────────────────────────────
def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # load everything
    adjacency, node_info                   = load_graph()
    node_matrix, node_to_idx, idx_to_node  = load_node_embeddings()
    edge_matrix, edge_to_idx, _            = load_edge_embeddings()
    query_matrix, query_lookup             = load_query_embeddings()

    with open(NODE_TEXTS_PATH, 'r', encoding='utf-8') as f:
        node_texts = json.load(f)
    with open(EDGE_TEXTS_PATH, 'r', encoding='utf-8') as f:
        edge_texts = json.load(f)
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    
    # pick 5 single-hop and 5 multi-hop
    single = [q for q in test_data if q['hop_type'] == 'single'][:5]
    multi  = [q for q in test_data if q['hop_type'] == 'multi'][:5]

    

    demo_questions = single + multi
    print(f"Running demo on {len(demo_questions)} questions "
          f"(5 single-hop + 5 multi-hop)\n")

    # load model
    env = Environment(adjacency, node_info, node_matrix,
                      node_to_idx, idx_to_node, edge_matrix, edge_to_idx)
    policy = PolicyNetwork().to(device)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    policy.load_state_dict(checkpoint['model_state'])
    policy.eval()
    print(f"Loaded best model from epoch {checkpoint['epoch']} "
          f"(val Hit@1: {checkpoint['val_hit1']:.4f})\n")

    max_hops   = 3
    beam_width = CONFIG["beam_width"]
    top_k      = CONFIG["top_k_start"]

    demo_results = []
    hits = 0

    print("=" * 70)

    for i, qa in enumerate(demo_questions):
        question     = qa['question']
        answer_ids   = {a['id'] for a in qa['answer_nodes']}
        answer_names = [a['name'] for a in qa['answer_nodes']]
        hop_type     = qa['hop_type']

        print(f"\nQuestion {i+1} [{hop_type}]:")
        print(f"   {question}")
        print(f"   Valid answers: {answer_names}")

        if question not in query_lookup:
            print(f"   WARNING: Question not in query lookup, skipping")
            continue

        query_emb   = query_matrix[query_lookup[question]['index']]
        start_nodes = env.get_start_nodes(query_emb, top_k=top_k)

        # get start node names
        start_names = []
        for n in start_nodes:
            info = node_info.get(n, {})
            name = info.get('common_name') or info.get('name') or n
            start_names.append(name)
        print(f"\n   Start nodes: {start_names}")

        # run verbose trajectory for each start node
        all_paths      = []
        all_candidates = set()

        for start_node in start_nodes:
            paths = run_trajectory_verbose(
                start_node, query_emb, answer_ids,
                env, policy, device, max_hops, beam_width, node_info
            )
            all_paths.extend(paths)
            for p in paths:
                all_candidates.add(p['final_node'])

        # print paths
        print(f"\n   Agent paths:")
        for p in all_paths[:6]:  # show max 6 paths
            path_str      = format_path(p['path'])
            answer_marker = "[HIT]" if p['is_answer'] else "[MISS]"
            stop_reason   = p['stopped_by']
            print(f"      {answer_marker} [{stop_reason}] {path_str}")

        # build context + gpt judge
        context      = build_context(
            list(all_candidates), node_texts, edge_texts, node_info)
        gpt_response = gpt_judge(question, context, answer_names)

        # classify
        gpt_clean = gpt_response.strip().upper()
        if gpt_clean in ["TYPE1", "TYPE2", "TYPE3"]:
            result = gpt_clean
            hit    = False
        else:
            hit = any(
                gpt_response.lower() in name.lower() or
                name.lower() in gpt_response.lower()
                for name in answer_names
            )
            result = "HIT" if hit else "TYPE1"

        hits += int(hit)

        print(f"\n   GPT Judge response: '{gpt_response}'")
        print(f"   Result: {'HIT' if hit else result}")
        print("=" * 70)

        demo_results.append({
            "question":     question,
            "hop_type":     hop_type,
            "answer_nodes": answer_names,
            "start_nodes":  start_names,
            "paths":        all_paths,
            "gpt_response": gpt_response,
            "result":       result,
            "hit":          hit
        })

    # summary
    print(f"\n── Demo Summary ─────────────────────────────────────────────")
    print(f"Total:      {len(demo_results)} questions")
    print(f"Hits:       {hits} ({hits/len(demo_results):.1%})")
    print(f"Misses:     {len(demo_results)-hits} "
          f"({(len(demo_results)-hits)/len(demo_results):.1%})")

    single_hits = sum(1 for r in demo_results
                      if r['hit'] and r['hop_type'] == 'single')
    multi_hits  = sum(1 for r in demo_results
                      if r['hit'] and r['hop_type'] == 'multi')
    print(f"   Single-hop: {single_hits}/5")
    print(f"   Multi-hop:  {multi_hits}/5")

    # save
    with open(DEMO_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(demo_results, f, indent=2, ensure_ascii=False)
    print(f"\nDemo results saved to {DEMO_OUTPUT}")


if __name__ == "__main__":
    run_demo()




"""


    # pick 5 single-hop and 5 multi-hop
    single = [q for q in test_data if q['hop_type'] == 'single'][:5]
    multi  = [q for q in test_data if q['hop_type'] == 'multi'][:5]

    

    import random
    single_all = [q for q in test_data if q['hop_type'] == 'single']
    multi_all  = [q for q in test_data if q['hop_type'] == 'multi']
    single = random.sample(single_all, min(5, len(single_all)))
    multi  = random.sample(multi_all,  min(5, len(multi_all)))
"""