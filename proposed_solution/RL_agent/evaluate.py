import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("proposed_solution/RL_agent")
from utils import (load_graph, load_node_embeddings, load_edge_embeddings,
                   load_query_embeddings)
from environment import Environment
from policy import PolicyNetwork
from train import run_trajectory, CONFIG

# ── paths ──────────────────────────────────────────────────────────────────
BEST_MODEL_PATH = "proposed_solution/RL_agent/checkpoints/best_model.pt"
TEST_DATA_PATH  = "proposed_solution/RL_agent/test_data.json"
RESULTS_PATH    = "proposed_solution/RL_agent/eval_results.json"


def evaluate_test(debug=False):

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load everything
    adjacency, node_info                  = load_graph()
    node_matrix, node_to_idx, idx_to_node = load_node_embeddings()
    edge_matrix, edge_to_idx, _           = load_edge_embeddings()
    query_matrix, query_lookup            = load_query_embeddings()

    # load test data
    with open(TEST_DATA_PATH, 'r') as f:
        test_data = json.load(f)

    if debug:
        test_data = test_data[:10]
        print(f"DEBUG MODE: {len(test_data)} test questions")

    print(f"Test questions: {len(test_data)}")

    # environment
    env = Environment(adjacency, node_info, node_matrix,
                      node_to_idx, idx_to_node, edge_matrix, edge_to_idx)

    # load best model
    policy = PolicyNetwork().to(device)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    policy.load_state_dict(checkpoint['model_state'])
    policy.eval()
    print(f"Loaded best model from epoch {checkpoint['epoch']} "
          f"(val Hit@1: {checkpoint['val_hit1']:.4f})")

    # get max_hops from checkpoint config
    max_hops   = 3   # use final decay value for evaluation
    beam_width = CONFIG["beam_width"]
    top_k      = CONFIG["top_k_start"]

    # ── run evaluation ─────────────────────────────────────────────────────
    results = []
    hits = 0
    single_hits, single_total = 0, 0
    multi_hits,  multi_total  = 0, 0

    for qa in tqdm(test_data, desc="Evaluating test set"):
        question   = qa['question']
        answer_ids = {a['id'] for a in qa['answer_nodes']}
        hop_type   = qa['hop_type']

        if question not in query_lookup:
            results.append({
                "question":       question,
                "hop_type":       hop_type,
                "answer_nodes":   qa['answer_nodes'],
                "candidates":     [],
                "hit":            False,
                "skipped":        True
            })
            continue

        query_emb   = query_matrix[query_lookup[question]['index']]
        start_nodes = env.get_start_nodes(query_emb, top_k=top_k)

        all_candidates = set()
        for start_node in start_nodes:
            _, _, _, final_nodes = run_trajectory(
                start_node, query_emb, answer_ids,
                env, policy, device, max_hops, beam_width
            )
            all_candidates.update(final_nodes)

        hit = bool(all_candidates & answer_ids)
        hits += int(hit)

        if hop_type == "single":
            single_hits += int(hit)
            single_total += 1
        else:
            multi_hits += int(hit)
            multi_total += 1

        results.append({
            "question":     question,
            "hop_type":     hop_type,
            "answer_nodes": qa['answer_nodes'],
            "candidates":   list(all_candidates),
            "hit":          hit,
            "skipped":      False
        })

    # ── compute final metrics ──────────────────────────────────────────────
    total = len(test_data)
    overall_hit1    = hits / total         if total        > 0 else 0
    single_hop_hit1 = single_hits / single_total if single_total > 0 else 0
    multi_hop_hit1  = multi_hits  / multi_total  if multi_total  > 0 else 0

    # ── print results ──────────────────────────────────────────────────────
    print(f"\n── RL Agent Test Results ────────────────────────────────────")
    print(f"Total questions:       {total}")
    print(f"✅ Overall Hit@1:      {overall_hit1:.4f}  ({hits}/{total})")
    print(f"   Single-hop Hit@1:   {single_hop_hit1:.4f}  ({single_hits}/{single_total})")
    print(f"   Multi-hop Hit@1:    {multi_hop_hit1:.4f}  ({multi_hits}/{multi_total})")
    print(f"\n── Baseline Comparison ──────────────────────────────────────")
    print(f"RAG baseline:")
    print(f"   Overall:            0.4912")
    print(f"   Single-hop:         0.8960")
    print(f"   Multi-hop:          0.3816")
    print(f"\nRL Agent:")
    print(f"   Overall:            {overall_hit1:.4f}")
    print(f"   Single-hop:         {single_hop_hit1:.4f}")
    print(f"   Multi-hop:          {multi_hop_hit1:.4f}")
    print(f"\nDelta vs RAG:")
    print(f"   Overall:            {overall_hit1 - 0.4912:+.4f}")
    print(f"   Single-hop:         {single_hop_hit1 - 0.8960:+.4f}")
    print(f"   Multi-hop:          {multi_hop_hit1 - 0.3816:+.4f}")
    print(f"─────────────────────────────────────────────────────────────")

    # ── save results ───────────────────────────────────────────────────────
    output = {
        "model_checkpoint":  BEST_MODEL_PATH,
        "best_model_epoch":  checkpoint['epoch'],
        "val_hit1":          checkpoint['val_hit1'],
        "test_results": {
            "overall_hit1":    overall_hit1,
            "single_hop_hit1": single_hop_hit1,
            "multi_hop_hit1":  multi_hop_hit1,
            "total":           total,
            "single_total":    single_total,
            "multi_total":     multi_total,
            "hits":            hits
        },
        "baseline_comparison": {
            "rag_overall":     0.4912,
            "rag_single_hop":  0.8960,
            "rag_multi_hop":   0.3816,
            "delta_overall":   overall_hit1 - 0.4912,
            "delta_single":    single_hop_hit1 - 0.8960,
            "delta_multi":     multi_hop_hit1 - 0.3816
        },
        "per_question_results": results
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Run on 10 questions only")
    args = parser.parse_args()
    evaluate_test(debug=args.debug)