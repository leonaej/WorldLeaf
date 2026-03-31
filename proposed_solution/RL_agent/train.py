import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

sys.path.append("proposed_solution/RL_agent")
from utils import (load_graph, load_node_embeddings, load_edge_embeddings,
                   load_query_embeddings, load_training_data, load_eval_data)
from environment import Environment
from policy import PolicyNetwork

# ── config ─────────────────────────────────────────────────────────────────
CONFIG = {
    "max_epochs":       30,
    "beam_width":       3,
    "top_k_start":      3,
    "lr":               1e-4,
    "early_stop_patience": 5,
    "val_every":        2,
    "checkpoint_every": 5,
    "debug_episodes":   100,
    "seed":             42,

    # hop decay schedule
    "hop_schedule": {
        1:  5,   # epochs 1-5:  max_hops = 5
        6:  4,   # epochs 6-10: max_hops = 4
        11: 3,   # epochs 11+:  max_hops = 3
    }
}

CHECKPOINT_DIR = "proposed_solution/RL_agent/checkpoints"
LOG_PATH       = "proposed_solution/RL_agent/training_log.json"


# ── hop schedule ───────────────────────────────────────────────────────────
def get_max_hops(epoch):
    max_hops = 5
    for start_epoch, hops in sorted(CONFIG["hop_schedule"].items()):
        if epoch >= start_epoch:
            max_hops = hops
    return max_hops


# ── beam search trajectory ─────────────────────────────────────────────────
def run_trajectory(start_node, query_emb, answer_node_ids,
                   env, policy, device, max_hops, beam_width):
    """
    Run one beam search trajectory from a single start node.

    Returns:
        total_log_prob: torch.Tensor — sum of log probs of all actions taken
        reward:         float — reward at end of trajectory
        hops_taken:     int
        final_nodes:    list of str — all nodes beam ended at
    """
    # beam: list of (node_id, log_prob_sum, hops_taken, log_probs_list)
    beam = [(start_node, 0.0, 0, [])]
    completed = []  # finished beams (stopped or dead end)

    for hop in range(max_hops):
        if not beam:
            break

        next_beam = []

        for (current_node, cum_log_prob, hops, log_probs_list) in beam:

            # dead end check
            if env.is_dead_end(current_node):
                completed.append((current_node, cum_log_prob,
                                  hops, log_probs_list, True))
                continue

            # get actions
            actions = env.get_actions(current_node, query_emb)


            # get all action scores + log probs in one forward pass
            query_tensor = torch.tensor(query_emb, dtype=torch.float32).to(device)
            action_embeddings = torch.tensor(np.array([a['embedding'] for a in actions]),dtype=torch.float32).to(device)
            cosine_sims = torch.tensor( np.array([a['cosine_sim'] for a in actions]),dtype=torch.float32).to(device)

            probs, log_probs_all = policy.forward(query_tensor, action_embeddings, cosine_sims)

            # get top beam_width actions by probability
            probs_np = probs.detach().cpu().numpy()
            top_indices = np.argsort(probs_np)[::-1][:beam_width]

            for idx in top_indices:
                action = actions[idx]
                action_log_prob = log_probs_all[idx]  # correct log prob per action


                if action["type"] == "stop":
                    completed.append((
                        current_node,
                        cum_log_prob + action_log_prob,
                        hops,
                        log_probs_list + [action_log_prob],
                        False  # not dead end, agent chose stop
                    ))
                else:
                    next_beam.append((
                        action["neighbor_id"],
                        cum_log_prob + action_log_prob,
                        hops + 1,
                        log_probs_list + [action_log_prob]
                    ))

        # force stop anything still in beam if max hops reached
        if hop == max_hops - 1:
            for (current_node, cum_log_prob, hops, log_probs_list) in next_beam:
                completed.append((current_node, cum_log_prob,
                                  hops, log_probs_list, False))
            next_beam = []

        beam = next_beam

    if not completed:
        return None, 0.0, 0, []

    # compute rewards for all completed paths
    best_reward = -999
    best_log_probs = []
    final_nodes = []

    for (node, cum_lp, hops, lp_list, is_dead) in completed:
        reward = env.compute_reward(
            node, answer_node_ids, hops,
            stopped_by_agent=not is_dead,
            is_dead_end=is_dead
        )
        final_nodes.append(node)
        if reward > best_reward:
            best_reward = reward
            best_log_probs = lp_list

    # sum log probs of best path for REINFORCE
    if best_log_probs:
        total_log_prob = sum(best_log_probs)
    else:
        total_log_prob = torch.tensor(0.0)

    return total_log_prob, best_reward, len(completed), final_nodes


# ── evaluate on val/test set ───────────────────────────────────────────────
def evaluate(env, policy, device, query_matrix, query_lookup,
             eval_data, max_hops, beam_width, top_k_start):
    policy.eval()
    hits = 0
    single_hits = 0
    single_total = 0
    multi_hits = 0
    multi_total = 0

    for qa in tqdm(eval_data, desc="Evaluating", leave=False):
        question = qa['question']
        answer_ids = {a['id'] for a in qa['answer_nodes']}
        hop_type = qa['hop_type']

        if question not in query_lookup:
            continue

        query_emb = query_matrix[query_lookup[question]['index']]
        start_nodes = env.get_start_nodes(query_emb, top_k=top_k_start)

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

    total = len(eval_data)
    results = {
        "overall_hit1":    hits / total if total > 0 else 0,
        "single_hop_hit1": single_hits / single_total if single_total > 0 else 0,
        "multi_hop_hit1":  multi_hits / multi_total if multi_total > 0 else 0,
        "total":           total,
        "single_total":    single_total,
        "multi_total":     multi_total
    }
    policy.train()
    return results


# ── main training loop ─────────────────────────────────────────────────────
def train(debug=False):
    # seed
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load everything
    adjacency, node_info           = load_graph()
    node_matrix, node_to_idx, idx_to_node = load_node_embeddings()
    edge_matrix, edge_to_idx, _    = load_edge_embeddings()
    query_matrix, query_lookup     = load_query_embeddings()
    training_data                  = load_training_data()
    eval_data                      = load_eval_data()

    # debug mode — small subset
    if debug:
        training_data = training_data[:CONFIG["debug_episodes"]]
        eval_data     = eval_data[:50]
        print(f"DEBUG MODE: {len(training_data)} episodes, {len(eval_data)} eval")

    # 70/15/15 split on eval data
    random.shuffle(eval_data)
    n = len(eval_data)
    val_data  = eval_data[:int(0.15 * n)]
    

    test_data = eval_data[int(0.15 * n):int(0.30 * n)]
    test_data_path = "proposed_solution/RL_agent/test_data.json"
    if not os.path.exists(test_data_path):
        with open(test_data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"Test data saved: {len(test_data)} questions → {test_data_path}")

    # environment + policy
    env = Environment(adjacency, node_info, node_matrix,
                      node_to_idx, idx_to_node, edge_matrix, edge_to_idx)
    policy = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=CONFIG["lr"])

    # logging
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log = {
        "config":       CONFIG,
        "train_losses": [],
        "val_results":  [],
        "best_val":     0.0,
        "best_epoch":   0,
        "started_at":   datetime.now().isoformat()
    }

    best_val_hit1   = 0.0
    patience_counter = 0

    print(f"\nStarting training: {len(training_data)} episodes, "
          f"{CONFIG['max_epochs']} epochs")
    print(f"Val set: {len(val_data)} questions\n")

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        max_hops = get_max_hops(epoch)
        print(f"── Epoch {epoch}/{CONFIG['max_epochs']} "
              f"(max_hops={max_hops}) ──")

        # shuffle training data
        random.shuffle(training_data)

        epoch_losses  = []
        epoch_rewards = []
        policy.train()

        for episode in tqdm(training_data, desc=f"Epoch {epoch}"):
            question       = episode['question']
            answer_node_id = episode['answer_node_id']

            if question not in query_lookup:
                continue

            query_emb   = query_matrix[query_lookup[question]['index']]
            start_nodes = env.get_start_nodes(query_emb,
                                              top_k=CONFIG["top_k_start"])
            answer_ids  = {answer_node_id}

            # run trajectory for each start node
            episode_log_probs = []
            episode_rewards   = []

            for start_node in start_nodes:
                total_lp, reward, _, _ = run_trajectory(
                    start_node, query_emb, answer_ids,
                    env, policy, device, max_hops, CONFIG["beam_width"]
                )
                if total_lp is not None:
                    episode_log_probs.append(total_lp)
                    episode_rewards.append(reward)

            if not episode_log_probs:
                continue

            # REINFORCE loss — average across 3 trajectories
            avg_reward = np.mean(episode_rewards)
            loss_terms = []
            for lp, r in zip(episode_log_probs, episode_rewards):
                if isinstance(lp, torch.Tensor) and lp.requires_grad:
                    loss_terms.append(-lp * r)

            if not loss_terms:
                continue

            loss = torch.stack(loss_terms).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()





            epoch_losses.append(loss.item())
            epoch_rewards.append(avg_reward)

        # epoch stats
        avg_loss   = np.mean(epoch_losses)   if epoch_losses   else 0
        avg_reward = np.mean(epoch_rewards)  if epoch_rewards  else 0
        print(f"  Loss: {avg_loss:.4f} | Avg Reward: {avg_reward:.4f}")

        log["train_losses"].append({
            "epoch":      epoch,
            "loss":       avg_loss,
            "avg_reward": avg_reward,
            "max_hops":   max_hops
        })

        # validation
        if epoch % CONFIG["val_every"] == 0:
            val_results = evaluate(
                env, policy, device, query_matrix, query_lookup,
                val_data, max_hops, CONFIG["beam_width"],
                CONFIG["top_k_start"]
            )
            print(f"  Val Hit@1: {val_results['overall_hit1']:.4f} "
                  f"(single: {val_results['single_hop_hit1']:.4f}, "
                  f"multi: {val_results['multi_hop_hit1']:.4f})")

            log["val_results"].append({
                "epoch": epoch,
                **val_results
            })

            # save best model
            if val_results["overall_hit1"] > best_val_hit1:
                best_val_hit1    = val_results["overall_hit1"]
                patience_counter = 0
                log["best_val"]  = best_val_hit1
                log["best_epoch"] = epoch
                torch.save({
                    "epoch":        epoch,
                    "model_state":  policy.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_hit1":     best_val_hit1,
                    "config":       CONFIG
                }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
                print(f"  ✅ New best model saved! Val Hit@1: {best_val_hit1:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: "
                      f"{patience_counter}/{CONFIG['early_stop_patience']}")

            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"\nEarly stopping at epoch {epoch}!")
                break

        # periodic checkpoint
        if epoch % CONFIG["checkpoint_every"] == 0:
            torch.save({
                "epoch":       epoch,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config":      CONFIG
            }, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pt"))
            print(f"  Checkpoint saved at epoch {epoch}")

        # save log after every epoch
        with open(LOG_PATH, 'w') as f:
            json.dump(log, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best Val Hit@1: {best_val_hit1:.4f} at epoch {log['best_epoch']}")
    print(f"Log saved to {LOG_PATH}")


# ── entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="Run on small subset for testing")
    args = parser.parse_args()
    train(debug=args.debug)