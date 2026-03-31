import torch
import sys
sys.path.append("proposed_solution/RL_agent")
from utils import *
from environment import Environment
from policy import PolicyNetwork

adjacency, node_info = load_graph()
node_matrix, node_to_idx, idx_to_node = load_node_embeddings()
edge_matrix, edge_to_idx, idx_to_edge = load_edge_embeddings()
query_matrix, query_lookup = load_query_embeddings()

env = Environment(adjacency, node_info, node_matrix,
                  node_to_idx, idx_to_node, edge_matrix, edge_to_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

policy = PolicyNetwork().to(device)
print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

# test one step
query_emb = query_matrix[0]
start_nodes = env.get_start_nodes(query_emb, top_k=3)
actions = env.get_actions(start_nodes[0], query_emb)

selected, log_prob, probs = policy.select_action(query_emb, actions, device)
print(f"Selected action type: {selected['type']}")
print(f"Log prob: {log_prob.item():.4f}")
print(f"Probs sum: {probs.sum().item():.4f}")