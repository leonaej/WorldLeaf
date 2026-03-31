import numpy as np
import sys
sys.path.append("proposed_solution/RL_agent")
from utils import cosine_similarity


class Environment:
    
    
    def __init__(self, adjacency, node_info, node_matrix, node_to_idx, idx_to_node, edge_matrix, edge_to_idx):
        self.adjacency   = adjacency
        self.node_info   = node_info
        self.node_matrix = node_matrix
        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node  # ← add this line
        self.edge_matrix = edge_matrix
        self.edge_to_idx = edge_to_idx

    # ── start node selection ───────────────────────────────────────────────
      
    
    def get_start_nodes(self, query_embedding, top_k=3):
        """
        Cosine similarity between query and all node embeddings.
        Returns top_k node_ids as candidate start nodes.

        Args:
            query_embedding: np.ndarray (3072,)
            top_k:           int
        Returns:
            list of node_id strings, length top_k
        """
        sims = cosine_similarity(query_embedding, self.node_matrix)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [self.idx_to_node[i] for i in top_indices]

    # ── action space ───────────────────────────────────────────────────────
    def get_actions(self, current_node_id, query_embedding):
        """
        Build the action space at the current node.
        Each action is a dict with everything the policy needs to score it.

        Returns:
            actions: list of dicts, last one is always STOP
            Each edge action:
                {
                    type:        'edge',
                    edge_key:    'Q140__Q42569',
                    neighbor_id: 'Q42569',
                    relation:    'preys_on',
                    embedding:   np.ndarray (3072,),
                    cosine_sim:  float
                }
            STOP action:
                {
                    type:       'stop',
                    embedding:  np.ndarray (3072,),  ← current node embedding
                    cosine_sim: float                 ← sim(query, current node)
                }
        """
        actions = []
        outgoing = self.adjacency.get(current_node_id, [])

        for (neighbor_id, relation, edge_key) in outgoing:
            if edge_key in self.edge_to_idx:
                idx = self.edge_to_idx[edge_key]
                edge_emb = self.edge_matrix[idx]
                sim = float(cosine_similarity(query_embedding, 
                                              edge_emb.reshape(1, -1))[0])
                actions.append({
                    "type":        "edge",
                    "edge_key":    edge_key,
                    "neighbor_id": neighbor_id,
                    "relation":    relation,
                    "embedding":   edge_emb,
                    "cosine_sim":  sim
                })

        # get current node embedding + its cosine sim to query
        if current_node_id in self.node_to_idx:
            node_idx = self.node_to_idx[current_node_id]
            node_emb = self.node_matrix[node_idx]
            node_sim = float(cosine_similarity(query_embedding,
                                               node_emb.reshape(1, -1))[0])
        else:
            node_emb = np.zeros(3072)
            node_sim = 0.0

        # STOP is always the last action
        actions.append({
            "type":       "stop",
            "embedding":  node_emb,
            "cosine_sim": node_sim
        })

        return actions

    # ── step ───────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Execute an action.

        Args:
            action: dict from get_actions()
        Returns:
            next_node_id: str or None (None if STOP)
            stopped:      bool
            is_dead_end:  bool (True if no outgoing edges at all)
        """
        if action["type"] == "stop":
            return None, True, False

        return action["neighbor_id"], False, False

    # ── dead end check ─────────────────────────────────────────────────────
    def is_dead_end(self, node_id):
        """
        True if node has no outgoing edges in the graph.
        """
        outgoing = self.adjacency.get(node_id, [])
        # dead end if no edges exist OR none have embeddings
        valid = [e for e in outgoing 
                 if f"{node_id}__{e[0]}" in self.edge_to_idx]
        return len(valid) == 0

    # ── reward ─────────────────────────────────────────────────────────────
    def compute_reward(self, current_node_id, answer_node_ids, 
                       hops_taken, stopped_by_agent, is_dead_end):
        """
        Args:
            current_node_id:  str — node agent stopped at
            answer_node_ids:  set of valid answer node id strings
            hops_taken:       int
            stopped_by_agent: bool — True if agent chose STOP
                                     False if forced (max hops or dead end)
            is_dead_end:      bool

        Returns:
            reward: float
        """
        HOP_PENALTY    = 0.1
        WRONG_STOP     = -0.1
        FORCED_STOP    = -0.2

        at_answer = current_node_id in answer_node_ids
        hop_cost  = hops_taken * HOP_PENALTY

        # correct answer
        if at_answer:
            return 1.0 - hop_cost

        # agent chose to stop at wrong node
        if stopped_by_agent and not is_dead_end:
            return WRONG_STOP - hop_cost

        # dead end — our fault, no penalty
        if is_dead_end:
            return 0.0

        # forced stop — agent wandered too long
        return FORCED_STOP