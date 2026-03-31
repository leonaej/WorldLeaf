import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Scores each action (edge or STOP) at a given node.

    Input per action:
        [query (3072d) | action_embedding (3072d) | cosine_sim (1d)] = 6145d

    For edge actions:   action_embedding = edge embedding
    For STOP action:    action_embedding = current node embedding

    Output:
        scalar score per action → softmax over all actions → probability distribution
    """

    def __init__(self):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(6145, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # scalar score
        )

    def forward(self, query_emb, action_embeddings, cosine_sims):
        """
        Score all actions at once.

        Args:
            query_emb:         torch.Tensor (3072,)
            action_embeddings: torch.Tensor (num_actions, 3072)
            cosine_sims:       torch.Tensor (num_actions,)

        Returns:
            probs: torch.Tensor (num_actions,) — probability per action
            log_probs: torch.Tensor (num_actions,) — log probability per action
        """
        num_actions = action_embeddings.shape[0]

        # expand query to match number of actions
        # (3072,) → (num_actions, 3072)
        query_expanded = query_emb.unsqueeze(0).expand(num_actions, -1)

        # cosine sims → (num_actions, 1)
        cosine_sims = cosine_sims.unsqueeze(1)

        # concatenate → (num_actions, 6145)
        x = torch.cat([query_expanded, action_embeddings, cosine_sims], dim=1)

        # score each action → (num_actions, 1) → (num_actions,)
        scores = self.network(x).squeeze(1)

        # softmax → probability distribution over actions
        probs = F.softmax(scores, dim=0)
        log_probs = F.log_softmax(scores, dim=0)

        return probs, log_probs

    def select_action(self, query_emb, actions, device):
        """
        Given the action list from environment.get_actions(),
        build tensors, score all actions, sample one.

        Args:
            query_emb: np.ndarray (3072,)
            actions:   list of dicts from environment.get_actions()
            device:    torch.device

        Returns:
            selected_action:  dict — the chosen action
            log_prob:         torch.Tensor scalar — log prob of chosen action
            probs:            torch.Tensor (num_actions,) — full distribution
        """
        import numpy as np

        # build tensors from action dicts
        action_embeddings = torch.tensor(
            np.array([a['embedding'] for a in actions]),
            dtype=torch.float32
        ).to(device)

        cosine_sims = torch.tensor(
            np.array([a['cosine_sim'] for a in actions]),
            dtype=torch.float32
        ).to(device)

        query_tensor = torch.tensor(
            query_emb, dtype=torch.float32
        ).to(device)

        # forward pass
        with torch.no_grad():
            probs, log_probs = self.forward(query_tensor,
                                            action_embeddings,
                                            cosine_sims)

        # sample action from distribution
        action_idx = torch.multinomial(probs, num_samples=1).item()
        selected_action = actions[action_idx]
        log_prob = log_probs[action_idx]

        return selected_action, log_prob, probs

    def evaluate_actions(self, query_emb, actions, device):
        """
        Same as select_action but keeps gradients for training.
        Used during the REINFORCE update.

        Args:
            query_emb: np.ndarray (3072,)
            actions:   list of dicts from environment.get_actions()
            device:    torch.device

        Returns:
            selected_action: dict
            log_prob:        torch.Tensor scalar — WITH gradients
            probs:           torch.Tensor (num_actions,)
        """
        import numpy as np

        action_embeddings = torch.tensor(
            np.array([a['embedding'] for a in actions]),
            dtype=torch.float32
        ).to(device)

        cosine_sims = torch.tensor(
            np.array([a['cosine_sim'] for a in actions]),
            dtype=torch.float32
        ).to(device)

        query_tensor = torch.tensor(
            query_emb, dtype=torch.float32
        ).to(device)

        # forward pass WITH gradients
        probs, log_probs = self.forward(query_tensor,
                                        action_embeddings,
                                        cosine_sims)

        # sample action
        action_idx = torch.multinomial(probs.detach(), num_samples=1).item()
        selected_action = actions[action_idx]
        log_prob = log_probs[action_idx]

        return selected_action, log_prob, probs