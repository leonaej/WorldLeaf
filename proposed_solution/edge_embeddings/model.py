import torch
import torch.nn as nn

RELATION_TYPES = [
    'parent_taxon',
    'eats',
    'preys_on',
    'scavenges_from',
    'disperses_seeds_of',
    'symbiotic_with',
    'migrates_with',
    'parasitizes',
    'pollinates'
]

class EdgeEmbeddingModel(nn.Module):
    def __init__(self, node_embedding_dim=6144, hidden_dims=[4096, 2048, 1024],
                 edge_embedding_dim=512, num_relations=9):
        super(EdgeEmbeddingModel, self).__init__()

        input_dim = node_embedding_dim * 2  # 12,288

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], edge_embedding_dim),
            nn.ReLU()
        )

        # classification head, thrown away after training
        self.classifier = nn.Sequential(
            nn.Linear(edge_embedding_dim, num_relations),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.encoder(x)
        prediction = self.classifier(embedding)
        return embedding, prediction

    def get_embedding(self, x):
        with torch.no_grad():
            embedding = self.encoder(x)
        return embedding