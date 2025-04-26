import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

LAYER_IDS = {}

def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]

class Aggregator(nn.Module, ABC):
    def __init__(self, batch_size, dim, dropout, act, name=None):
        super(Aggregator, self).__init__()
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        return self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks)

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            user_embeddings = user_embeddings.view(self.batch_size, 1, 1, self.dim)
            user_relation_scores = torch.mean(user_embeddings * neighbor_relations, dim=-1)
            user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)
            user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(-1)
            neighbors_aggregated = torch.mean(user_relation_scores_normalized * neighbor_vectors, dim=2)
        else:
            neighbors_aggregated = torch.mean(neighbor_vectors, dim=2)
        return neighbors_aggregated

class SumAggregatorWithDropout(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=F.relu, name=None):
        super(SumAggregatorWithDropout, self).__init__(batch_size, dim, dropout, act, name)
        self.linear = nn.Linear(dim, dim)

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        output = (self_vectors + neighbors_agg).view(-1, self.dim)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.linear(output)
        output = output.view(self.batch_size, -1, self.dim)

        return self.act(output)

class LabelAggregator(Aggregator):
    def __init__(self, batch_size, dim, name=None):
        super(LabelAggregator, self).__init__(batch_size, dim, 0., None, name)

    def _call(self, self_labels, neighbor_labels, neighbor_relations, user_embeddings, masks):
        user_embeddings = user_embeddings.view(self.batch_size, 1, 1, self.dim)
        user_relation_scores = torch.mean(user_embeddings * neighbor_relations, dim=-1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)
        neighbors_aggregated = torch.mean(user_relation_scores_normalized * neighbor_labels, dim=-1)
        masks = masks.float()
        output = masks * self_labels + (1 - masks) * neighbors_aggregated
        return output

if __name__ == "__main__":
    # This is a test case to check if the aggregator works correctly

    # Settings
    batch_size = 4
    n_entities = 10
    n_neighbors = 5
    dim = 16
    dropout = 0.5

    # Fake input tensors
    self_vectors = torch.randn(batch_size, n_entities, dim)
    neighbor_vectors = torch.randn(batch_size, n_entities, n_neighbors, dim)
    neighbor_relations = torch.randn(batch_size, n_entities, n_neighbors, dim)
    user_embeddings = torch.randn(batch_size, dim)
    masks = torch.randint(0, 2, (batch_size, n_entities))

    # Initialize aggregator
    aggregator = SumAggregatorWithDropout(batch_size, dim, dropout)

    # Set to train mode to apply dropout
    aggregator.train()

    # Forward pass
    output = aggregator(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks)

    print(f"Output shape: {output.shape}")  # Should be [batch_size, n_entities, dim] -> [4, 10, 16]
