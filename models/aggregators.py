"""
The Aggregator of KGNN-LS.

This module defines the base class for aggregators used in the KGNN-LS model.
It includes methods for initializing the aggregator, calling it with input vectors and relations, and defining the forward pass.
It also provides a method to get the layer ID for a given layer name.

Need to rewrite the code to be more pythonic and to use PyTorch instead of TensorFlow.
The code is based on the original implementation of KGNN-LS, which can be found at: 
https://github.com/hwwang55/KGNN-LS .
"""

from abc import abstractmethod

import torch
from torch import nn
import torch.nn.functional as F


LAYER_IDS = {}

def get_layer_id(layer_name: str) -> int:
    """Get the layer ID for a given layer name."""
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        # Should return 0
    else:
        # If the layer name is already in the dictionary, increment its ID
        # and return the new ID.
        LAYER_IDS[layer_name] += 1
    return LAYER_IDS[layer_name]

class Aggregator(nn.Module):
    """Base class for aggregators."""
    
    def __init__(self, batch_size: int, dim: int, dropout: float = 0.0, act, name, cfg=None) -> None:
        super(Aggregator, self).__init__()
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.batch_size = batch_size
        self.dim = dim
        self.dropout = dropout
        self.act = act  # activation function
        self.cfg = cfg  # config for the aggregator
    
    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        """Call the aggregator with the given vectors and relations."""
        # return self.forward(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks)
        outputs = self._call(
            self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks
        )
        return outputs
    
    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        """Abstract method to be implemented by subclasses."""
        # dimension: 
        # self_vectors: (batch_size, -1, dim). For LabelAggregator -> (batch_size, -1)
        # neighbor_vectors: (batch_size, -1, n_neighbor, dim). For LabelAggregator -> (batch_size, -1, n_neighbor)
        # neighbor_relations: (batch_size, -1, n_neighbor, dim).
        # user_embeddings: (batch_size, dim)
        # masks: Only for LabelAggregator.
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        """Mix neighbor vectors"""
        avg = False

        if not avg:
            # (batch_size, 1, 1, dim)
            user_embeddings = user_embeddings.view(self.batch_size, 1, 1, self.dim)
            
            # (batch_size, -1, n_neighbor)
            user_relation_scores = torch.mean(user_embeddings * neighbor_relations, dim=-1)
            user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)

            # (batch_size, -1, n_neighbor, 1)
            user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(-1)

            # (batch_size, -1, dim)
            neighbors_aggregated = torch.mean(user_relation_scores_normalized * neighbor_vectors, dim=-2)  # axis=2 for the orig tensorflow code

        else:
            # (batch_size, -1, dim)
            neighbors_aggregated = torch.mean(neighbor_vectors, dim=-2)
        
        return neighbors_aggregated
    
class SumAggregatorWithDropout(Aggregator):
    """Sum aggregator with dropout."""
    
    def __init__(self, batch_size: int, dim: int, dropout: float = 0.0, act = F.leaky_relu, name=None, cfg=None) -> None:
        super(SumAggregatorWithDropout, self).__init__(batch_size, dim, dropout, act, name)
        
        self.weights = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)
        # with tf.variable_scope(self.name):
        #     self.weights = tf.get_variable(
        #         shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
        #     self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        neighbors_aggregated = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)  # (batch_size, -1, dim)
        
        output = torch.reshape(self_vectors + neighbors_aggregated, (-1, self.dim))  # (batch_size, -1, dim) -> (-1, dim)
        output = F.dropout(output, p=1-self.dropout)
        output = F.matmul(output, self.weights) + self.bias

        # back to (batch_size, -1, dim)
        output = torch.reshape(output, (self.batch_size, -1, self.dim))
        output = self.act(output)

        return output
    
class LabelAggregator(Aggregator):
    """Label aggregator."""
    def __init__(self, batch_size: int, dim: int, name=None):
        super(LabelAggregator, self).__init__(batch_size, dim, 0., None, name=name)
    
    def _call(self, self_labels, neighbor_labels, neighbor_relations, user_embeddings, masks):
        # (batch_size, 1, 1, dim)
        user_embeddings = user_embeddings.view(self.batch_size, 1, 1, self.dim)  # view vs reshape??
        
        # (batch_size, -1, n_neighbor)
        user_relation_scores = torch.mean(user_embeddings * neighbor_relations, dim=-1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)

        # (batch_size, -1)
        neighbors_aggregated = torch.mean(user_relation_scores_normalized * neighbor_labels, dim=-1)  # or dim=2? 
        output = masks.float() * self_labels + (~masks).float() * neighbors_aggregated

        return output
        
    
            