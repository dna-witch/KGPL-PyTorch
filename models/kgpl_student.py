"""KGPL student model for knowledge graph completion and recommendation.
This model is a student model that learns from the teacher model (KGPL_COT)."""

import random
import itertools
import numpy as np
import tqdm
import copy
from collections import defaultdict, Counter
import multiprocessing as mp
from loguru import logger
import hydra

from sklearn.metrics import f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

# change according to your project structure and file names
from kapl import KaPLMixin
from aggregators import SumAggregatorWithDropout
from utils import grouper

class KGPLStudent(nn.Module):
    def __init__(self, n_users, n_entities, n_relations, adj_entity, adj_relation,
                 emb_dim, n_iter, neighbor_size, dropout_rate):
        super(KGPLStudent, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.emb_dim = emb_dim
        self.n_iter = n_iter
        self.neighbor_size = neighbor_size
        self.dropout_rate = dropout_rate

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.entity_embedding = nn.Embedding(n_entities, emb_dim)
        self.relation_embedding = nn.Embedding(n_relations, emb_dim)

        self.adj_entity = torch.LongTensor(adj_entity)
        self.adj_relation = torch.LongTensor(adj_relation)

        self.aggregators = nn.ModuleList()
        for layer in range(n_iter):
            act_fn = nn.Tanh() if layer == n_iter - 1 else nn.LeakyReLU()
            self.aggregators.append(SumAggregator(emb_dim, dropout_rate, act_fn))

    def get_neighbors(self, item_indices):
        batch_size = item_indices.size(0)
        entities = [item_indices.unsqueeze(1)]
        relations = []

        for _ in range(self.n_iter):
            cur_entities = entities[-1]
            neigh_ent = self.adj_entity[cur_entities]
            neigh_rel = self.adj_relation[cur_entities]
            entities.append(neigh_ent)
            relations.append(neigh_rel)

        return entities, relations

    def forward(self, user_indices, item_indices):
        batch_size = user_indices.size(0)
        user_vec = self.user_embedding(user_indices)
        user_vec = F.dropout(user_vec, p=self.dropout_rate, training=self.training)

        entities_list, relations_list = self.get_neighbors(item_indices)

        entity_vectors = [self.entity_embedding(e) for e in entities_list]
        relation_vectors = [self.relation_embedding(r) for r in relations_list]

        for i in range(self.n_iter):
            self_vectors = entity_vectors[i]
            neighbor_vectors = entity_vectors[i + 1].view(batch_size, -1, self.neighbor_size, self.emb_dim)
            neighbor_relations = relation_vectors[i].view(batch_size, -1, self.neighbor_size, self.emb_dim)
            entity_vectors[i] = self.aggregators[i](self_vectors, neighbor_vectors, neighbor_relations, user_vec)

        item_vec = entity_vectors[0].squeeze(1)
        scores = (user_vec * item_vec).sum(dim=-1)
        return torch.sigmoid(scores)