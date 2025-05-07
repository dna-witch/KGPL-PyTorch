import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from abc import ABC, abstractmethod

def kgpl_loss(pos_scores, neg_scores, pseudo_scores):
    # BCE loss like TensorFlow version
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    pseudo_labels = torch.ones_like(pseudo_scores)

    # criterion = nn.BCELoss(reduction='mean')
    
    loss = F.binary_cross_entropy_loss(pos_scores, pos_labels) + \
           F.binary_cross_entropy_loss(neg_scores, neg_labels) + \
           F.binary_cross_entropy_loss(pseudo_scores, pseudo_labels)
    return loss


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

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None):
        return self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks)

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None):
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
      avg = False
      if not avg:
          batch_size = user_embeddings.size(0)  # <-- dynamic batch size
          user_embeddings = user_embeddings.view(batch_size, 1, 1, self.dim)
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
        output = self_vectors + neighbors_agg  # [batch, k, dim]
        output = output.view(-1, self.dim)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.linear(output)
        batch_size = self_vectors.size(0)
        output = output.view(batch_size, -1, self.dim)
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

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# <------- Sum Aggregator with Dropout ------->
class SumAggregator(nn.Module):
    def __init__(self, dim, dropout=0.0, act_fn=nn.LeakyReLU()):
        super(SumAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.act = act_fn
        # Weight and bias parameters (like a graph conv)
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        """
        self_vectors: [batch, 1, dim] or [batch, k, dim] (entity embeddings at current hop)
        neighbor_vectors: [batch, k, num_neighbors, dim] (neighbor embeddings)
        neighbor_relations: [batch, k, num_neighbors, dim] (relation embeddings)
        user_embeddings: [batch, dim] (user embedding)
        """
        # 1) Compute attention scores for neighbors based on (user * relation)
        batch_size = user_embeddings.size(0)
        # Expand user to [batch, 1, 1, dim] to match neighbor_relations
        u = user_embeddings.view(batch_size, 1, 1, self.dim)
        # Compute score = softmax( <user, neighbor_relation> ) along neighbors
        scores = torch.softmax((u * neighbor_relations).mean(dim=-1), dim=-1)  # [batch, k, num_neighbors]
        scores = scores.unsqueeze(-1)  # [batch, k, num_neighbors, 1]
        # 2) Weighted average of neighbor vectors
        neigh_agg = (scores * neighbor_vectors).sum(dim=2)  # [batch, k, dim]
        neigh_agg = F.dropout(neigh_agg, p=self.dropout, training=self.training)
        # 3) Sum with self vectors and apply linear + activation.
        output = self_vectors + neigh_agg  # [batch, k, dim]
        output = output.view(-1, self.dim)  # flatten [batch*k, dim]
        output = output @ self.weight + self.bias  # linear transform
        output = output.view(batch_size, -1, self.dim)  # [batch, k, dim]
        output = self.act(output)
        return output


# <------- KGPL Student Class Definition ------->

class KGPLStudent(nn.Module):
    def __init__(self, n_user, n_entity, n_relation, adj_entity, adj_relation, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.adj_entity = torch.LongTensor(adj_entity).to(device)   # [n_entity, neigh_size]
        self.adj_relation = torch.LongTensor(adj_relation).to(device)
        self.user_emb = nn.Embedding(n_user, config['emb_dim'])
        self.entity_emb = nn.Embedding(n_entity, config['emb_dim'])
        self.relation_emb = nn.Embedding(n_relation, config['emb_dim'])
        self.emb_dim = config['emb_dim']

        # Pre-build aggregators for each hop
        self.aggregators = nn.ModuleList([
            SumAggregatorWithDropout(config['optimize']['batch_size'], config['emb_dim'],
                                     dropout=config['model']['dropout_rate'],
                                     act=F.relu if i < config['model']['n_iter']-1 else torch.tanh)
            for i in range(config['model']['n_iter'])
        ])
        self.to(device)

    def forward(self, user_indices, item_indices, masks=None):
        """
        user_indices: [batch]
        item_indices: [batch]
        masks: dict with 'neg_mask' and 'plabel_mask' [batch] (if used)
        """
        batch_size = user_indices.size(0)
        device = self.device

        # Fetch embeddings and apply dropout on user
        u_emb = self.user_emb(user_indices)       # [batch, dim]
        u_emb = F.dropout(u_emb, p=self.config['model']['dropout_rate'], training=self.training)

        # Multi-hop neighbor retrieval for items
        entities, relations = self.get_neighbors(item_indices, batch_size)
        # Embed neighbors
        ent_vecs = [self.entity_emb(e.to(device)) for e in entities]       # list of [batch, #nodes, dim]
        rel_vecs = [self.relation_emb(r.to(device)) for r in relations]     # list of [batch, #nodes, dim]

        # GNN aggregation
        x = ent_vecs  # initial item vectors per hop
        for i in range(self.config['model']['n_iter']):
            aggregator = self.aggregators[i]
            next_x = []
            # For each depth hop, aggregate (like TensorFlow loop)
            for hop in range(self.config['model']['n_iter'] - i):
                self_vecs = x[hop]
                neigh_vecs = x[hop+1].view(batch_size, -1, self.config['model']['neighbor_sample_size'], self.emb_dim)
                neigh_rels = rel_vecs[hop].view(batch_size, -1, self.config['model']['neighbor_sample_size'], self.emb_dim)
                agg_out = aggregator(self_vecs, neigh_vecs, neigh_rels, u_emb)
                next_x.append(agg_out)
            x = next_x
        item_repr = x[0].squeeze()  # [batch, dim] final item embedding

        scores = torch.sum(u_emb * item_repr, dim=1)  # [batch]
        return scores

    def get_neighbors(self, items, batch_size):
        """
        Retrieves multi-hop neighbor lists (entities and relations).
        """
        items = items.to(self.device).view(-1, 1)  # [batch,1]
        entities = [items]
        relations = []
        for _ in range(self.config['n_iter']):
            neigh_e = self.adj_entity[entities[-1]].view(batch_size, -1)
            neigh_r = self.adj_relation[entities[-1]].view(batch_size, -1)
            entities.append(neigh_e)
            relations.append(neigh_r)
        return entities, relations

# <----- KGPL Co-Training Model ----->
class KGPLCOT(nn.Module):
    def __init__(self, cfg, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, path_list_dict, train_data, eval_data, device):
        super().__init__()
        _cfg = deepcopy(cfg)
        self.device = device

        self.model_f = KGPLStudent(n_user, n_entity, n_relation, adj_entity, adj_relation, _cfg, device)
        self.model_g = KGPLStudent(n_user, n_entity, n_relation, adj_entity, adj_relation, _cfg, device)

        # Placeholder for item candidates if needed later
        # Example: self.item_candidates = {user: [items]}

    def split_data(self, data):
        mid = len(data) // 2
        return data[:mid], data[mid:]

    def forward(self, batch_data):
        # Forward pass is not typically called for Co-Training directly
        raise NotImplementedError("Use train_step for co-training")

    def train_step(self, batch_data, optimizer_f, optimizer_g):
        log_f, log_g = self.split_data(batch_data)

        users_f, pos_f, neg_f, pseudo_f = log_f[:, 0], log_f[:, 1], log_f[:, 2], log_f[:, 3]
        users_g, pos_g, neg_g, pseudo_g = log_g[:, 0], log_g[:, 1], log_g[:, 2], log_g[:, 3]

        # Convert all to device
        users_f = users_f.to(self.device)
        pos_f = pos_f.to(self.device)
        neg_f = neg_f.to(self.device)
        pseudo_f = pseudo_f.to(self.device)

        users_g = users_g.to(self.device)
        pos_g = pos_g.to(self.device)
        neg_g = neg_g.to(self.device)
        pseudo_g = pseudo_g.to(self.device)

        # === Train model_f on g's data ===
        pos_scores_f = self.model_f(users_g, pos_g)
        neg_scores_f = self.model_f(users_g, neg_g)
        pseudo_scores_f = self.model_f(users_g, pseudo_g)
        loss_f = kgpl_loss(pos_scores_f, neg_scores_f, pseudo_scores_f)

        optimizer_f.zero_grad()
        loss_f.backward()
        optimizer_f.step()

        # === Train model_g on f's data ===
        pos_scores_g = self.model_g(users_f, pos_f)
        neg_scores_g = self.model_g(users_f, neg_f)
        pseudo_scores_g = self.model_g(users_f, pseudo_f)
        loss_g = kgpl_loss(pos_scores_g, neg_scores_g, pseudo_scores_g)

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        return {'loss_f': loss_f.item(), 'loss_g': loss_g.item()}

    def evaluate(self, model_name, users, items):
        # Evaluate only model_f typically
        model = self.model_f if model_name == 'f' else self.model_g
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(users.to(self.device), items.to(self.device)))
        return scores.cpu()
