"""
This is a refactored version of the original code.

The refactoring includes:
+ Convert everything to torch.nn.Module-style classes

+ Replace sess.run and TensorFlow feed_dict patterns with direct tensor operations

+ Use PyTorch dataloaders and batch tensors natively

+ Get rid of "placeholder" style programming

+ Move "evaluation mode" toggling to proper .eval() / .train() methods,

+ General polish (naming, typing, deep copies, logic)
"""

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
    def __init__(self, n_user, n_entity, n_relation, adj_entity, adj_relation, config: ModelConfig, device):
        super().__init__()
        self.device = device
        self.config = config
        self.adj_entity = torch.LongTensor(adj_entity).to(device)   # [n_entity, neigh_size]
        self.adj_relation = torch.LongTensor(adj_relation).to(device)
        self.user_emb = nn.Embedding(n_user, config.emb_dim)
        self.entity_emb = nn.Embedding(n_entity, config.emb_dim)
        self.relation_emb = nn.Embedding(n_relation, config.emb_dim)
        self.emb_dim = config.emb_dim

        # Pre-build aggregators for each hop
        self.aggregators = nn.ModuleList([
            SumAggregatorWithDropout(config.batch_size, config.emb_dim, 
                                     dropout=config.dropout,
                                     act=F.relu if i < config.n_iter-1 else torch.tanh)
            for i in range(config.n_iter)
        ])
        self.to(device)

    def forward(self, user_indices, item_indices, masks):
        """
        user_indices: [batch]
        item_indices: [batch]
        masks: dict with 'neg_mask' and 'plabel_mask' [batch] (if used)
        """
        batch_size = user_indices.size(0)
        device = self.device

        # Fetch embeddings and apply dropout on user
        u_emb = self.user_emb(user_indices)       # [batch, dim]
        u_emb = F.dropout(u_emb, p=self.config.dropout, training=self.training)

        # Multi-hop neighbor retrieval for items
        entities, relations = self.get_neighbors(item_indices, batch_size)
        # Embed neighbors
        ent_vecs = [self.entity_emb(e.to(device)) for e in entities]       # list of [batch, #nodes, dim]
        rel_vecs = [self.relation_emb(r.to(device)) for r in relations]     # list of [batch, #nodes, dim]

        # GNN aggregation
        x = ent_vecs  # initial item vectors per hop
        for i in range(self.config.n_iter):
            aggregator = self.aggregators[i]
            next_x = []
            # For each depth hop, aggregate (like TensorFlow loop)
            for hop in range(self.config.n_iter - i):
                self_vecs = x[hop]
                neigh_vecs = x[hop+1].view(batch_size, -1, self.config.neighbor_sample_size, self.emb_dim)
                neigh_rels = rel_vecs[hop].view(batch_size, -1, self.config.neighbor_sample_size, self.emb_dim)
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
        for _ in range(self.config.n_iter):
            neigh_e = self.adj_entity[entities[-1]].view(batch_size, -1)
            neigh_r = self.adj_relation[entities[-1]].view(batch_size, -1)
            entities.append(neigh_e)
            relations.append(neigh_r)
        return entities, relations

# <--- KGPL Co-Training Model ---
class KGPLCOT(nn.Module):
    def __init__(self, cfg, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, path_list_dict, train_data, eval_data, device):
        super().__init__()
        _cfg = copy.deepcopy(cfg)
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

        users_f, items_f, labels_f = log_f[:,0], log_f[:,1], log_f[:,2].float()
        users_g, items_g, labels_g = log_g[:,0], log_g[:,1], log_g[:,2].float()

        # Train model_f on g's data
        scores_f = self.model_f(users_g.to(self.device), items_g.to(self.device))
        loss_f = F.binary_cross_entropy_with_logits(scores_f, labels_g.to(self.device))
        optimizer_f.zero_grad()
        loss_f.backward()
        optimizer_f.step()

        # Train model_g on f's data
        scores_g = self.model_g(users_f.to(self.device), items_f.to(self.device))
        loss_g = F.binary_cross_entropy_with_logits(scores_g, labels_f.to(self.device))
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