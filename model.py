# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from abc import ABC, abstractmethod
import pandas as pd
from tqdm import tqdm

def kgpl_loss(pos_scores, neg_scores, pseudo_scores):
    """
    Computes the KGPL binary cross-entropy loss for positive, negative, and pseudo-labeled samples (interactions).
    The binary cross-entropy loss function for semi-supervised learning is implemented with three parts:
        - Positive pairs: supervised positive labels (1).
        - Negative pairs: sampled unobserved items treated as negative labels (0).
        - Pseudo-labeled pairs: unobserved items assigned soft labels from the model.
    
    This corresponds to the loss function in Eq. (4) of the KGPL paper.
    
    Args:
        pos_scores (torch.Tensor): Scores for positive user-item interactions.
        neg_scores (torch.Tensor): Scores for negative user-item interactions.
        pseudo_scores (torch.Tensor): Scores for pseudo-labeled user-item interactions.

    Returns:
        loss (torch.Tensor): The combined binary cross-entropy loss across all three sets of samples.
    """
    # BCE loss like TensorFlow version
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    pseudo_labels = torch.ones_like(pseudo_scores)
    loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels) + \
           F.binary_cross_entropy_with_logits(neg_scores, neg_labels) + \
           F.binary_cross_entropy_with_logits(pseudo_scores, pseudo_labels)
    return loss

LAYER_IDS = {}  # Dictionary to keep track of layer IDs

def get_layer_id(layer_name=''):
    """
    Returns a unique ID for each layer based on its name. 
    Used to assign a layer ID to each instance of an Aggregator subclass.

    Args:
        layer_name (str): Base name of the layer.
    
    Returns:
        int: The incremented ID for the given layer name.
    """
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]

class Aggregator(nn.Module, ABC):
    """
    Abstract base class for aggregation layers used in the knowledge-graph aware graph neural network (GNN).

    Used for learning node representations in the knowledge graph by aggregating neighbor
    embeddings across sampled relations.

    Args:
        batch_size (int): Number of users or items in a batch.
        dim (int): Dimension size of the embeddings.
        dropout (float): Dropout rate.
        act (callable): Activation function.
        name (str): Name for the layer instance.

    Attributes:
        name (str): Name for the layer instance.
        dropout (float): Dropout rate.
        act (callable): Activation function.
        batch_size (int): Number of users or items in a batch.
        dim (int): Dimension size of the embeddings.

    Methods:
        forward(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None):
            Forward pass for the aggregator layer.
        _call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None):
            Abstract method to be implemented by subclasses for specific aggregation logic.
        _mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings):
            Mixes neighbor vectors based on their relations and user embeddings.
    """
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

class LabelAggregator(Aggregator):
    """
    Aggregates the soft(weak) pseudo-labels from neighbor nodes in the knowledge graph.
    This can be used to enhance pseudo-labeling by refining the labels based on the neighbors' labels.
    Implements label smoothing via neighbor averaging, similar to in KGNN-LS.
    """
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

class SumAggregatorWithDropout(Aggregator):
    """
    Aggregates neighbor embeddings with dropout and applies a linear transformation followed by an activation function (default is ReLU).

    This aggregator is used in the KGPL model to combine an entity's embedding with its neighbors' embeddings,
    incorporating dropout for regularization and a linear transformation to project the aggregated embedding
    into the desired space.

    Args:
        batch_size (int): Number of entities in the batch.
        dim (int): Dimensionality of the input and output embeddings.
        dropout (float): Dropout rate applied to the neighbor embeddings for regularization.
        act (callable): Activation function applied after the linear transformation.
        name (str): Optional name for the aggregator instance.
    """
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

# <------- Sum Aggregator with Dropout ------->
class SumAggregator(nn.Module):
    """
    Computes attention-weighted sums of neighbor embeddings to generate refined representations.

    In the KGPL model, this aggregator is used to combine embeddings of neighboring entities,
    weighting each neighbor's contribution based on its relevance to the target entity (weighted average). This process
    enhances the representation of an entity by incorporating contextual information from its neighbors.

    Args:
        dim (int): Dimensionality of the input and output embeddings.
        dropout (float): Dropout rate applied to the attention weights for regularization.
        act_fn (callable): Activation function applied to the aggregated output.
    """
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
    """
    Graph Neural Network (GNN) student model for use in the Knowledge Graph Pseudo-Labeling (KGPL) framework.
    This model is designed to learn user and item embeddings by recursively aggregating information from the Knowledge Graph (KG) neighbors.
    This is a student model that is trained using the co-training approach.

    Args:
        n_user (int): Number of users.
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.
        adj_entity (list): Adjacency list for entities.
        adj_relation (list): Adjacency list for relations.
        config (dict): Configuration dictionary containing model parameters and settings.
        device (torch.device): Device to run the model on (CPU or GPU).

    Attributes:
        device (torch.device): Device to run the model on.
        config (dict): Configuration dictionary containing model parameters and settings.
        adj_entity (torch.Tensor): Adjacency list for entities as a tensor.
        adj_relation (torch.Tensor): Adjacency list for relations as a tensor.
        user_emb (nn.Embedding): Embedding layer for users.
        entity_emb (nn.Embedding): Embedding layer for entities.
        relation_emb (nn.Embedding): Embedding layer for relations.
        emb_dim (int): Dimension of the embeddings.
        aggregators (nn.ModuleList): List of aggregator layers for multi-hop aggregation.

    Methods:
        forward(user_indices, item_indices, masks=None):
            Forward pass for the model, computes scores for user-item pairs.
        get_neighbors(items, batch_size):
            Retrieves multi-hop neighbor lists (entities and relations) for the given items.
    """
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
        Forward pass for the model, computes scores for user-item pairs.

        Args:
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

        Args:
            items (torch.Tensor or torch.LongTensor): List of item indices.
            batch_size (int): Size of the batch for multi-hop neighbor expansion.

        Returns:
            tuple: (entities, relations) per hop as lists of tensors.
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
    """
    Co-training model for the Knowledge Graph Pseudo-Labeling (KGPL) framework. 
    This class consists of two student models (model_f and model_g). Each model's predictions 
    are used as pseudo-labels for training the other model.
    The co-training process allows the models to learn from each other's predictions,
    enhancing the robustness and performance of the recommender.

    This class implements the co-training algorithm as described in section 3.5 of the KGPL paper.

    Args:
        cfg (dict): Configuration dictionary containing model parameters and settings.
        n_user (int): Number of users.
        n_item (int): Number of items.
        n_entity (int): Number of entities.
        n_relation (int): Number of relations.
        adj_entity (list): Adjacency list for entities.
        adj_relation (list): Adjacency list for relations.
        path_list_dict (dict): Dictionary containing paths for the knowledge graph.
        train_data (torch.Tensor): Training data tensor.
        eval_data (torch.Tensor): Evaluation data tensor.
        device (torch.device): Device to run the model on (CPU or GPU).

    Attributes:
        model_f (KGPLStudent): First student model.
        model_g (KGPLStudent): Second student model.
        device (torch.device): Device to run the model on.
    """
    def __init__(self, cfg, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, path_list_dict, train_data, eval_data, device):
        super().__init__()
        _cfg = deepcopy(cfg)
        self.device = device

        self.model_f = KGPLStudent(n_user, n_entity, n_relation, adj_entity, adj_relation, _cfg, device)
        self.model_g = KGPLStudent(n_user, n_entity, n_relation, adj_entity, adj_relation, _cfg, device)

        # Placeholder for item candidates if needed later
        # Example: self.item_candidates = {user: [items]}

    def split_data(self, data):
        """ 
        Splits the input data into two halves for co-training.
        Used to alternate training between model_f and model_g.

        Args:
            data (torch.Tensor): Input data tensor.
        
        Returns:
            tuple: Two halves of the input data.

        Note:
            The data is not shuffled before splitting.
        """
        mid = len(data) // 2
        return data[:mid], data[mid:]

    def forward(self, batch_data):
        """ 
        Not supported for co-training.
        Use train_step() instead for alternating co-training updates.
        """
        # Forward pass is not typically called for Co-Training directly
        raise NotImplementedError("Use train_step for co-training")

    def train_step(self, batch_data, optimizer_f, optimizer_g):
        """
        Implements Algorithm 1 and Eq. (8) from the KGPL paper.
        
        Performs a co-training update step:
            - model_f is trained on labels generated by model_g
            - model_g is trained on labels generated by model_f 

        Args:
            batch_data (torch.Tensor): Batch of (user, observed item, negative item, item for pseudo-labeling).
            optimizer_f (torch.optim.Optimizer): Optimizer for model_f.
            optimizer_g (torch.optim.Optimizer): Optimizer for model_g.
        
        Returns:
            dict: Dictionary containing the loss values for both models.
        """
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
        """ 
        Predicts user-item scores using one of the student models. Typically, model_f is used for evaluation.
        
        Args:
            model_name (str): Name of the model to use for evaluation ('f' or 'g').
            users (torch.Tensor): User indices.
            items (torch.Tensor): Item indices.
        
        Returns:
            scores (torch.Tensor): Predicted scores for the user-item pairs.
        """
        # Evaluate only model_f typically
        model = self.model_f if model_name == 'f' else self.model_g
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(users.to(self.device), items.to(self.device)))
        return scores.cpu()


# <------- Function to Run Co-Training (used in notebook) ------->
def run_cotrain(model, train_loader, val_loader, test_loader, cfg, device, num_epochs=100):
    '''
    Run the CoTrain algorithm with the given model, data loaders, configuration, and device, using the "music" experiment.
    Optimizers are defined here with epsilon = 1e-7 to match tensorflow default.

    Inputs:
    - model: The CoTrain model to be trained.
    - train_loader: The data loader for the training set.
    - val_loader: The data loader for the validation set.
    - test_loader: The data loader for the test set.
    - cfg: The configuration dictionary.
    - device: PyTorch device
    - num_epochs: Training epochs

    Returns:
    - pd.DataFrame: A DataFrame containing the training loss and validation metrics for each epoch.
    '''
    optimizer_f = torch.optim.Adam(
        model.model_f.parameters(),
        lr=cfg['optimize']['lr'],
        eps=1e-7,
      )
    optimizer_g = torch.optim.Adam(
        model.model_g.parameters(),
        lr=cfg['optimize']['lr'],
        eps=1e-7,
      )
    iter_per_epoch = cfg['optimize']['iter_per_epoch']

    dataframe_rows = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}")
        model.train()
        total_loss_f, total_loss_g = 0.0, 0.0

        train_iter = iter(train_loader)  # re-initialize iterator every epoch

        for i in tqdm(range(iter_per_epoch)):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            users, pos, neg, pseudo = [b.to(device) for b in batch]
            batch_data = torch.stack([users, pos, neg, pseudo], dim=1)

            losses = model.train_step(batch_data, optimizer_f, optimizer_g)
            total_loss_f += losses['loss_f']
            total_loss_g += losses['loss_g']

        avg_loss_f = total_loss_f / iter_per_epoch
        avg_loss_g = total_loss_g / iter_per_epoch
        print(f"Train Loss - f: {avg_loss_f:.4f}, g: {avg_loss_g:.4f}")

        print("Evaluating model_f:")

        row = pd.DataFrame(
            {"Epoch": epoch,
             "Train Loss F": avg_loss_f,
             "Train Loss G": avg_loss_g
             }, index=[0])

        model.eval()

        print("Validation:")
        val_row = run_topk_eval(
          model=model,
          cfg = cfg,
          train_data=music.train_dataset.ratings.numpy(),
          eval_data=music.val_dataset.ratings.numpy(),
          test_data=music.test_dataset.ratings.numpy(),
          n_items=music.n_item,
          device=device,
          test_mode=False  # or True for test set
        ).add_prefix("Validation ")

        row = row.join(val_row)

        dataframe_rows.append(row)
    return pd.concat(dataframe_rows, ignore_index=True)