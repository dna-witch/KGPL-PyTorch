from torch.utils.data import Dataset, DataLoader
import random
import torch
import numpy as np
import pickle
from collections import defaultdict
from .functions import *


class KGPLDataset(Dataset):
  """
    A PyTorch Dataset class for handling knowledge graph-based pseudo-labelling (KGPL) data.

    Attributes:
      ratings (torch.Tensor): A tensor containing user-item interaction data in the format [user, item, rating].
      users (torch.Tensor): A tensor of unique user IDs extracted from the ratings data.
      items (torch.Tensor): A tensor of unique item IDs extracted from the ratings data.
      n_user (int): The total number of unique users.
      n_item (int): The total number of unique items.
      
    Methods:
      sample_positive(user):
        Samples a positively rated item for a given user.
      __len__():
        Returns the number of unique users in the dataset.
      __getitem__(index):
        Retrieves a user-item pair for evaluation purposes. 
        The item is sampled based on the user's positive ratings.
  """
  def __init__(self, ratings):
    super().__init__()
    self.ratings = ratings
    self.users = torch.unique(self.ratings[:,0])
    self.items = torch.unique(self.ratings[:,1])
    self.n_user = len(self.users)
    self.n_item = len(self.items)

  def sample_positive(self, user):
      """
      Samples a positively rated item for a given user.
      This is done by filtering the ratings data for the specified user
      and selecting an item that has a positive rating (rating >= 1).
      Otherwise, it raises an error.
      """
      choice_seq = self.ratings[(self.ratings[:,0] == user) & (self.ratings[:,2]) >= 1][:,1]
      if choice_seq.numel() == 0:
        print('Something went wrong.')
      return random.choice(choice_seq)

  def __len__(self):
      return self.n_user

  def __getitem__(self, index):
      '''
      This is getitem for the evaluation set.
      It will be overridden for the training set.
      '''
      user = self.users[index]
      item = self.sample_positive(user)
      return user, item

class KGPLTrainDataset(KGPLDataset):
  """ 
  A subclass of KGPLDataset specifically designed for training purposes.
  It includes additional functionality for sampling negative items and pseudo-labels.

  Attributes: 
    user_seed_dict (defaultdict): A dictionary mapping each user to a set of items they have interacted with.
    item_dist_dict (dict): A dictionary mapping each user to a tuple containing the reachable items and their corresponding distribution.
    
  Methods:
    sample_negative(user):
      Samples a negative item for a given user that they have not interacted with.
    sample_pseudo_label(user):
      Samples a pseudo-label for a given user based on the distribution of reachable items.
  """
  def __init__(self, ratings):
    super().__init__(ratings)
    self.user_seed_dict = defaultdict(set)
    self.item_dist_dict = {}

  def sample_negative(self, user):
    """
    Samples a negative item for a given user that they have not interacted with.
    This is done by randomly selecting an item from the set of all items 
    and checking if it is not in the user's seen items.

    Args:
      user: A user ID for which to sample a negative item.
    """
    seen = torch.unique(self.ratings[(self.ratings[:,0] == user)][:,1])
    while True:
        item = random.choice(self.items)
        if item not in seen:
          return item #torch.tensor(item)

  def sample_pseudo_label(self, user):
    """
    Samples a pseudo-label for a given user based on a precomputed item distribution.

    Args:
      user: A key used to retrieve the item distribution and cumulative probabilities.

    Returns:
      pl (torch.Tensor): A pseudo-label sampled from the user's item distribution.
    """
    udst, F = self.item_dist_dict[user.item()]
    r = random.random()
    F = torch.as_tensor(F)
    idx = torch.searchsorted(F, r, right=True)
    pl = udst[idx]
    if type(pl) == np.int64:
        pl = torch.tensor(pl)
    return pl

  def __getitem__(self, idx):
    user = self.users[idx]
    pos_item = self.sample_positive(user)
    neg_item = self.sample_negative(user)
    pseudo_label = self.sample_pseudo_label(user)
    return user, pos_item, neg_item, pseudo_label

class KGPLExperiment():
  """
  A class to manage the KGPL experiment setup, including loading data 
  and splitting it into training, validation, and test sets.

  Attributes:
    exp_name (str): The name of the experiment.
    cfg (dict): Configuration settings for the experiment.
    data_path (str): Path to the data files for this experiment.
    adj_entity_path (str): Path to the entity adjacency file.
    adj_relation_path (str): Path to the relation adjacency file.
    ratings_path (str): Path to the ratings file.
    path_list_path (str): Path to the path list file.
    adj_entity (torch.Tensor): Entity adjacency matrix loaded from file.
    n_entity (int): Number of entities in the dataset.
    adj_relation (torch.Tensor): Relation adjacency matrix loaded from file.
    n_relation (int): Number of relations in the dataset.
    ratings (torch.Tensor): Ratings data loaded from file.
    path_list_dict (dict): Dictionary containing path lists loaded from file.
    dst_dict (dict): Dictionary containing distances between items.

  Methods:
    __init__(exp_name, cfg):
      Initializes the KGPLExperiment class with the given experiment name and configuration.
    _split_data(ratings, split_ratio=0.2):
      Splits the ratings data into training and test sets based on the specified ratio.
    train_val_test_split():
      Splits the data into training, validation, and test sets.
    _build_freq_dict(seq, all_candidates):
      Builds a frequency dictionary for the given sequence and candidates.
    set_item_candidates():
      Constructs sampling distributions for negative/pseudo-labelled instances for each user.
    train_collate_fn(batch):
      Collates a batch of training data into a format suitable for model input.
    create_dataloaders():
      Creates DataLoader objects for the training, validation, and test datasets.
  """

  base_data_path = 'KGPL-PyTorch/data/'
  def __init__(self, exp_name, cfg):
    self.exp_name = exp_name
    self.cfg = cfg

    self.data_path = KGPLExperiment.base_data_path + exp_name + '/'
    self.adj_entity_path = self.data_path + f"adj_entity_{cfg['plabel_lp_depth']}_{cfg['model']['num_neighbor_samples']}.npy"
    self.adj_relation_path = self.data_path + f"adj_relation_{cfg['plabel_lp_depth']}_{cfg['model']['num_neighbor_samples']}.npy"
    self.ratings_path = self.data_path + 'ratings_final.npy'
    self.path_list_path = self.data_path + f"path_list_{cfg['plabel_lp_depth']}_{cfg['model']['num_neighbor_samples']}.pkl"

    # load from paths
    print('Loading Entity Adjacencies...')
    self.adj_entity = torch.from_numpy(np.load(self.adj_entity_path))
    self.n_entity = self.adj_entity.shape[0]
    print('Loading Relation Adjacencies...')
    self.adj_relation = torch.from_numpy(np.load(self.adj_relation_path))
    self.n_relation = len(np.unique(np.reshape(self.adj_relation, -1)))
    print('Loading Ratings...')
    self.ratings = torch.from_numpy(np.load(self.ratings_path))
    print('Loading Path List...')
    self.path_list_dict = pickle.load(open(self.path_list_path, 'rb'))
    print('Loading Distances...')
    self.dst_dict = setup_dst_dict(self.path_list_dict)

  @staticmethod
  def _split_data(ratings, split_ratio=0.2):
    """
    Splits the dataset into two subsets based on the given split ratio.
    This is used in `train_val_test_split` to create training and test datasets.
    The split is done randomly, and the function returns the remaining data and the split data.

    Args:
      ratings (torch.Tensor): The dataset to be split.
      split_ratio (float, optional): The proportion of the dataset to include in the split. Defaults to 0.2.

    Returns:
      tuple: A tuple containing the rest (remaining) data (torch.Tensor) and the split data (torch.Tensor).
    """
    #split dataset
    n_ratings = len(ratings)
    split_indices = torch.randperm(n_ratings)[:int(n_ratings * split_ratio)]
    splitted_data = ratings[split_indices]
    rest_data = ratings[~torch.isin(torch.arange(n_ratings), split_indices)]
    #create new sets of ratings
    return rest_data, splitted_data

  def train_val_test_split(self):
    """
    Splits the ratings data into training, validation, and test datasets.
    
    The method divides the ratings into exploratory and test sets, then further 
    splits the exploratory set into training and validation sets. It initializes 
    the corresponding datasets (training, validation, test) and sets user and item counts.

    Side Effects:
      - Initializes `train_dataset`, `val_dataset`, and `test_dataset` attributes.
      - Sets `n_user` and `n_item` attributes based on the training dataset.
      - Calls `set_item_candidates()` to configure item candidates for users.
    """
    exp_ratings, test_ratings = self._split_data(self.ratings)
    exp_ratings = exp_ratings[exp_ratings[:, 2]==1] #replaced function
    train_ratings, val_ratings = self._split_data(exp_ratings)
    self.train_dataset = KGPLTrainDataset(train_ratings)
    self.val_dataset = KGPLDataset(val_ratings)
    self.test_dataset = KGPLDataset(test_ratings)
    self.set_item_candidates()
    self.n_user = self.train_dataset.n_user
    self.n_item = self.train_dataset.n_item
    # return train_dataset, val_dataset, test_dataset
    # return train_ratings, val_ratings, test_ratings

  @staticmethod
  def _build_freq_dict(seq, all_candidates): # need all_candidates
    """
    Builds a frequency dictionary for a given sequence and a list of *all* possible candidates.

    Args:
      seq (iterable): The input sequence to calculate frequencies from.
      all_candidates (iterable): A list of all possible candidates to ensure they are included in the frequency dictionary.

    Returns:
      dict: A dictionary mapping each candidate in `all_candidates` to its frequency in `seq`.
    """
    _freq = Counter(seq)
    for i in all_candidates:
        if i not in _freq:
            _freq[i] += 1
    freq = [_freq[i] for i in all_candidates]
    return dict(zip(all_candidates, freq))

  #### THIS IS THE BIG ONE #####
  def set_item_candidates(self):
    """
    Constructs sampling distributions for negative and pseudo-labelled instances for each user.

    This method prepares the training data for pseudo-label learning by:
      1. Building frequency dictionaries for users and items from both training and validation sets.
      2. Computing a sampling distribution over items using the frequency distribution raised to a power (controlled by `plabel_neg_pn`).
      3. Creating a user-item interaction dictionary (`user_seed_dict`) to track positively interacted items per user.
      4. Generating a dictionary of reachable item distributions (`item_dist_dict`) for each user using path-based heuristics.

    The resulting distributions are used for:
      - Negative item sampling (via item frequency).
      - Pseudo-label generation (via path-based reachable item sampling).

    Side Effects:
      - Sets `self.user_seed_dict`: maps each user to their interacted items.
      - Sets `self.item_freq`: a cumulative distribution function over items for sampling.
      - Sets `self.train_dataset.item_dist_dict`: maps each user to a tuple of reachable items and probabilities.

    Notes:
      - This function internally uses `grouper()` and `compute_reachable_items_()` for chunked computation.
      - Optionally supports multiprocessing (commented out in current version).
    """
    train_data = self.train_dataset.ratings
    eval_data = self.val_dataset.ratings

    all_users = torch.unique(train_data[:,0])
    all_items = torch.unique(train_data[:,1])
    n_user = len(all_users)
    n_item = len(all_items)
    self.user_seed_dict = defaultdict(set)
    self.all_items = set(torch.arange(n_item))
    self.neg_c_dict_user = self._build_freq_dict(
        torch.concat([train_data[:, 0], eval_data[:, 0]]), all_users
    )

    print('Neg C Dict User:', len(self.neg_c_dict_user))

    self.neg_c_dict_item = self._build_freq_dict(
        np.concatenate([train_data[:, 1], eval_data[:, 1]]), all_items
    )

    print('Neg C Dict Item:', len(self.neg_c_dict_item))

    item_cands = tuple(self.neg_c_dict_item.keys())
    F = np.array(tuple(self.neg_c_dict_item.values())) ** self.cfg['plabel_neg_pn']
    sort_inds = np.argsort(F)
    item_cands = [item_cands[i] for i in sort_inds]
    F = F[sort_inds]
    F = (F / F.sum()).cumsum()
    self.item_freq = (item_cands, F)

    for u, i in tqdm(train_data[:, 0:2]):
        self.user_seed_dict[u.item()].add(i.item())

    item_dist_dict = {}
    src_itr = map(
        lambda iu: (
            all_users[iu].item(),
            tuple(self.user_seed_dict[all_users[iu].item()]),
            self.dst_dict,
            self.neg_c_dict_item,
            self.cfg['plabel_pl_pn'],
        ),
        range(len(all_users)),
    )

    grouped = grouper(self.cfg['plabel_chunk_size'], src_itr, squash=set([2, 3]))

    # --------- commented out multiprocessing --------
    # with mp.Pool(self.cfg.plabel_par) as pool:
    #     for idd in pool.imap_unordered(compute_reachable_items_, grouped):
    #         item_dist_dict.update(idd)
    #         print(idd)
    print('Populating item dist dict...')
    item_dist_dict = {}
    for group in tqdm(grouped):
        # print('Group sample:', group[0])
        idd = compute_reachable_items_(group)
        item_dist_dict.update(idd)
    self.train_dataset.item_dist_dict = item_dist_dict

  @staticmethod
  def train_collate_fn(batch):
    """
    Collates a batch of training data into a format suitable for model input.
    This function stacks the user, positive item, negative item, 
    and pseudo-label tensors into a single tensor for each type.

    Args:
      batch (list): A list of tuples, where each tuple contains user, positive item, negative item, and pseudo-label.
    """
    users, pos_items, neg_items, pseudo_labels = zip(*batch)
    return (
        torch.stack(users),
        torch.stack(pos_items),
        torch.stack(neg_items),
        torch.stack(pseudo_labels),
    )

  def create_dataloaders(self):
    """
    Creates DataLoader objects for the training, validation, and test datasets.
    The DataLoader objects are configured with the specified batch size, shuffling, and other parameters.
    """
    self.train_loader = DataLoader(
        self.train_dataset,
        batch_size=self.cfg['optimize']['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    self.val_loader = DataLoader(
        self.val_dataset,
        batch_size=self.cfg['optimize']['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    self.test_loader = DataLoader(
        self.test_dataset,
        batch_size=self.cfg['optimize']['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
