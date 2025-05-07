from torch.utils.data import Dataset, DataLoader
import random
import torch
import numpy as np
import pickle
from collections import defaultdict
from .functions import *


class KGPLDataset(Dataset):
  def __init__(self, ratings):
    super().__init__()
    self.ratings = ratings
    # self.users = torch.arange(len(torch.unique(self.ratings[:,0])))
    # self.items = torch.arange(len(torch.unique(self.ratings[:,1])))
    self.users = torch.unique(self.ratings[:,0])
    self.items = torch.unique(self.ratings[:,1])
    self.n_user = len(self.users)
    self.n_item = len(self.items)

  def sample_positive(self, user):
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
  def __init__(self, ratings):
    super().__init__(ratings)
    self.user_seed_dict = defaultdict(set)
    self.item_dist_dict = {}
    # self._fix_set_for_positives()

  # def _fix_set_for_positives(self):
  #   '''
  #   Train set needs at least one positive example per user.
  #   '''
  #   # Step 1: Get unique groups
  #   groups = self.ratings[:, 0].unique()

  #   # Step 2: Find valid groups
  #   valid_groups = []
  #   for g in groups:
  #       # mask for current group
  #       mask = self.ratings[:, 0] == g
  #       # check if any third column > 0
  #       if (self.ratings[mask][:, 2] > 0).any():
  #           valid_groups.append(g)
  #   valid_groups = torch.Tensor(valid_groups)
  #   self.ratings = self.ratings[(self.ratings[:, 0][:, None] == valid_groups).any(dim=1)]
  #   # fix counts
  #   self.users = torch.unique(self.ratings[:,0])
  #   self.items = torch.unique(self.ratings[:,1])
  #   self.n_user = len(self.users)
  #   self.n_item = len(self.items)

  def sample_negative(self, user):
      seen = torch.unique(self.ratings[(self.ratings[:,0] == user)][:,1])
      while True:
          item = random.choice(self.items)
          if item not in seen:
              return item #torch.tensor(item)

  def sample_pseudo_label(self, user):
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
  base_data_path = 'KGPL-PyTorch/data/'
  def __init__(self, exp_name, cfg):
    self.exp_name = exp_name
    self.cfg = cfg
    # set paths
    # pathlist_path: data/music/path_list_6_32.pkl
    # adj_entity_path: data/music/adj_entity_6_32.npy
    # adj_relation_path: data/music/adj_relation_6_32.npy

    # pl_save_path = hydra.utils.to_absolute_path(
    #     str(
    #         save_dir / f"path_list_{cfg.lp_depth}_{cfg.num_neighbor_samples}.pkl"
    #     )
    # )
    # adje_save_path = hydra.utils.to_absolute_path(
    #     str(save_dir / f"adj_entity_{cfg.lp_depth}_{cfg.num_neighbor_samples}")
    # )
    # adjr_save_path = hydra.utils.to_absolute_path(
    #     str(save_dir / f"adj_relation_{cfg.lp_depth}_{cfg.num_neighbor_samples}")
    # )

    
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
    #split dataset
    n_ratings = len(ratings)
    split_indices = torch.randperm(n_ratings)[:int(n_ratings * split_ratio)]
    splitted_data = ratings[split_indices]
    rest_data = ratings[~torch.isin(torch.arange(n_ratings), split_indices)]
    #create new sets of ratings
    return rest_data, splitted_data

  def train_val_test_split(self):
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
    _freq = Counter(seq)
    for i in all_candidates:
        if i not in _freq:
            _freq[i] += 1
    freq = [_freq[i] for i in all_candidates]
    return dict(zip(all_candidates, freq))

  #### THIS IS THE BIG ONE #####
  def set_item_candidates(self):
      """
      Construct the sampling distrbiutions for negative/pseudo-labelled instances for each user
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
    users, pos_items, neg_items, pseudo_labels = zip(*batch)
    return (
        torch.stack(users),
        torch.stack(pos_items),
        torch.stack(neg_items),
        torch.stack(pseudo_labels),
    )

  def create_dataloaders(self):
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
