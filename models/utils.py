import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

def grouper(n, iterable, squash=None):
    it = iter(iterable)
    while True:
        if squash:
            chunk = [
                [None if (j != 0 and i in squash) else el[i] for i in range(len(el))]
                for j, el in enumerate(itertools.islice(it, n))
            ]
        else:
            chunk = list(itertools.islice(it, n))

        if not chunk:
            return
        elif len(chunk) != n:
            chunk += [None] * (n - len(chunk))
        yield chunk

def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        users, items, labels, neg_mask, plabel_mask = batch
        users = users.to(device); items = items.to(device)
        labels = labels.to(device); neg_mask = neg_mask.to(device); plabel_mask = plabel_mask.to(device)

        optimizer.zero_grad()
        scores = model(users, items)  # [batch]
        # Compute losses as above
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        loss_all = loss_fn(scores, labels)
        loss_obs = (loss_all * (1 - neg_mask) * (1 - plabel_mask)).mean()
        loss_neg = (loss_all * neg_mask).mean()
        loss_pl  = (loss_all * plabel_mask).mean()
        loss = loss_obs + loss_neg + loss_pl

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def compute_reachable_items(users, user_seed_dict, path_dict, item_freq, power):
    """
    For each user, find reachable items via KG paths and compute a pseudo-label distribution.
    This is adapted from compute_reachable_items_.
    Returns: dict of user -> (item_list, cumulative_probs)
    """
    reachable = {}
    for user in users:
        seed_items = user_seed_dict.get(user, [])
        # Count reachable paths from each seed
        dst = Counter()
        for item in seed_items:
            dst.update(path_dict.get(item, {}))
        if not dst: 
            continue
        udst = np.array(list(dst.keys()))
        freq = np.array(list(dst.values())) ** power
        # Remove seeds from candidates
        mask = ~np.isin(udst, list(seed_items))
        udst = udst[mask]
        freq = freq[mask]
        # Add 'unreachable' items with small weight
        all_items = np.array(list(item_freq.keys()))
        unreachable = np.setdiff1d(all_items, udst, assume_unique=True)
        freq = np.concatenate([freq, np.ones(len(unreachable)) * 0.5])
        udst = np.concatenate([udst, unreachable])
        # Sort and make CDF
        order = np.argsort(freq)
        udst = udst[order]; freq = freq[order]
        cdf = (freq / freq.sum()).cumsum()
        reachable[user] = (udst.tolist(), cdf.tolist())
    return reachable
