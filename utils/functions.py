import itertools
from tqdm import tqdm
from collections import Counter
import numpy as np

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


def setup_dst_dict(path_list_dict):
        """
        Transform path representations:
        `list of nodes` to `dictionary of source to sink (dst_dict)`
        """
        print("Setting up dst dict...")
        dst_dict = {}
        for item in tqdm(path_list_dict):
            dst = []
            paths = path_list_dict[item]
            for p in paths:
                dst.append(p[-1])
            dst_dict[item] = Counter(dst)

        print("Start updating path info...")
        print("Path info updated.")
        return dst_dict

def compute_reachable_items_(args_list):
  """Construct the sampling distributions based on paths in KG.
  Args:
      args_list: list of list of arguments. Each arguments' list must contains;
      (1) user_id;
      (2) user's interacted item ids (seed items);
      (3) item-to-(item, #paths) dict found in the BFS (start and end points of some paths);
      (4) item-to-frequency dict;
      (5) power coefficient to control the skewness of sampling distributions
  Returns:
      dict in which (key, value) = (item list, np.array of sampling distribution).
      sampling distribution is transformed to CDF for fast sampling.
  """
  idd = {}
  _, _, dst_dict, item_freq, pn = args_list[0]
  for args in args_list:
      if args is None:
          continue
      user, seed_items, _, _, _ = args

      # print('User:', user)
      # print('Seed Items:', seed_items)

      # Collect user's reachable items with the number of reachable paths
      dst = Counter()
      for item in seed_items:
          if item in dst_dict:
              dst += dst_dict[item]

      if len(dst) != 0:
          # Unique reachable items for the user
          udst = np.array(tuple(dst.keys()))

          # Histogram of paths with power transform
          F = np.array(tuple(dst.values())) ** pn

          # Remove the seed (positve) items
          inds = ~np.isin(udst, seed_items)
          udst = udst[inds]
          F = F[inds]

          # Compute unreachable items and concat those to the end of item lists
          udst = set(udst)
          unreachable_items = [i for i in item_freq if i not in udst]
          udst = list(udst) + unreachable_items

          # For unreachable items, assume 0.5 virtual paths for full support
          F = np.concatenate([F, np.ones(len(unreachable_items)) * 0.5])

          # Transform histogram to CDF
          sort_inds = np.argsort(F)
          udst = [udst[i] for i in sort_inds]
          F = F[sort_inds]
          F = (F / np.sum(F)).cumsum()
          idd[user] = (udst, F)
  return idd
