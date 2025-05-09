import numpy as np
from numpy.random import default_rng, SeedSequence
import pickle
import joblib
from joblib import Parallel, delayed
import hydra
from collections import defaultdict
from pathlib import Path
from operator import itemgetter

global kg
global adj_entity
global adj_relation
global depth


def prepare_kg(kg_path: str) -> dict:
    """
    Prepares a knowledge graph (KG) from a numpy file and represents it as an undirected graph.
    This function loads a numpy file containing triples (head, relation, tail) and constructs
    a dictionary-based representation of the KG. Each entity in the KG is treated as a node,
    and each triple is treated as an undirected edge between two nodes with an associated relation.

    Args:
        kg_path (str): The file path to the numpy file containing the KG triples.
    Returns:
        dict: A dictionary representing the KG, where keys are entities (nodes) and values
              are lists of tuples. Each tuple contains a connected entity (node) and the
              relation between them.
    """
    global kg

    kg_np = np.load(kg_path)

    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(kg: dict) -> tuple:
    """
    Constructs adjacency matrices for entities and relations from a knowledge graph (KG).
    This function generates two adjacency matrices: one for entities (`adj_entity`) and one for 
    relations (`adj_relation`). Each entity in the KG is associated with its neighbors, and the 
    adjacency matrices are padded to ensure uniform dimensions.
    
    Args:
        kg (dict): A dictionary representing the knowledge graph where each key is an entity 
                   (int or hashable object) and the value is a list of tuples. Each tuple contains 
                   a neighboring entity (int) and the relation (int) connecting them.
    Returns:
        tuple: A tuple containing:
            - adj_entity (np.ndarray): A 2D numpy array where each row corresponds to an entity 
              and contains the indices of its neighboring entities. Rows are padded with -1 to 
              match the maximum number of neighbors.
            - adj_relation (np.ndarray): A 2D numpy array where each row corresponds to an entity 
              and contains the indices of the relations connecting it to its neighbors. Rows are 
              padded with -1 to match the maximum number of neighbors.
    Notes:
        - The function uses global variables `adj_entity` and `adj_relation` to store the 
          adjacency matrices, which are also returned as outputs.
        - Padding ensures that all rows in the adjacency matrices have the same length, determined 
          by the maximum number of neighbors for any entity in the KG.
    """
    
    global adj_entity
    global adj_relation

    max_len = np.max([len(kg[e]) for e in kg])
    adj_entity = []
    adj_relation = []
    for entity in kg:
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        sampled_indices = list(range(n_neighbors)) + [-1] * max(max_len - n_neighbors, 0)
        adj_entity.append(np.array([neighbors[i][0] for i in sampled_indices]))
        adj_relation.append(np.array([neighbors[i][1] for i in sampled_indices]))
    adj_entity = np.array(adj_entity, dtype=np.int64)
    adj_relation = np.array(adj_relation, dtype=np.int64)

    return adj_entity, adj_relation


def construct_adj_random(kg: dict, num_neighbor_samples: int, rng: np.random.Generator) -> tuple:
    """
    Constructs adjacency matrices for entities and relations by sampling neighbors.

    Args:
        kg (dict): Knowledge graph represented as a dictionary where keys are entities 
                   and values are lists of tuples (neighbor_entity, relation).
        num_neighbor_samples (int): Number of neighbors to sample for each entity.
        rng (np.random.Generator): Random number generator for sampling.
    Returns:
        tuple: Two numpy arrays:
            - adj_entity (np.ndarray): Adjacency matrix of sampled neighbor entities.
            - adj_relation (np.ndarray): Adjacency matrix of sampled relations.
    """
    global adj_entity
    global adj_relation

    n_entity = np.max(list(kg.keys())) + 1
    adj_entity = np.zeros((n_entity, num_neighbor_samples), dtype=np.int64)
    adj_relation = np.zeros((n_entity, num_neighbor_samples), dtype=np.int64)
    for entity in kg:
        entity = int(entity)
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= num_neighbor_samples:
            sampled_indices = rng.choice(
                list(range(n_neighbors)), size=num_neighbor_samples, replace=False
            )
        else:
            sampled_indices = rng.choice(
                list(range(n_neighbors)), size=num_neighbor_samples, replace=True
            )
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])
    return adj_entity, adj_relation


def get_all_items(rating_path: str) -> list:
    """
    Extracts a list of unique items from a NumPy array loaded from the given file path.

    Args:
        rating_path (str): Path to the .npy file containing the ratings data.

    Returns:
        list: A list of unique item IDs (second column of the array).
    """
    rating_np = np.load(rating_path)
    all_items = list(set(rating_np[:, 1]))
    return all_items


def get_paths(seed_item: int) -> tuple:
    """
    Performs a breadth-first search (BFS) to find paths starting from a seed item and 
    collects accepted neighbors based on the discovered paths.

    Args:
        seed_item (int): The starting item for pathfinding.

    Returns:
        tuple:
            - paths (set of tuple): A set of paths, where each path is represented as a tuple 
              of items starting from the seed item and ending at an item in `all_items`.
            - accepted_neighbors (defaultdict of set): A mapping where each key is an item, 
              and the value is a set of its accepted neighbors based on the discovered paths.

    Notes:
        - The BFS processes neighbors in a fixed, sorted order to ensure consistent results.
        - Paths are only added to the result if they end at an item in `all_items`.
        - The `depth` variable determines the maximum length of paths to explore.
        - The adjacency list `adj_entity` is used to find neighbors of each item.
        - The function avoids revisiting the immediate previous item in the path.
    """
    # path finding based on BFS
    accepted_neighbors = defaultdict(set)
    paths = set()
    queue = [[seed_item]]
    while len(queue) > 0:
        pt = queue.pop(0)
        e = pt[-1]
        next_e = sorted(set(adj_entity[e]))  # BFS queue processes neighbors in fixed, sorted order for consistent results
        for ne in next_e:
            if ne == -1 and (len(pt) > 1 and ne == pt[-2]):
                continue
            next_pt = pt[::] + [ne]
            if ne in all_items:
                paths.add(tuple(next_pt))
                for s, e in zip(next_pt[:-1], next_pt[0:]):
                    accepted_neighbors[s].add(e)
                    accepted_neighbors[e].add(s)
            if len(next_pt) < depth:
                queue.append(next_pt)
    return paths, accepted_neighbors


@hydra.main(config_path="../conf/preprocess.yaml")
def main(cfg):
    """
    Main function to preprocess data for path-based link prediction.
    
    Args:
        cfg (object): Configuration object containing the following attributes:
            - lp_depth (int): Depth of the link prediction paths.
            - rating_path (str): Path to the rating file.
            - kg_path (str): Path to the knowledge graph file.
            - num_neighbor_samples (int): Number of neighbor samples for adjacency matrix construction.
            - dataset (str): Name of the dataset.
    Workflow:
        1. Reads configuration and initializes global variables.
        2. Loads rating data and extracts all items.
        3. Sets up reproducibility using a fixed random seed.
        4. Prepares the knowledge graph and constructs adjacency matrices for entities and relations.
        5. Performs path finding using Breadth-First Search (BFS) in parallel.
        6. Computes statistics on the number of paths found.
        7. Saves the path list, adjacency entity matrix, and adjacency relation matrix to disk.
    Outputs:
        - Path list saved as a pickle file.
        - Adjacency entity matrix saved as a NumPy file.
        - Adjacency relation matrix saved as a NumPy file.
    Notes:
        - The function uses Hydra for configuration management.
        - Parallel processing is performed using joblib's Parallel and delayed utilities.
    """
    global depth
    global all_items

    depth = cfg.lp_depth
    print(cfg.pretty())

    rating_path = hydra.utils.to_absolute_path(cfg.rating_path)
    all_items = get_all_items(rating_path)

    # <----- Reproducibility ----->
    ss = SeedSequence(2021)  # maybe add 2021 as seed in config file?
    # Spawn child seed sequences for each parallel process
    child_seeds = ss.spawn(len(all_items))
    rngs = [default_rng(seed) for seed in child_seeds]


    kg_path = hydra.utils.to_absolute_path(cfg.kg_path)
    kg = prepare_kg(kg_path)

    # contruct adjacency matrix
    adj_entity, adj_relation = construct_adj_random(kg, cfg.num_neighbor_samples, rngs[0])

    # path finding based on BFS
    results = Parallel(n_jobs=32, verbose=10, backend="multiprocessing")(
        [delayed(get_paths)(i) for i in all_items]
    )

    path_set_list = list(map(itemgetter(0), results))

    save_dir = Path("data") / cfg.dataset
    pl_save_path = hydra.utils.to_absolute_path(
        str(
            save_dir / f"path_list_{cfg.lp_depth}_{cfg.num_neighbor_samples}.pkl"
        )
    )
    adje_save_path = hydra.utils.to_absolute_path(
        str(save_dir / f"adj_entity_{cfg.lp_depth}_{cfg.num_neighbor_samples}")
    )
    adjr_save_path = hydra.utils.to_absolute_path(
        str(save_dir / f"adj_relation_{cfg.lp_depth}_{cfg.num_neighbor_samples}")
    )

    lens = []
    for ps in path_set_list:
        lens.append(len(ps))
    print("average number of paths:", np.average(lens))
    print("median number of paths:", np.median(lens))
    print("min number of paths:", np.min(lens))
    print("max number of paths:", np.max(lens))

    pickle.dump(dict(zip(all_items, path_set_list)), open(pl_save_path, "wb"))
    np.save(adje_save_path, adj_entity)
    np.save(adjr_save_path, adj_relation)


if __name__ == "__main__":
    main()
