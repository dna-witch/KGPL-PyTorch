import pandas as pd
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

def get_user_record(data, is_train):
    """
    Get user-item interaction history from the dataset.

    Args:
        data (np.ndarray): The dataset containing user-item interactions.
        is_train (bool): If True, consider all interactions; if False, only positive interactions.

    Returns:
        user_history_dict (dict): A dictionary where keys are user IDs and values are sets of item IDs.
    """
    user_history_dict = dict()
    for interaction in data:
        user, item, label = interaction
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def topk_eval_pytorch(
    model,
    user_list,
    train_record,
    test_record,
    all_items,
    k_list,
    device,
    model_name='f',
    batch_size=256,
    return_topk=False,
    sample_size=100
    ):
    """
    Performs Top-K evaluation for a given model using Precision@K and Recall@K metrics.

    For each user in `user_list`, the function:
      - Constructs a candidate item set by excluding training items and including ground-truth positives.
      - Scores all candidate items using the model.
      - Ranks the items and computes hit-based precision and recall for each value in `k_list`.

    Args:
        model (torch.nn.Module): The trained recommendation model.
        user_list (list): List of user IDs to evaluate.
        train_record (dict): Dictionary mapping user IDs to sets of items seen during training.
        test_record (dict): Dictionary mapping user IDs to sets of ground-truth positive items.
        all_items (list): Full list of item IDs.
        k_list (list): List of cutoff values K to compute Precision@K and Recall@K.
        device (torch.device): Device to run model inference on.
        model_name (str): Optional name for the model head used in evaluation.
        batch_size (int): Unused, placeholder for future batching.
        return_topk (bool): If True, also returns a ranking list of top-K predictions for each user.
        sample_size (int): Unused, placeholder for future item sampling.

    Returns:
        tuple:
            - avg_precision (list): Average Precision@K for each value in `k_list`.
            - avg_recall (list): Average Recall@K for each value in `k_list`.
            - ranking_list (list or None): Top-K ranked item lists per user (if `return_topk` is True).
    """
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ranking_list = []

    model.eval()
    with torch.no_grad():
        for user in tqdm(user_list, desc="Evaluating"):
            if user not in test_record:
                continue

            pos_items = test_record[user]
            user_train_items = train_record.get(user, set())

            # Negatives = items not seen in training
            candidates = list(set(all_items) - user_train_items)

            # Add positives back
            candidates += list(pos_items)

            # Get scores
            item_tensor = torch.tensor(candidates, device=device)
            user_tensor = torch.tensor([user] * len(candidates), device=device)

            scores = model.evaluate(model_name, user_tensor, item_tensor)
            _, topk_indices = torch.topk(scores, max(k_list))

            ranked_items = item_tensor[topk_indices].tolist()
            if return_topk:
                ranking_list.append([(i, i in pos_items) for i in ranked_items[:max(k_list)]])

            for k in k_list:
                top_k = set(ranked_items[:k])
                hit_count = len(top_k & pos_items)

                precision = hit_count / k
                recall = hit_count / len(pos_items)

                precision_list[k].append(precision)
                recall_list[k].append(recall)

    avg_precision = [np.mean(precision_list[k]) for k in k_list]
    avg_recall = [np.mean(recall_list[k]) for k in k_list]

    return avg_precision, avg_recall, ranking_list if return_topk else None

def run_topk_eval(
    model,
    cfg,
    train_data,
    eval_data,
    test_data,
    n_items,
    device,
    k_list=[1, 2, 5, 10, 20],
    # user_num=500,
    test_mode=False,
    batch_size=256
    ):
    """
    Wrapper function to run Top-K evaluation for the recommendation model.

    This function prepares user interaction records, selects a subset of users to evaluate,
    and calls `topk_eval_pytorch()` to compute Precision@K and Recall@K. It handles different 
    evaluation modes (validation or test) and prints out the results (Precision@K and Recall@K metrics).

    Args:
        model (torch.nn.Module): The trained recommendation model.
        cfg (dict): Experiment configuration dictionary. Must contain `evaluate.user_num_topk`.
        train_data (np.ndarray): Training set in [user, item, label] format.
        eval_data (np.ndarray): Validation set in the same format.
        test_data (np.ndarray): Test set in the same format.
        n_items (int): Total number of items in the dataset.
        device (torch.device): Device to run model inference on.
        k_list (list): List of K values for top-K evaluation.
        test_mode (bool): Whether to run in test mode (use eval+train as history, test as target).
        batch_size (int): Unused, placeholder for future support.

    Side Effects:
        - Prints Precision@K and Recall@K for all values in `k_list`.

    Returns:
        results (pd.DataFrame): DataFrame containing Precision@K and Recall@K metrics.
    """
    user_num = cfg['evaluate']['user_num_topk']
    if test_mode:
        train_record = get_user_record(np.vstack([train_data, eval_data]), True)
        test_record = get_user_record(test_data, False)
        user_pool = list(set(train_data[:, 0]) & set(test_record.keys()))
    else:
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(eval_data, False)
        user_pool = list(set(train_record.keys()) & set(test_record.keys()))

    if user_num and len(user_pool) > user_num:
        user_list = np.random.choice(user_pool, size=user_num, replace=False)
    else:
        user_list = user_pool

    all_items = list(range(n_items))

    precision, recall, _ = topk_eval_pytorch(
        model,
        user_list,
        train_record,
        test_record,
        all_items,
        k_list,
        device,
        model_name='f',
        batch_size=batch_size
    )

    print("\nTop-K Evaluation Results")
    for k, p, r in zip(k_list, precision, recall):
        print(f"K={k:>3} | Precision: {p:.4f} | Recall: {r:.4f}")

    # Create a dictionary to store the results in the desired format
    results = {}
    for k, p, r in zip(k_list, precision, recall):
        results[f'Precision@{k}'] = p
        results[f'Recall@{k}'] = r

    # Create a pandas dataframe with the results
    return pd.DataFrame(results, index=[0])

def cold_start_eval(model, exp, n_items, device):
    """ 
    Evaluates the recommendation model on cold start users.
    Cold start users are those who have less than or equal to `n_items` interactions in the training set.
    
    Args:
        model (torch.nn.Module): The trained recommendation model.
        exp (object): Experiment object containing datasets.
        n_items (int): Threshold for cold start users.
        device (torch.device): Device to run model inference on.
    
    Returns:
        pd.DataFrame: DataFrame containing Precision@K and Recall@K metrics for cold start users. 
            The metrics are calculated using the test dataset via the `run_topk_eval` function.
    """
    # Get unique values and their counts
    unique_users, counts = torch.unique(exp.train_dataset.ratings[:,0], return_counts=True)

    # Select values with count <= n_items
    cold_starters = unique_users[counts <= n_items]

    print(f'Cold Starters <= {n_items}:', len(cold_starters))

    cold_mask = torch.isin(exp.test_dataset.ratings[:, 0], cold_starters)

    cold_test_data = exp.test_dataset.ratings[cold_mask]

    return run_topk_eval(
          model=model,
          cfg = exp.cfg,
          train_data=exp.train_dataset.ratings.numpy(),
          eval_data=exp.val_dataset.ratings.numpy(),
          test_data=cold_test_data.numpy(),
          n_items=music.n_item,
          device=device,
          test_mode=True
        )
