def get_user_record(data, is_train):
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
