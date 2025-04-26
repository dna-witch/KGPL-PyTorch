class KGPL_COT(object):
    def __init__(
        self,
        cfg,
        n_user, n_item, n_entity, n_relation,
        adj_entity, adj_relation, path_list_dict,
        train_data, eval_data, eval_mode=False
    ):
        self.cfg = cfg
        _cfg = copy.deepcopy(cfg)

        # Create two models
        self.f = KGPL_STUDENT(
            _cfg,
            n_user, n_entity, n_relation,
            adj_entity, adj_relation, path_list_dict,
            name="f", eval_mode=eval_mode,
        )
        self.g = KGPL_STUDENT(
            _cfg,
            n_user, n_entity, n_relation,
            adj_entity, adj_relation, path_list_dict,
            name="g", eval_mode=eval_mode,
        )
        if not eval_mode:
            _pos_inds = train_data[:, 2] == 1
            _pos_inds_ev = eval_data[:, 2] == 1
            self.f.set_item_candidates(
                n_user,
                n_item,
                train_data[_pos_inds],
                eval_data[_pos_inds_ev],
                path_list_dict,
            )

            # copy data from f to g
            self.g.n_item = self.f.n_item
            self.g.all_items = self.f.all_items
            self.g.all_users = self.f.all_users
            self.g.item_freq = self.f.item_freq
            self.g.neg_c_dict_user = self.f.neg_c_dict_user
            self.g.neg_c_dict_item = self.f.neg_c_dict_item
            self.g.user_seed_dict = self.f.user_seed_dict
            self.g.item_dist_dict = self.f.item_dist_dict

    def get_feed_dict(self, data, start, end, epoch=0, eval_mode=False, sess=None):
        if not eval_mode:
            feed_dict = {"data": data, "start": start, "end": end, "epoch": epoch}
        else:
            feed_dict = self.f.get_feed_dict(data[start:end], eval_mode=True, sess=sess)
        return feed_dict

    def split_data(self, data):
        N = len(data) // 2
        return data[:N], data[N:]

    def get_swap_feed_dict(self, sess, meta_feed_dict):
        fd = meta_feed_dict

        # Split meta feed-dict
        s, e = fd["start"], fd["end"]
        log_f, log_g = self.split_data(fd["data"][s:e])

        # Construct feed-dicts augmented with pseudo-labels
        f_fd = self.f.get_feed_dict(log_f, epoch=fd["epoch"], eval_mode=False, sess=sess)
        g_fd = self.g.get_feed_dict(log_g, epoch=fd["epoch"], eval_mode=False, sess=sess)

        # Target entries of a feed-dict
        fd_entries = [
            "user_indices",
            "item_indices",
            "labels",
            "plabel_mask",
            "neg_mask",
            "dropout_rate",
        ]

        # Exchange feed-dicts
        f_fd_train, g_fd_train = {}, {}
        for ky in fd_entries:
            f_fd_train[getattr(self.f, ky)] = g_fd[getattr(self.g, ky)]
            g_fd_train[getattr(self.g, ky)] = f_fd[getattr(self.f, ky)]
        return f_fd_train, g_fd_train

    def train(self, sess, meta_feed_dict):
        # Get augmented and swapped feed-dicts
        f_fd_train, g_fd_train = self.get_swap_feed_dict(sess, meta_feed_dict)

        # Train two models
        f_results = self.f.train(sess, f_fd_train)
        g_results = self.g.train(sess, g_fd_train)
        
        # Aggregate summaries
        ret_dict = {}
        for name, res in zip(["f_", "g_"], [f_results, g_results]):
            ret_dict.update({name + k: v for k, v in res.items()})
        return ret_dict

    def get_scores(self, sess, users, items, get_emb=False):
        # For evaluation, use only model f
        N = self.f.cfg.optimize.batch_size
        scores = sess.run(
            self.f.scores_normalized,
            {
                self.f.user_indices: users[:N],
                self.f.item_indices: items[:N],
                self.f.dropout_rate: 0.0,
            },
        )
        return items, scores

    def eval(self, sess, feed_dict):
        N = self.f.cfg.optimize.batch_size
        scores = sess.run(
            self.f.scores_normalized,
            {
                self.f.user_indices: feed_dict[self.f.user_indices][:N],
                self.f.item_indices: feed_dict[self.f.item_indices][:N],
                self.f.dropout_rate: 0.0,
            },
        )
        labels = feed_dict[self.f.labels]
        raw_scores = scores.copy()
        try:
            auc = roc_auc_score(y_true=labels, y_score=scores)
        except ValueError:
            auc = np.nan
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1, labels, raw_scores