"""KGPL student model for knowledge graph completion and recommendation.
This model is a student model that learns from the teacher model (KGPL_COT)."""

import random
import itertools
import numpy as np
import tqdm
import copy
from collections import defaultdict, Counter
import multiprocessing as mp
from loguru import logger
import hydra

from sklearn.metrics import f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

# change according to your project structure and file names
from kapl import KaPLMixin
from aggregators import SumAggregatorWithDropout
from utils import grouper

class KGPL_STUDENT(KaPLMixin):
    def __init__(
        self,
        cfg,
        n_user, n_entity, n_relation,
        adj_entity, adj_relation, path_list_dict,
        name, eval_mode=False,
    ):
        self.n_user = n_user
        self.name = name
        self._parse_cfg(cfg, adj_entity, adj_relation)
        KaPLMixin.__init__(
            self,
            cfg,
            n_entity,
            n_relation,
            adj_entity,
            adj_relation,
            path_list_dict,
            eval_mode=eval_mode,
        )

        with tf.variable_scope(self.name) as scope:
            self._build_inputs()
            self._build_model(n_user, n_entity, n_relation)
            self._build_train(scope)

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_cfg(self, cfg, adj_entity, adj_relation):
        self.cfg = cfg
        self.batch_size = cfg.optimize.batch_size

        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name="user_indices")
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name="item_indices")
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name="labels")
        self.neg_mask = tf.placeholder(dtype=tf.float32, shape=[None], name="negative_mask")
        self.plabel_mask = tf.placeholder(dtype=tf.float32, shape=[None], name="plabel_mask")
        self.dropout_rate = tf.placeholder(dtype=tf.float32, name="dropout_rate")

    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.cfg.emb_dim],
            initializer=KGPL_STUDENT.get_initializer(),
            name="user_emb_matrix",
        )
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.cfg.emb_dim],
            initializer=KGPL_STUDENT.get_initializer(),
            name="entity_emb_matrix",
        )
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.cfg.emb_dim],
            initializer=KGPL_STUDENT.get_initializer(),
            name="relation_emb_matrix",
        )

        # [batch_size, dim]
        self.user_embeddings = tf.nn.dropout(
            tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices), 1 - self.dropout_rate
        )

        entities, relations = self.get_neighbors(self.item_indices)
        self.batch_entities = entities
        self.entity_indices = tf.concat([tf.reshape(es, [-1]) for es in entities], 0)
        self.entity_embeddings = tf.nn.embedding_lookup(self.entity_emb_matrix, self.entity_indices)
        self.relation_indices = tf.concat([tf.reshape(rs, [-1]) for rs in relations], 0)
        self.relation_embeddings = tf.nn.embedding_lookup(
            self.relation_emb_matrix, self.relation_indices
        )

        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.cfg.n_iter):
            neighbor_entities = tf.reshape(
                tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1]
            )
            neighbor_relations = tf.reshape(
                tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1]
            )
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]
        for i in range(self.cfg.n_iter):
            if i == self.cfg.n_iter - 1:
                aggregator = SumAggregatorWithDropout(
                    self.batch_size,
                    self.cfg.emb_dim,
                    self.dropout_rate,
                    act=tf.nn.tanh,
                    cfg=self.cfg,
                    name=f"agg_{i}",
                )
            else:
                aggregator = SumAggregatorWithDropout(
                    self.batch_size,
                    self.cfg.emb_dim,
                    self.dropout_rate,
                    act=tf.nn.leaky_relu,
                    cfg=self.cfg,
                    name=f"agg_{i}",
                )
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.cfg.n_iter - i):
                shape = [self.batch_size, -1, self.cfg.neighbor_sample_size, self.cfg.emb_dim]
                vector = aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                    user_embeddings=self.user_embeddings,
                    masks=None,
                )
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.cfg.emb_dim])

        return res, aggregators

    def _build_train(self, scope):
        """Implementation of the risk and optimizer.
        """
        # compute losses for all samples and then split those into positve, negative, and pseudo-labelled by using binary masks
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores)
        loss_obs = loss * (1 - self.neg_mask) * (1 - self.plabel_mask)
        loss_uobs = loss * self.neg_mask
        loss_pl = loss * self.plabel_mask

        # compute the risk
        self.obs_loss = tf.reduce_mean(loss_obs)
        self.uobs_loss = tf.reduce_mean(loss_uobs)
        self.pl_loss = tf.reduce_mean(loss_pl)
        self.base_loss = tf.reduce_sum(loss_obs + loss_uobs + loss_pl)

        self.loss = self.base_loss
        self.opt = tf.train.AdamOptimizer(self.cfg.optimize.lr)

        # to check gradients
        tvs = tf.trainable_variables(self.name)
        self.accum_vars_obs = [
            (tv.name, tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False))
            for tv in tvs
        ]
        self.accum_vars_uobs = [
            (tv.name, tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False))
            for tv in tvs
        ]
        self.accum_vars_pl = [
            (tv.name, tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False))
            for tv in tvs
        ]
        self.zero_ops = [tv.assign(tf.zeros_like(tv)) for _, tv in self.accum_vars_obs]
        self.zero_ops += [tv.assign(tf.zeros_like(tv)) for _, tv in self.accum_vars_uobs]
        self.zero_ops += [tv.assign(tf.zeros_like(tv)) for _, tv in self.accum_vars_pl]

        gvs_obs = self.opt.compute_gradients(self.obs_loss, tvs)
        gvs_uobs = self.opt.compute_gradients(self.uobs_loss, tvs)
        gvs_pl = self.opt.compute_gradients(self.pl_loss, tvs)

        self.accum_ops = [
            self.accum_vars_obs[i][1].assign_add(tf.math.abs(gv[0])) for i, gv in enumerate(gvs_obs)
        ]
        self.accum_ops += [
            self.accum_vars_uobs[i][1].assign_add(tf.math.abs(gv[0]))
            for i, gv in enumerate(gvs_uobs)
        ]
        self.accum_ops += [
            self.accum_vars_pl[i][1].assign_add(tf.math.abs(gv[0])) for i, gv in enumerate(gvs_pl)
        ]
        self.optimizer = self.opt.minimize(self.loss)

    def train(self, sess, feed_dict):
        _, loss, obs_loss, uobs_loss, pl_loss, _, grad_obs, grad_uobs, grad_pl = sess.run(
            [
                self.optimizer,
                self.base_loss,
                self.obs_loss,
                self.uobs_loss,
                self.pl_loss,
                self.accum_ops,
                [v for n, v in self.accum_vars_obs],
                [v for n, v in self.accum_vars_uobs],
                [v for n, v in self.accum_vars_pl],
            ],
            feed_dict,
        )
        grad_dict_obs = dict(zip([vname for vname, _ in self.accum_vars_obs], grad_obs))
        grad_dict_uobs = dict(zip([vname for vname, _ in self.accum_vars_uobs], grad_uobs))
        grad_dict_pl = dict(zip([vname for vname, _ in self.accum_vars_pl], grad_pl))
        user_grad_obs, entity_grad_obs = (
            grad_dict_obs[f"{self.name}/user_emb_matrix:0"],
            grad_dict_obs[f"{self.name}/entity_emb_matrix:0"],
        )
        user_grad_uobs, entity_grad_uobs = (
            grad_dict_uobs[f"{self.name}/user_emb_matrix:0"],
            grad_dict_uobs[f"{self.name}/entity_emb_matrix:0"],
        )
        user_grad_pl, entity_grad_pl = (
            grad_dict_pl[f"{self.name}/user_emb_matrix:0"],
            grad_dict_pl[f"{self.name}/entity_emb_matrix:0"],
        )
        return {
            "loss": loss,
            "obs_loss": obs_loss,
            "uobs_loss": uobs_loss,
            "pl_loss": pl_loss,
            "user_grad_obs": np.mean(np.sum(user_grad_obs, 1)),
            "entity_grad_obs": np.mean(np.sum(entity_grad_obs, 1)),
            "user_grad_uobs": np.mean(np.sum(user_grad_uobs, 1)),
            "entity_grad_uobs": np.mean(np.sum(entity_grad_uobs, 1)),
            "user_grad_pl": np.mean(np.sum(user_grad_pl, 1)),
            "entity_grad_pl": np.mean(np.sum(entity_grad_pl, 1)),
        }

    def get_feed_dict(self, data, epoch=0, eval_mode=False, sess=None):
        if not eval_mode:
            # Sample "1/3 batch-size" users 
            users = np.random.choice(
                self.all_users, self.cfg.optimize.batch_size // 3, replace=False
            )
            # Sample positive items
            items = [random.choice(tuple(self.user_seed_dict[u])) for u in users]

            # Sample pseudo-labelled items
            pl_users, pl_items, pl_labels = self._get_mini_batch_pl(sess, users)

            # Sample negative items
            seen_pair = set(zip(users, items)) | set(zip(pl_users, pl_items))
            cands, F = self.item_freq
            neg_items = []
            for u in users:
                while True:
                    j = cands[np.searchsorted(F, random.random())]
                    if j not in self.user_seed_dict[u] and j not in seen_pair:
                        break
                neg_items.append(j)

            # Create masks for positve, negative, pseudo-labelled instances in the mini-batch
            labels = [1] * len(items) + [0] * len(neg_items)
            plabel_mask = [0] * len(items) + [0] * len(neg_items) + [1] * len(pl_items)
            neg_mask = [0] * len(items) + [1] * len(neg_items) + [0] * len(pl_items)

            all_users = np.concatenate([users, users, pl_users])
            all_items = np.concatenate([items, neg_items, pl_items])

            feed_dict = {
                self.user_indices: all_users,
                self.item_indices: all_items,
                self.labels: np.concatenate([labels, pl_labels]),
                self.plabel_mask: plabel_mask,
                self.neg_mask: neg_mask,
                self.dropout_rate: self.cfg.dropout_rate,
            }
        else:
            feed_dict = {
                self.user_indices: data[:, 0],
                self.item_indices: data[:, 1],
                self.labels: data[:, 2],
                self.dropout_rate: 0.0,
            }
        return feed_dict
