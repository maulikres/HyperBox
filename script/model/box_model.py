import math
import os
import pickle
import random
from random import shuffle

import numpy as np
import tensorflow as tf

from reader import Reader


class BoxE:
    def __init__(self, params, corpus_type, work_dir):
        self.params = params
        self.alpha = params.alpha
        self.num_rel = 1
        self.bounded_norm = params.bounded_norm
        self.normed_bumps = params.normed_bumps
        self.fixed_width = params.fixed_width
        self.hard_size = params.hard_size
        self.total_size = params.total_size
        self.learnable_shape = params.learnable_shape
        self.corpus_type = corpus_type
        self.cwd = work_dir
        self.word_vectors = np.load(
            f"{self.cwd}/../../word_vectors_processed/{self.corpus_type}_word_vectors_processed.npy"
        )

    def setup_reader(self):
        self.reader = Reader(self.corpus_type)
        self.reader.read_triples()
        self.reader.set_batch_size(self.params.batch_size)
        self.reader.set_neg_samples(self.params.no_neg_samples)
        self.num_batch = self.reader.num_batch()
        self.num_ent = self.reader.num_ent()

    def setup_loader(self):
        self.loader = tf.compat.v1.train.Saver(self.var_list)

    def setup_saver(self):
        self.saver = tf.compat.v1.train.Saver(max_to_keep=0)

    def create_session(self):
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def load_session(self, itr):
        self.loader.restore(
            self.sess,
            f"{self.cwd}/BoxModel_"
            + self.corpus_type
            + "_weights/"
            + "/"
            + itr
            + ".ckpt",
        )

    def close_session(self):
        self.sess.close()

    def product_normalise(self, input_tensor, bounded_norm=True):
        step1_tensor = tf.abs(input_tensor)
        step2_tensor = step1_tensor + (10 ** -8)
        log_norm_tensor = tf.math.log(step2_tensor)
        step3_tensor = tf.reduce_mean(log_norm_tensor, axis=2, keepdims=True)
        norm_volume = tf.math.exp(step3_tensor)
        pre_norm_out = input_tensor / norm_volume
        if not bounded_norm:
            return pre_norm_out
        else:
            minsize_tensor = tf.minimum(
                tf.reduce_min(log_norm_tensor, axis=2, keepdims=True), -1
            )
            maxsize_tensor = tf.maximum(
                tf.reduce_max(log_norm_tensor, axis=2, keepdims=True), 1
            )
            minsize_ratio = -1 / minsize_tensor
            maxsize_ratio = 1 / maxsize_tensor
            size_norm_ratio = tf.minimum(minsize_ratio, maxsize_ratio)
            normed_tensor = log_norm_tensor * size_norm_ratio
            return tf.exp(normed_tensor)

    def create_train_placeholders(self):
        self.ph = tf.compat.v1.placeholder(tf.int32, [None])
        self.pt = tf.compat.v1.placeholder(tf.int32, [None])
        self.nh = tf.compat.v1.placeholder(tf.int32, [None])
        self.nt = tf.compat.v1.placeholder(tf.int32, [None])
        self.r = tf.compat.v1.placeholder(tf.int32, [None])

    def create_test_placeholders(self):
        self.head = tf.compat.v1.placeholder(tf.int32, [None])
        self.rel = tf.compat.v1.placeholder(tf.int32, [None])
        self.tail = tf.compat.v1.placeholder(tf.int32, [None])

    def distance_function(self, points):
        self.rel_bx_low, self.rel_bx_high = self.compute_box(
            self.rel_bases_emb, self.rel_deltas_emb
        )
        lower_corner = self.rel_bx_low
        upper_corner = self.rel_bx_high
        centres = 1 / 2 * (lower_corner + upper_corner)

        widths = upper_corner - lower_corner
        widths_p1 = widths + tf.constant(1.0)
        width_cond = tf.where(
            tf.logical_and(lower_corner <= points, points <= upper_corner),
            tf.abs(points - centres) / widths_p1,
            widths_p1 * tf.abs(points - centres)
            - (widths / 2) * (widths_p1 - 1 / widths_p1),
        )
        distance = tf.norm(
            width_cond, axis=2, ord=self.params.p_norm
        )  ###batch*2*1 after norm
        distance = tf.reduce_sum(distance, axis=1)

        return distance

    def create_optimizer(self):
        self.loss = -1 * tf.math.reduce_mean(
            tf.math.log_sigmoid(self.params.gamma - self.pos_dissims)
        ) - tf.math.reduce_mean(
            tf.math.log_sigmoid(self.neg_dissims - self.params.gamma)
        )

        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            self.params.learning_rate
        ).minimize(self.loss)

    def save_model(self, itr):
        filename = (
            f"{self.cwd}/BoxModel_"
            + self.corpus_type
            + "_weights/"
            + str(itr)
            + ".ckpt"
        )
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        self.saver.save(self.sess, filename)

    def setup_weights(self):
        sqrt_size = 6.0 / math.sqrt(self.params.emb_size)

        self.ent_emb = tf.Variable(self.word_vectors, dtype=tf.float32, name="ent_emb")
        self.ent_emb_bmp = tf.Variable(
            self.word_vectors, dtype=tf.float32, name="ent_emb_bmp"
        )

        self.base_weight_ent_emb = tf.Variable(
            name="base_weight_ent_emb",
            initial_value=tf.random.uniform(
                shape=[300, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size
            ),
        )
        self.bump_weight_ent_emb = tf.Variable(
            name="bump_weight_ent_emb",
            initial_value=tf.random.uniform(
                shape=[300, self.params.emb_size], minval=-sqrt_size, maxval=sqrt_size
            ),
        )

        if self.learnable_shape:  # If shape is learnable, define variables accordingly
            self.rel_shapes = tf.Variable(
                name="rel_shapes",
                initial_value=tf.random.uniform(
                    shape=[self.num_rel, 2, self.params.emb_size],
                    minval=-sqrt_size,
                    maxval=sqrt_size,
                ),
            )
            self.norm_rel_shapes = self.product_normalise(
                self.rel_shapes, self.bounded_norm
            )
        else:
            self.norm_rel_shapes = tf.ones(
                [self.num_rel, 2, self.params.emb_size], name="norm_rel_shapes"
            )

        self.rel_bases = tf.Variable(
            name="rel_bases",
            initial_value=tf.random.uniform(
                shape=[self.num_rel, 2, self.params.emb_size],
                minval=-sqrt_size,
                maxval=sqrt_size,
            ),
        )

        if self.fixed_width:
            self.rel_multiples1 = tf.zeros([self.num_rel, 2, 1])
        else:
            self.rel_multiples1 = tf.Variable(
                name="rel_multiples",
                initial_value=tf.random.uniform(
                    shape=[self.num_rel, 2, 1], minval=-sqrt_size, maxval=sqrt_size
                ),
            )

        if self.hard_size:
            self.rel_multiples = self.total_size * tf.nn.softmax(
                self.rel_multiples1, axis=0
            )
        else:
            self.rel_multiples = tf.nn.elu(self.rel_multiples1) + tf.constant(1.0)

        self.rel_deltas = tf.multiply(
            self.rel_multiples, self.norm_rel_shapes, name="rel_deltas"
        )

        self.var_list = [
            self.rel_bases,
            self.rel_shapes,
            self.rel_multiples1,
            self.base_weight_ent_emb,
            self.bump_weight_ent_emb,
            self.ent_emb,
            self.ent_emb_bmp,
        ]

    def gather_train_embeddings(self):
        temp = tf.matmul(self.ent_emb, self.base_weight_ent_emb)
        self.ph_base_emb = tf.gather(temp, self.ph)
        self.pt_base_emb = tf.gather(temp, self.pt)
        self.nh_base_emb = tf.gather(temp, self.nh)
        self.nt_base_emb = tf.gather(temp, self.nt)

        temp1 = tf.matmul(self.ent_emb_bmp, self.bump_weight_ent_emb)
        if self.normed_bumps:  # Normalization of bumps option
            temp1 = tf.math.l2_normalize(temp1, axis=1)

        self.ph_bump_emb = tf.gather(temp1, self.ph)
        self.pt_bump_emb = tf.gather(temp1, self.pt)
        self.nh_bump_emb = tf.gather(temp1, self.nh)
        self.nt_bump_emb = tf.gather(temp1, self.nt)

        self.rel_bases_emb = tf.math.tanh(tf.gather(self.rel_bases, self.r))
        self.rel_deltas_emb = tf.math.tanh(tf.gather(self.rel_deltas, self.r))

    def gather_test_embeddings(self):
        temp = tf.matmul(self.ent_emb, self.base_weight_ent_emb)
        self.h_base_emb = tf.gather(temp, self.head)
        self.t_base_emb = tf.gather(temp, self.tail)

        temp1 = tf.matmul(self.ent_emb_bmp, self.bump_weight_ent_emb)
        if self.normed_bumps:  # Normalization of bumps option
            temp1 = tf.math.l2_normalize(temp1, axis=1)

        self.h_bump_emb = tf.gather(temp1, self.head)
        self.t_bump_emb = tf.gather(temp1, self.tail)

        self.rel_bases_emb = tf.math.tanh(tf.gather(self.rel_bases, self.rel))
        self.rel_deltas_emb = tf.math.tanh(tf.gather(self.rel_deltas, self.rel))

    def compute_box(self, box_base, box_delta):
        box_second = box_base + tf.constant(0.5) * box_delta
        box_first = box_base - tf.constant(0.5) * box_delta
        box_low = tf.minimum(box_first, box_second, "box_low")
        box_high = tf.maximum(box_first, box_second, "box_high")
        return box_low, box_high

    def create_train_model(self):

        self.pos_h_points = tf.expand_dims(self.ph_base_emb + self.pt_bump_emb, 1)
        self.pos_t_points = tf.expand_dims(self.pt_base_emb + self.ph_bump_emb, 1)

        self.neg_h_points = tf.expand_dims(self.nh_base_emb + self.nt_bump_emb, 1)
        self.neg_t_points = tf.expand_dims(self.nt_base_emb + self.nh_bump_emb, 1)

        self.pos_points = tf.math.tanh(
            tf.concat([self.pos_h_points, self.pos_t_points], 1)
        )
        self.neg_points = tf.math.tanh(
            tf.concat([self.neg_h_points, self.neg_t_points], 1)
        )
        #### concat dimension is batch*2*100 ####
        self.pos_dissims = self.distance_function(self.pos_points)
        self.neg_dissims = self.distance_function(self.neg_points)

    def create_test_model(self):

        self.h_points = tf.math.tanh(
            tf.expand_dims(self.h_base_emb + self.t_bump_emb, 1)
        )
        self.t_points = tf.math.tanh(
            tf.expand_dims(self.t_base_emb + self.h_bump_emb, 1)
        )
        self.test_pt = tf.concat([self.h_points, self.t_points], 1)
        self.dissims = self.distance_function(self.test_pt)

    def optimize(self):
        for itr in range(0, self.params.max_iterate + 1):
            total_loss = 0.0

            for b in range(self.num_batch):

                ph, pt, nh, nt, r = self.reader.next_batch()
                _, err = self.sess.run(
                    [self.optimizer, self.loss],
                    feed_dict={
                        self.ph: ph,
                        self.pt: pt,
                        self.nh: nh,
                        self.nt: nt,
                        self.r: r,
                    },
                )
                total_loss += err
            if math.isnan(total_loss):
                break
            print("Loss in iteration", itr, "=", total_loss)

            if itr % self.params.save_each == 0 and itr >= self.params.save_after:
                self.save_model(itr)
