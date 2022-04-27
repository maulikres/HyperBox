import math
import os
import pickle
import random


class Reader:
    def __init__(self, corpus_type):
        self.word2id = dict()
        self.triples = {"train": [], "valid": [], "test": []}
        self.start_batch = 0
        self.corpus_type = corpus_type
        self.cwd = os.path.dirname(os.path.abspath(__file__))
        with open(
            f"{self.cwd}/../../vocab/" + f"{corpus_type}_word_to_idx" + ".pkl", "rb"
        ) as f:
            self.word2id = pickle.load(f)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_neg_samples(self, neg_size):
        self.no_neg_samples = neg_size

    def train_triples(self):
        return self.triples["train"]

    def valid_triples(self):
        return self.triples["valid"]

    def test_triples(self):
        return self.triples["test"]

    def all_triples(self):
        return self.triples["train"] + self.triples["valid"] + self.triples["test"]

    def num_ent(self):
        return len(self.word2id)

    def num_batch(self):
        return int(math.ceil(float(len(self.triples["train"])) / self.batch_size))

    def next_pos_batch(self):
        if self.start_batch + self.batch_size > len(self.triples["train"]):
            ret_triples = self.triples["train"][self.start_batch :]
            self.start_batch = 0
        else:
            ret_triples = self.triples["train"][
                self.start_batch : self.start_batch + self.batch_size
            ]
            self.start_batch += self.batch_size
        return ret_triples

    def read_triples(self):
        temp_pairs = []
        with open(
            f"{self.cwd}/../.."
            + "/dataset/"
            + f"{self.corpus_type}_training_pairs"
            + ".pkl",
            "rb",
        ) as f:
            temp_pairs = pickle.load(f)
        for pair in temp_pairs:
            self.triples["train"].append(
                (self.word2id[pair[0]], 0, self.word2id[pair[1]])
            )

        temp_pairs = []
        with open(
            f"{self.cwd}/../.."
            + "/dataset/"
            + f"{self.corpus_type}_test_pairs"
            + ".pkl",
            "rb",
        ) as f:
            temp_pairs = pickle.load(f)
        for pair in temp_pairs:
            self.triples["test"].append(
                (self.word2id[pair[0]], 0, self.word2id[pair[1]])
            )

        self.train_gold_ids = [
            set() for _ in range(self.num_ent())
        ]  ### For tail neg samples
        self.train_gold_ids1 = [
            set() for _ in range(self.num_ent())
        ]  ### For head neg samples
        for i in range(len(self.triples["train"])):
            q_id = int(self.triples["train"][i][0])
            h_id = int(self.triples["train"][i][1])
            self.train_gold_ids[q_id].add(h_id)
            self.train_gold_ids1[h_id].add(q_id)

    def rand_ent_except(self, except_ent):
        rand_ent = random.randint(0, self.num_ent() - 1)
        while rand_ent == except_ent:
            rand_ent = random.randint(0, self.num_ent() - 1)
        return rand_ent

    def generate_neg_triples(self, batch_pos_triples):
        neg_triples = []
        for head, rel, tail in batch_pos_triples:
            head_or_tail = 1  ################random.randint(0,1)
            if head_or_tail == 0:  # head
                new_head = self.rand_ent_except(head)
                while new_head in self.train_gold_ids1[tail]:
                    new_head = self.rand_ent_except(head)

                neg_triples.append((new_head, rel, tail))
            else:  # tail
                new_tail = self.rand_ent_except(tail)
                while new_tail in self.train_gold_ids[head]:
                    new_tail = self.rand_ent_except(tail)
                neg_triples.append((head, rel, new_tail))
        return neg_triples

    def shred_triples(self, triples):
        h_idx = [triples[i][0] for i in range(len(triples))]
        r_idx = [0 for i in range(len(triples))]
        t_idx = [triples[i][2] for i in range(len(triples))]

        return h_idx, r_idx, t_idx

    def next_batch(self, neg_ratio=1):
        bp_triples = self.next_pos_batch()
        bp_triples = [i for i in bp_triples for j in range(self.no_neg_samples)]
        bn_triples = self.generate_neg_triples(bp_triples)
        ph_idx, pr_idx, pt_idx = self.shred_triples(bp_triples)
        nh_idx, nr_idx, nt_idx = self.shred_triples(bn_triples)
        return ph_idx, pt_idx, nh_idx, nt_idx, pr_idx
