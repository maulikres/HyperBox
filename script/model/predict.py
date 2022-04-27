import csv
import logging
import os
import pickle
import sys
from argparse import Namespace
from configparser import ConfigParser

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from box_model import BoxE


def load_model(params, corpus_type, work_dir, checkpoint_epoch):
    model = BoxE(params=params, corpus_type=corpus_type, work_dir=work_dir)
    model.setup_reader()
    tf.compat.v1.reset_default_graph()
    model.setup_weights()
    model.setup_loader()
    model.create_test_placeholders()
    model.gather_test_embeddings()
    model.create_test_model()
    model.create_session()
    model.load_session(str(checkpoint_epoch))  # iter as arguments
    return model


def single_test(mdl, word):
    head = mdl.reader.word2id[word]
    h_idx = [head for i in range(mdl.reader.num_ent())]
    r_idx = [0 for i in range(mdl.reader.num_ent())]
    t_idx = [i for i in range(mdl.reader.num_ent())]
    raw_preds = mdl.sess.run(
        mdl.dissims, feed_dict={mdl.head: h_idx, mdl.rel: r_idx, mdl.tail: t_idx}
    )
    raw_preds[head] = 100000

    return np.argsort(raw_preds)[:15]


def create_predict_file(
    fn, test_file_entity, test_file_gold, output_file, word_to_idx, model
):
    file_vocab = list(word_to_idx.keys())
    test_hyponyms = []
    test_hypernyms = []

    with open(test_file_entity) as f:
        hyponymsls = f.readlines()
    with open(test_file_gold) as f:
        hypernymls = f.readlines()

    for i in range(len(hyponymsls)):

        temp = hyponymsls[i].strip("\n").split("\t")
        if temp[0] in file_vocab:
            test_hyponyms.append(temp[0])
        else:
            continue

        temp = hypernymls[i].strip("\n").split("\t")
        buffer = []
        for word in temp:
            if word in file_vocab:
                buffer.append(word)

        test_hypernyms.append(buffer)

    prediction = []
    for query in tqdm(test_hyponyms):
        prediction.append([idx_to_word[idd] for idd in fn(model, query)])

    with open(output_file + "_gold.txt", "w", encoding="utf8", newline="") as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter="\t", lineterminator="\n")
        for i in range(len(hyponymsls)):
            tsv_writer.writerow(test_hypernyms[i])

    with open(output_file + "_pred.txt", "w", encoding="utf8", newline="") as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter="\t", lineterminator="\n")
        for i in range(len(hyponymsls)):
            tsv_writer.writerow(prediction[i])


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    config = ConfigParser()
    config.read(f"{cwd}/../../hparams.ini")
    corpus_type = config.get("General", "Corpus_type")

    params = Namespace(
        batch_size=config.getint("train", "batch_size"),
        emb_size=config.getint("train", "emb_proj_size"),
        p_norm=config.getint("train", "p_norm"),
        gamma=config.getfloat("train", "gamma"),
        alpha=config.getfloat("train", "alpha"),
        learning_rate=config.getfloat("train", "learning_rate"),
        max_iterate=config.getint("train", "max_iterate"),
        save_after=config.getint("train", "save_after"),
        save_each=config.getint("train", "save_each"),
        no_neg_samples=config.getint("train", "no_neg_samples"),
        bounded_norm=config.getboolean("train", "bounded_norm"),
        normed_bumps=config.getboolean("train", "normed_bumps"),
        fixed_width=config.getboolean("train", "fixed_width"),
        hard_size=config.getboolean("train", "hard_size"),
        total_size=config.getint("train", "total_size"),
        learnable_shape=config.getboolean("train", "learnable_shape"),
    )

    tf.compat.v1.disable_eager_execution()

    word_to_idx = {}
    with open(f"{cwd}/../../vocab/" + f"{corpus_type}_word_to_idx" + ".pkl", "rb") as f:
        word_to_idx = pickle.load(f)

    idx_to_word = {}
    with open(f"{cwd}/../../vocab/" + f"{corpus_type}_idx_to_word" + ".pkl", "rb") as f:
        idx_to_word = pickle.load(f)

    vocab_size = len(word_to_idx)

    data_dir = f"{cwd}/../../SemEval2018-Task9/"

    if corpus_type == "music":
        vocab_file = data_dir + "vocabulary/2B.music.vocabulary.txt"
        validation_file_entity = data_dir + "trial/data/2B.music.trial.data.txt"
        validation_file_gold = data_dir + "trial/gold/2B.music.trial.gold.txt"

        test_file_entity = data_dir + "test/data/2B.music.test.data.txt"
        test_file_gold = data_dir + "test/gold/2B.music.test.gold.txt"

    elif corpus_type == "medical":
        vocab_file = data_dir + "vocabulary/2A.medical.vocabulary.txt"
        validation_file_entity = data_dir + "trial/data/2A.medical.trial.data.txt"
        validation_file_gold = data_dir + "trial/gold/2A.medical.trial.gold.txt"

        test_file_entity = data_dir + "test/data/2A.medical.test.data.txt"
        test_file_gold = data_dir + "test/gold/2A.medical.test.gold.txt"

    elif corpus_type == "general":
        vocab_file = data_dir + "vocabulary/1A.english.vocabulary.txt"
        validation_file_entity = data_dir + "trial/data/1A.english.trial.data.txt"
        validation_file_gold = data_dir + "trial/gold/1A.english.trial.gold.txt"

        test_file_entity = data_dir + "test/data/1A.english.test.data.txt"
        test_file_gold = data_dir + "test/gold/1A.english.test.gold.txt"

    model = load_model(params, corpus_type, cwd, sys.argv[1])
    create_predict_file(
        single_test,
        test_file_entity,
        test_file_gold,
        f"{cwd}/output/{corpus_type}",
        word_to_idx,
        model,
    )
