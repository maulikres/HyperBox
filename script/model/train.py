import logging
import os
from argparse import Namespace
from configparser import ConfigParser

import tensorflow as tf

from box_model import BoxE


def train(model):
    model.setup_reader()
    print("Number of batches: ", model.num_batch)
    print("Number of words: ", model.num_ent)

    model.setup_weights()
    model.setup_saver()
    model.create_train_placeholders()
    model.gather_train_embeddings()
    model.create_train_model()
    model.create_optimizer()
    model.create_session()
    model.optimize()
    model.close_session()


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    config = ConfigParser()
    config.read(f"{cwd}/../../hparams.ini")
    corpus_type = config.get("General", "Corpus_type")

    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"{cwd}/../../logs/{corpus_type}_training.log",
        filemode="w",
        format="%(asctime)s%(name)s - %(levelname)s - %(message)s",
    )

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
    print()
    print(params)
    print()
    logging.info(params)

    tf.compat.v1.disable_eager_execution()

    model = BoxE(
        params=params, corpus_type=config.get("General", "Corpus_type"), work_dir=cwd
    )

    train(model)

    logging.info("Done Training")
    print("Done Training")
