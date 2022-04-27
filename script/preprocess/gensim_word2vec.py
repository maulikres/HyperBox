import logging
import os
import pickle
import sys
import time
from configparser import ConfigParser

from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from nltk import word_tokenize


class MySentences(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        for line in open(self.filepath):
            yield word_tokenize(line.strip("\n"))


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    config = ConfigParser()
    config.read(f"{cwd}/../../hparams.ini")
    corpus_type = config.get("General", "Corpus_type")
    print(f"Training for {corpus_type} word embeddings.")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    output_file_handler = logging.FileHandler(
        f"{cwd}/../../logs/{corpus_type}_gensim_word2vec.log"
    )
    output_file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)

    if corpus_type == "music":
        corpus_file = f"{cwd}/../../dataset/2B_music_bioreviews_tokenized.txt"
        min_count = 3
    elif corpus_type == "medical":
        corpus_file = f"{cwd}/../../dataset/2A_med_pubmed_tokenized.txt"
        min_count = 3
    elif corpus_type == "general":
        corpus_file = f"{cwd}/../../dataset/UMBC_tokenized.txt"
        min_count = 5

    sentences_iter = MySentences(corpus_file)

    # Phrase Detection
    # Give some common terms that can be ignored in phrase detection
    # For example, 'state_of_affairs' will be detected because 'of' is provided here:
    common_terms = ["of", "with", "without", "and", "or", "the", "a"]
    # Create the relevant phrases from the list of sentences:
    phrases = Phrases(sentences_iter, common_terms=common_terms)
    # The Phraser object is used from now on to transform sentences
    bigram = Phraser(phrases)
    # Applying the Phraser to transform our sentences is simply
    sentences_iter = list(bigram[sentences_iter])

    embedding_size = config.getint("Gensim", "embedding_size")
    workers = config.getint("Gensim", "workers")
    window = config.getint("Gensim", "window")
    gensim_iter = config.getint("Gensim", "gensim_iter")
    logger.info(
        f"Gensim Model with min_count: {min_count}, embedding_size: {embedding_size}, workers: {workers}, window: {window} and epochs: {gensim_iter}"
    )

    start = time.time()
    model = Word2Vec(
        sentences_iter,
        min_count=min_count,  # Ignore words that appear less than this
        size=embedding_size,  # Dimensionality of word embeddings
        workers=workers,  # Number of processors (parallelisation)
        iter=gensim_iter,
        window=window,
    )  # Context window for words during training

    end = time.time()

    logger.info(f"Gensim Model trained in {end-start} seconds")
    logger.info(f"Gensim Model Total Vocab Length: {len(model.wv.vocab)}")
    print("Completed in: %s sec" % (end - start))

    model.save(f"{cwd}/../../gensim_model/{corpus_type}_gensim_word2vec")
    logger.info(
        f"Gensim Model saved at {cwd}/../../gensim_model/{corpus_type}_gensim_word2vec"
    )
