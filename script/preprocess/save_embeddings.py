import os
import pickle
from configparser import ConfigParser

import gensim
import numpy as np
from gensim.models import Word2Vec

cwd = os.path.dirname(os.path.abspath(__file__))
config = ConfigParser()
config.read(f"{cwd}/../../hparams.ini")
corpus_type = config.get("General", "Corpus_type")

word_to_idx = {}
with open(f"{cwd}/../../vocab/" + f"{corpus_type}_word_to_idx" + ".pkl", "rb") as f:
    word_to_idx = pickle.load(f)

idx_to_word = {}
with open(f"{cwd}/../../vocab/" + f"{corpus_type}_idx_to_word" + ".pkl", "rb") as f:
    idx_to_word = pickle.load(f)

vocab_size = len(word_to_idx)

# Load pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load(
    f"{cwd}/../../gensim_model/{corpus_type}_gensim_word2vec"
)

word_vectors = np.zeros((len(word_to_idx), 300))
bigram_cnt = 0

for index, word in idx_to_word.items():
    total_terms = 0
    if word.replace(" ", "_") in model.wv.vocab:
        word_vectors[index] = word_vectors[index] + model.wv.get_vector(
            word.replace(" ", "_")
        )
        bigram_cnt += 1
        continue

    for term in word.split(" "):
        if term in model.wv.vocab:
            total_terms += 1
            word_vectors[index] = word_vectors[index] + model.wv.get_vector(term)
    word_vectors[index] = word_vectors[index] / total_terms


np.save(f"{cwd}/../../gensim_model/{corpus_type}_gensim_word2vec")
