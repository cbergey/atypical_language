from __future__ import absolute_import, division, print_function, unicode_literals
import io
import os
from gensim import utils
import gensim.models
import gensim.models.word2vec
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)             
import numpy as np   
import csv
from numpy import cov
import itertools

model = gensim.models.Word2Vec.load("ldp_adult_word2vec.model")

wiki_model = KeyedVectors.load_word2vec_format('../data/wiki-news-300d-1M.vec')

print("wiki wordsim")

wiki_model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

print("wiki simlex")

wiki_model.wv.evaluate_word_pairs(datapath('simlex999.txt'))

print("ldp wordsim")

model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))

print("ldp simlex")

model.wv.evaluate_word_pairs(datapath('simlex999.txt'))

