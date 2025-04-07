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

sim_judgments = {}
with open('../data/final_pairs_ldp_cabnc.csv', mode='r') as csv_file:
    readCSV = csv.DictReader(csv_file, delimiter=',')
    for row in readCSV:
        if row['adjective'] not in sim_judgments:
            sim_judgments[row['adjective']] = {}
        sim_judgments[row['adjective']][row['noun']] = 0

model = gensim.models.Word2Vec.load("ldp_adult_word2vec.model")
wiki_model = KeyedVectors.load_word2vec_format('../data/wiki-news-300d-1M.vec')

wiki_sim_judgments = []
ldp_sim_judgments =[]
word_pairs = []

for adjective in sim_judgments:
    for noun in sim_judgments[adjective]:
        word_pairs.append((adjective,noun))
        if adjective in wiki_model.key_to_index and noun in wiki_model.key_to_index: 
            wiki_sim_judgments.append(float(wiki_model.similarity(adjective, noun)))
        else:
            wiki_sim_judgments.append(float('nan'))
        if adjective in model.wv.key_to_index and noun in model.wv.key_to_index:
        	ldp_sim_judgments.append(float(model.wv.similarity(adjective, noun)))
        else:
        	ldp_sim_judgments.append(float('nan'))


all_judgments = zip(word_pairs, wiki_sim_judgments, ldp_sim_judgments)

with open('../data/w2v_sims_ldp_cabnc.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["noun","adjective","wiki_similarity","ldp_similarity"])
    for (words, wiki, ldp) in all_judgments:
        writer.writerow([words[1],words[0],str(wiki),str(ldp)])

