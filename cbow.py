#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:05:21 2019

@author: zehra
"""

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords


stoplist = stopwords.words('turkish')


comments = pd.read_csv('thesis_data.csv')

print(comments.head())


sentences = []
new_comment = []


for i in range (900):
    comment = comments['Comment'][i]
    tkn = nltk.WordPunctTokenizer()
    comment = comment.lower()
    tokens = tkn.tokenize(comment)
    comment = [token for token in tokens if not token in set(stoplist)]
    table = str.maketrans('','',string.punctuation)
    comment = [c.translate(table) for c in comment]
    table2 = str.maketrans('', '', string.digits)
    comment = [c.translate(table2) for c in comment]
    comment=' '.join(comment)
    new_comment.append(comment)
    sentences.append(comment.split())

    
from gensim.models import Word2Vec

size = 300

import time
time_start = time.process_time()

model = Word2Vec(sentences, min_count =4, size = size, window = 7) 
  

model.train(new_comment, total_examples=model.corpus_count, epochs=model.epochs)

time_elapsed = (time.process_time() - time_start)

print("CBOW computation time : " + str(time_elapsed))

vocab=model.wv.vocab

#model.wv.most_similar('bide')

print("Cosine similarity between 'umarım' and 'insallah' - CBow : ", model.wv.similarity('umarım', 'insallah')) 

