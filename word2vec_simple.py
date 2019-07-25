#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:44:22 2019

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
    sentences.append(comment.split())
   
    
from gensim.models import Word2Vec
model = Word2Vec(sentences, min_count=1,size=300)

#model.wv.most_similar('iyi')
#model.wv['adam']
#model.wv.similarity('umarım', 'insallah')

vocabulary= model.wv.vocab
print(vocabulary)
