#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 23:27:18 2019

@author: zehra
"""

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import time

stoplist = stopwords.words('turkish')


comments = pd.read_csv('thesis_data.csv')


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
    new_comment.append(' '.join(comment))


time_start = time.process_time()

tfidf = TfidfVectorizer(min_df=1)

time_elapsed = (time.process_time() - time_start)
print("TF-IDF computation time : " + str(time_elapsed))

tfidf.fit_transform(new_comment).toarray()

vocab = tfidf.get_feature_names()

