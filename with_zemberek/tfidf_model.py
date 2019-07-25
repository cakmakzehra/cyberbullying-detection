#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:26:58 2019

@author: zehra
"""


import pandas as pd
import string
import numpy as np
import nltk
from nltk.corpus import stopwords

import jpype as jp

stoplist = stopwords.words('turkish')

comments = pd.read_csv('thesis_data.csv')

jp.startJVM(jp.get_default_jvm_path(),"-Djava.class.path=/home/zehra/mak_ogr/thesis/bin/zemberek-full.jar", "-ea")

TurkishMorphology = jp.JClass('zemberek.morphology.TurkishMorphology')

morphology = TurkishMorphology.createWithDefaults()
cümleler=[]
new_comment = []
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
    analysis = morphology.analyzeSentence(comment)
    results = morphology.disambiguate(comment, analysis).bestAnalysis()
    sentences.append(comment)
    
    
sentences_stem = []

for i in range (900):
    comment = sentences[i]
    analysis = morphology.analyzeSentence(comment)
    results = morphology.disambiguate(comment, analysis).bestAnalysis()
    words = []
    for result in results:
        words.append(str(result.getStem()))
    sentences_stem.append(words)
    new_comment.append(' '.join(words))
    

from sklearn.feature_extraction.text import TfidfVectorizer

import time
time_start = time.process_time()

tfidf = TfidfVectorizer(min_df=1)

time_elapsed = (time.process_time() - time_start)

print("TF-IDF computation time : " + str(time_elapsed))

jp.shutdownJVM()