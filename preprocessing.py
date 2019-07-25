#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:46:24 2019

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

print('-'*90)
print('A preprocessing example : ')
print('-'*90)
sample=comments['Comment'][0]
print('Our sample is : ',sample)
print('-'*90)
sample = sample.lower()
print('After lower case : ',sample)
print('-'*90)
tkn = nltk.WordPunctTokenizer()
sample=tkn.tokenize(sample)
print('After tokenization : ',sample)
print('-'*90)
sample=[token for token in sample if not token in set(stoplist)]
print('After removing stopwords : ',sample)
print('-'*90)
table = str.maketrans('','',string.punctuation)
sample = [s.translate(table) for s in sample]
print('After removing punctuation : ',sample)
print('-'*90)
table2 = str.maketrans('', '', string.digits)
sample = [s.translate(table2) for s in sample]
print('After removing digits : ',sample)
print('-'*90)
sample=' '.join(sample)
print('And comment is a sentence now : ',sample)
print('-'*90)

for i in range (900):
    comment = comments['Comment'][i]   
    comment = comment.lower()
    tokens = tkn.tokenize(comment)
    comment = [token for token in tokens if not token in set(stoplist)]    
    comment = [c.translate(table) for c in comment]   
    comment = [c.translate(table2) for c in comment]
    comment=' '.join(comment)
    new_comment.append(comment)
    sentences.append(comment.split())

   