#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 22:14:23 2019

@author: zehra
"""

from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import string
from sklearn.model_selection import cross_val_score
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB


nltk.download('punkt')
nltk.download('stopwords')

comments = pd.read_csv('thesis_data.csv')

print(comments.head())

new_comment = []
sentences = []
for i in range (450):
    comment = comments['Comment'][i]
    comment = comment.split()
    table = str.maketrans('','',string.punctuation)
    comment = [c.translate(table).lower() for c in comment]
    table2 = str.maketrans('', '', string.digits)
    comment = [c.translate(table2) for c in comment]
    ps = PorterStemmer()
    comment = [ps.stem(piece) for piece in comment if not piece in set(stopwords.words('turkish'))]
    new_comment.append(' '.join(comment))
    sentences.append(comment)
    

scaler = MinMaxScaler()

model = Word2Vec(sentences, min_count =4, size = 300, window = 7, sg = 1) 
  
X = scaler.fit_transform(model[model.wv.vocab])
y = comments.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print('Gaussian result is : ')
accuracies = cross_val_score(estimator=gnb, X=X_train, y=y_train, cv=10)
print("Average score (mean): %",accuracies.mean()*100)
print("std: %",accuracies.std()*100)
