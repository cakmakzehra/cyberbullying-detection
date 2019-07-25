#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:05:21 2019

@author: zehra
"""

import pandas as pd
import string
import numpy as np
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

model = Word2Vec(sentences, min_count =4, size = size, window = 7) 
  

model.train(new_comment, total_examples=model.corpus_count, epochs=model.epochs)

vocab=model.wv.vocab

#model.wv.most_similar('bide')

print("Cosine similarity between 'umarım' and 'insallah' - CBow : ", model.wv.similarity('umarım', 'insallah')) 


def average_word_vectors(words, model, vocabulary, num_features):
    
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
        
    return feature_vector
    
   
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)


w2v_feature_array = averaged_word_vectorizer(corpus=sentences, model=model, num_features=size)
#print(pd.DataFrame(w2v_feature_array))


X = w2v_feature_array
y = comments.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


from sklearn.model_selection import cross_val_score

import time

from sklearn.svm import SVC

time_start = time.process_time()

svc = SVC(kernel='rbf',gamma='auto')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

time_elapsed = (time.process_time() - time_start)

print('SVM results ( with RBF kernel ) : ')
accuracies = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=10)
print("Average score (mean) : %",accuracies.mean()*100)
print("std : %",accuracies.std()*100)

print("SVM computation time : " + str(time_elapsed))


from sklearn.ensemble import RandomForestClassifier

time_start = time.process_time()

rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

time_elapsed = (time.process_time() - time_start)

print('Random Forest Classifier results : ')
accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10)
print("Average score (mean) : %",accuracies.mean()*100)
print("std : %",accuracies.std()*100)

print("Random Forest Classifier computation time : " + str(time_elapsed))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
X =scaler.fit_transform(pd.DataFrame(w2v_feature_array))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)    

from sklearn.naive_bayes import MultinomialNB

time_start = time.process_time()

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

time_elapsed = (time.process_time() - time_start)

print('Multinomial NB results : ')
accuracies = cross_val_score(estimator=mnb, X=X_train, y=y_train, cv=10)
print("Average score (mean) : %",accuracies.mean()*100)
print("std : %",accuracies.std()*100)

print("Multinomial NB computation time : " + str(time_elapsed))