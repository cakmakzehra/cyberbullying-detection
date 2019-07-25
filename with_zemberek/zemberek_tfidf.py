#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:56:00 2019

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
tfidf = TfidfVectorizer(min_df=1)


X = tfidf.fit_transform(new_comment).toarray()
y = comments.iloc[:,1].values


vocab = tfidf.get_feature_names()

print(pd.DataFrame(np.round(X, 2), columns=vocab))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

import time

from sklearn.svm import SVC

time_start = time.process_time()

svc = SVC(kernel="linear",gamma="auto")
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

time_elapsed = (time.process_time() - time_start)

print("SVM results : ")

#kFold
from sklearn.model_selection import cross_val_score
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

print("Random Forest Classifier results : ")


accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10)
print("Average score (mean) : %",accuracies.mean()*100)
print("std : %",accuracies.std()*100)

print("Random Forest Classifier computation time : " + str(time_elapsed))

from sklearn.naive_bayes import MultinomialNB

time_start = time.process_time()

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

time_elapsed = (time.process_time() - time_start)

print("Multinomial NB results : ")


accuracies = cross_val_score(estimator=mnb, X=X_train, y=y_train, cv=10)
print("Average score (mean) : %",accuracies.mean()*100)
print("std : %",accuracies.std()*100)

print("Multinomial NB computation time : "+ str(time_elapsed))    