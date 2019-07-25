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
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

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


tfidf = TfidfVectorizer(min_df=1)



X = tfidf.fit_transform(new_comment).toarray()
y = comments.iloc[:,1].values


vocab = tfidf.get_feature_names()
#print(pd.DataFrame(np.round(X, 2), columns=vocab))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


time_start = time.process_time()

svc = SVC(kernel="linear")
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

time_elapsed = (time.process_time() - time_start)

print("SVM ( with linear kernel ) results : ")


accuracies = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=10)
print("Average score (mean) : %",accuracies.mean()*100)
print("std : %",accuracies.std()*100)

print("SVM ( with linear kernel ) computation time : " + str(time_elapsed))



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

time_start = time.process_time()    