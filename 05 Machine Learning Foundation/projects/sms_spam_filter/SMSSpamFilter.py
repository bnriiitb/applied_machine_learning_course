#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMS Spam Filter implemented using Multinomial Naive Base from sklearn.
Dataset Used : SMS Spam Collection Data Set 
URL: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Created on Sat Sep  9 12:11:52 2017

@author: Nagaraju Budigam
"""

import pandas as pd

#Load the data into dataframe
data_frame=pd.read_table(
        '/Users/panda/Documents/mlnd/smsspamcollection/SMSSpamCollection', 
        sep='\t', header=None, names=['lable','sms_text'])
print('Step 1 : Load SMS Spam Data')
print('Step 2 : Transform Lables into numerical data')
data_frame['lable']=data_frame.lable.map({'ham':0,'spam':1})
print('############################ HEAD DATA #########################')
print(data_frame.head())
print('##############################################################')
#Import CountVectorizer from sklearn, which returns docuent term matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vec=CountVectorizer()

#Split the data for training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data_frame['sms_text'],
                                               data_frame['lable'],
                                               random_state=1)
print('Step 3: Split the data for training and testing')
traing_data=count_vec.fit_transform(x_train)#traiing data
print('Step 4: Fit and transform training data')
testing_data=count_vec.transform(x_test)#test data
print('Step 5: Transform test data')
from sklearn.naive_bayes import MultinomialNB
mutliNBClassifier=MultinomialNB()
mutliNBClassifier.fit(traing_data,y_train)
print('Step 6: Fit training data into mutliNBClassifier')
predictions=mutliNBClassifier.predict(testing_data)
print('Step 7: Predict the labels using testing_data')
print(predictions)

print('Step 8: Get the metrics')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))

myinput=[]
myinput.append('viagra offer adsfasdf win you?')
print('Predicting User input',format(myinput))
outputval='spam'
if(mutliNBClassifier.predict(count_vec.transform(myinput))==0):
    outputval='ham'
print(outputval)