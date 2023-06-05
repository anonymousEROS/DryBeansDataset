#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:43:11 2023

@author: blkcap
"""

import numpy as np 
from numpy import unique
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 

# import dataset
df = pd.read_excel('Dry_Bean_Dataset.xlsx')
df.head(10)
df.info()
df.isnull().sum()
df.describe()

df['Class'].unique()
df['Class'].value_counts()

plt.figure(figsize=(9,7))
sns.countplot(x='Class', data=df)
plt.show()

# Compute correlation matrix
plt.figure(figsize=(14,14))
corr_matrix = df.corr()

# Create a heatmap of the correlation matrix
sns.heatmap(corr_matrix,annot = True, cmap=plt.cm.Reds)

# Set chart title
plt.title("Correlation Matrix")

# Show the chart
plt.show()

# Create histogram for each feature
df.hist(bins=30, figsize=(15,15))
plt.show()

from sklearn.preprocessing import LabelEncoder
# Since class is a str convert to a int

labelencoder = LabelEncoder()
df["Class"] = labelencoder.fit_transform(df['Class'])

df.head(10)


X = df.drop(columns='Class')
y = df['Class']

from sklearn.feature_selection import f_classif, SelectKBest

# Feature selection

FeatureSelection = SelectKBest(score_func= f_classif ,k=10)
X = FeatureSelection.fit_transform(X, y)

S = FeatureSelection.get_support()
# Showing X Dimension 
print('X Shape is ' , X.shape)
print('Selected Features are : ' , S )


# First classifier LogisitcRegressionModel 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.9, shuffle = 'True')

LogisticRegressionModel = LogisticRegression(C=1, solver='newton-cg', class_weight='balanced', multi_class='multinomial',
                            fit_intercept=True, max_iter=100, random_state=44)
LogisticRegressionModel.fit(X_train, y_train)

print('LogisiticRegressionModel Train Score is: ', LogisticRegressionModel.score(X_train, y_train))
print('LogisiticRegressionModel Test Score is: ', LogisticRegressionModel.score(X_test, y_test))


# Second classifier DecisionTreeClassifierModel
from sklearn.tree import DecisionTreeClassifier

DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini',splitter='best', max_depth=12,random_state=44)
DecisionTreeClassifierModel.fit(X_train, y_train)

print('DecisionTreeClassifierModel Train Score is: ' , DecisionTreeClassifierModel.score(X_train, y_train))
print('DecisionTreeClassifierModel Test Score is: ' , DecisionTreeClassifierModel.score(X_test, y_test))


# Thrid Classifier Support Vector Machine
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=100)
clf = SVC  (kernel = 'linear')
clf.fit(X_train, y_train)
  
print('Support Vector Machine Train Score is: ' , clf.score(X_train, y_train))
print('Support Vector Machine Test Score is: ' , clf.score(X_test, y_test))















        



