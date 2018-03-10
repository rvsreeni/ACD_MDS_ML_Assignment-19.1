# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 19:20:41 2018

@author: HP
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier

train_set = pd.read_csv('adult.data.csv', header = None)
test_set = pd.read_csv('adult.test.csv', skiprows = 1, header = None)
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
'occupation','relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
'native_country', 'wage_class']
train_set.columns = col_labels
test_set.columns = col_labels
print(train_set.shape)
print(test_set.shape)

train_set['workclass'] = train_set['workclass'].astype('category')
train_set['workclass_cat'] = train_set['workclass'].cat.codes
train_set['education'] = train_set['education'].astype('category')
train_set['education_cat'] = train_set['education'].cat.codes
train_set['marital_status'] = train_set['marital_status'].astype('category')
train_set['marital_status_cat'] = train_set['marital_status'].cat.codes
train_set['occupation'] = train_set['occupation'].astype('category')
train_set['occupation_cat'] = train_set['occupation'].cat.codes
train_set['relationship'] = train_set['relationship'].astype('category')
train_set['relationship_cat'] = train_set['relationship'].cat.codes
train_set['race'] = train_set['race'].astype('category')
train_set['race_cat'] = train_set['race'].cat.codes
train_set['sex'] = train_set['sex'].astype('category')
train_set['sex_cat'] = train_set['sex'].cat.codes
train_set['native_country'] = train_set['native_country'].astype('category')
train_set['native_country_cat'] = train_set['native_country'].cat.codes
train_set['wage_class'] = train_set['wage_class'].astype('category')
train_set['wage_class_cat'] = train_set['wage_class'].cat.codes
X_train = train_set.drop(['workclass','education','marital_status','occupation','relationship','race','sex',
                          'native_country','wage_class','wage_class_cat'], axis=1)
y_train = train_set['wage_class_cat']

test_set['workclass'] = test_set['workclass'].astype('category')
test_set['workclass_cat'] = test_set['workclass'].cat.codes
test_set['education'] = test_set['education'].astype('category')
test_set['education_cat'] = test_set['education'].cat.codes
test_set['marital_status'] = test_set['marital_status'].astype('category')
test_set['marital_status_cat'] = test_set['marital_status'].cat.codes
test_set['occupation'] = test_set['occupation'].astype('category')
test_set['occupation_cat'] = test_set['occupation'].cat.codes
test_set['relationship'] = test_set['relationship'].astype('category')
test_set['relationship_cat'] = test_set['relationship'].cat.codes
test_set['race'] = test_set['race'].astype('category')
test_set['race_cat'] = test_set['race'].cat.codes
test_set['sex'] = test_set['sex'].astype('category')
test_set['sex_cat'] = test_set['sex'].cat.codes
test_set['native_country'] = test_set['native_country'].astype('category')
test_set['native_country_cat'] = test_set['native_country'].cat.codes
test_set['wage_class'] = test_set['wage_class'].astype('category')
test_set['wage_class_cat'] = test_set['wage_class'].cat.codes
X_test = test_set.drop(['workclass','education','marital_status','occupation','relationship','race','sex',
                          'native_country','wage_class','wage_class_cat'], axis=1)
y_test = test_set['wage_class_cat']

params = {
'objective': 'binary:logistic',
'max_depth': 2,
'learning_rate': 1.0,
'silent': 1.0,
'n_estimators': 5
}

bst = XGBClassifier(**params).fit(X_train, y_train)

preds = bst.predict(X_test)
print(preds)

correct = 0
for i in range(len(preds)):
    if (y_test[i] == preds[i]):
        correct += 1

acc = accuracy_score(y_test, preds)
print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
print('Error: {0:.4f}'.format(1-acc))