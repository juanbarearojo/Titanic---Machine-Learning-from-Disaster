#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: barearojo
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib 
import scipy as sp
import random
import time
import warnings

def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
df_all = pd.concat([df_train, df_test], sort=True).reset_index(drop=True)



#quitamos las columna id pero la guardamos para luego

if 'PassengerId' in df_train:
    df_train.drop('PassengerId', axis=1, inplace=True)
    
test_sin_id = df_test.drop('PassengerId', axis=1)



train_emb = df_train.fillna({"Embarked": "S"})


#al total le quitamos el id y el survived para gacer el imputer
df_all_sin_id_surv = df_all.drop('PassengerId',axis = 1)
df_all_sin_id_surv = df_all_sin_id_surv.drop('Survived', axis=1)

#columnas con varoles numericos
col_num = list(df_all_sin_id_surv.select_dtypes(include=np.number).columns)

print(df_all_sin_id_surv)
#Usamos dos KNNImputer distintos porque el test 
imputer_num = KNNImputer(n_neighbors=15)
imputer_num.fit(df_all_sin_id_surv[col_num])
train_emb[col_num] = imputer_num.transform(df_train[col_num])
test_sin_id[col_num] = imputer_num.transform(test_sin_id[col_num])


y = df_train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch","Age","Fare","Embarked"]
X = pd.get_dummies(train_emb[features])
X_test = pd.get_dummies(test_sin_id[features])


model_xgb = xgb.XGBClassifier( max_depth=5, min_child_weight=1.7817, n_estimators=3000)

model_xgb.fit(X, y)
predictions = model_xgb.predict(X_test)

output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})
output.to_csv('../data/submission.csv', index=False)
print("Resultados en Submission.csv")

