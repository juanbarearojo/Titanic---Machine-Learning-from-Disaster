#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:29:04 2023

@author: juan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
import lightgbm as lgb


#leemos los datos tanto del entrenamiento como del test y marcamos lo perdidos como NaN
train_data = pd.read_csv("../data/train.csv", na_values="NaN") # Definimos na_values para identificar bien los valores perdidos
test_data = pd.read_csv("../data/test.csv")




from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('../data/submission.csv', index=False)
print("Resultados en Submission.csv")
