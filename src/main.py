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
PassengerId = df_test['PassengerId']
full_data = [df_train, df_test]

##Feature Engineering Vamos a crear aquellas variables que vamos a utilizar

##Creación de variables

'''
HAS_CABIN
Si tiene cabina o no . La función verifica si el tipo de valor en la columna "Cabin" es float.
Si es así, significa que no hay información de la cabina para ese pasajero, y se asigna el valor 0. 
Si el tipo no es float, significa que hay información de la cabina, y se asigna el valor 1.
''' 
df_train['Has_Cabin'] = df_train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
df_test['Has_Cabin'] = df_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


'''
FAMILY SIZE y IS_ALONE
Sumamos variabes de numeros de hermanos y padres
Derivada de FAMILY SIZE si  FS = 0 -> Is_alone = 1
''' 
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

''' Quitamos valores nulos de EMBARKED y  FARE(KNNIMputer)'''

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
for dataset in full_data:
    fare_imputer = KNNImputer(n_neighbors=15)  # Podemos ajustar número de vecinos
    fare_values = dataset[['Fare']].values
    dataset['Fare'] = fare_imputer.fit_transform(fare_values)
    
df_train['CategoricalFare'] = pd.qcut(df_train['Fare'], 4)
print("\n Intervalos de las tarifas y índices de supervivencias de estos ")
print (df_train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


''' Quitamos valores nulos de AGE seguimos misma idea que en FARE'''

for dataset in full_data:
    fare_imputer = KNNImputer(n_neighbors=15)  # Podemos ajustar número de vecinos
    fare_values = dataset[['Age']].values
    dataset['Age'] = fare_imputer.fit_transform(fare_values)
    
df_train['CategoricalAge'] = pd.cut(df_train['Age'], 5) #usamos cut en ve de qcut para dividri en intervalos de rango fijo
print("\n Intervalos de las edades y índices de supervivencias de estos ")
print (df_train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())

'''
En esta porción del código en vamos a hcer un encodeao de forma manual en vez de automatica con 
el objetivo de obtener mejores resultados.
'''

'''
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

'''