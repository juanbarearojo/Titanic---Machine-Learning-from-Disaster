#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: barearojo
"""

import pandas as pd
from sklearn.impute import KNNImputer
import xgboost as xgb


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

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16.336) & (dataset['Age'] <= 32.252), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32.252) & (dataset['Age'] <= 48.168), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48.168) & (dataset['Age'] <= 64.084), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64.084, 'Age'] = 4 ;

'''
ELIMINAMOS AQUELLAS CARACTERISTICAS QUE NOS APAORTAN NADA DE INTERÉS 
PassengerId -> por lo general lo id no aportan nada útil a la hora de modelo solo hacer que sea menos interpretable
Ticket -> No parece haber un patrón aparente al respecto
Cabin -> Poca infromación al respecto gran cantidad de valors peridos
Sibsp -> ya tenemos family size
CAtegorical Age/Fare -> las queriamos para los intervalos -> ya no son necesarias
'''
Variables_eliminar_train = ['PassengerId','Ticket','Cabin','SibSp','CategoricalFare','CategoricalAge','Name']
Variables_eliminar_test = ['PassengerId','Ticket','Cabin','SibSp','Name']
df_train = df_train.drop(Variables_eliminar_train, axis = 1)
df_test = df_test.drop(Variables_eliminar_test, axis = 1)




y = df_train["Survived"]
df_train = df_train.drop("Survived", axis = 1)
model_xgb = xgb.XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2,gamma=0.9,                        
                        subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread= -1,scale_pos_weight=1)


model_xgb.fit(df_train, y)
predictions = model_xgb.predict(df_test)

output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
output.to_csv('../data/submission.csv', index=False)
print("Resultados en Submission.csv")

