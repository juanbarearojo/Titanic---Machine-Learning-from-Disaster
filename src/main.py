#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: barearojo
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import xgboost as xgb
from ensemble import SklearnHelper
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier





SEED = 95

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


'''
CREAMOS LOS CLASIFICADORES QEU VAMOS A USAR
'''
# Put in our parameters for said classifiers
# Random Forest parametros
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}


# AdaBoost parametros
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parametros
gb_params = {
    'n_estimators': 500,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}
# Support Vector Classifier parametros 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


'''
Creación de matrices NumPy a partir de los conjuntos de entrenamiento y prueba:
 Aquí, se está preparando los datos para el entrenamiento y la prueba. y_train recibe la columna 'Survived' 
 de los datos de entrenamiento, aplanada en un array unidimensional usando ravel().
 Luego, la columna 'Survived' se elimina de los datos de entrenamiento, y las características restantes 
 se almacenan en x_train como una matriz NumPy. De manera similar, x_test se crea como una matriz
 NumPy a partir de los datos de prueba.   
 '''
# Crear matrices NumPy de los conjuntos de entrenamiento, prueba y destino (Survived) para alimentar a nuestros modelos
y_train = df_train['Survived'].ravel()
df_train = df_train.drop(['Survived'], axis=1)
x_train = df_train.values  # Crea una matriz de los datos de entrenamiento
x_test = df_test.values    # Crea una matriz de los datos de prueba


#Definimos objetos de la clase creada con anterioridad para
rf = SklearnHelper(clf=RandomForestClassifier, train=df_train, test=df_test, params=rf_params)
ada = SklearnHelper(clf=AdaBoostClassifier,  train=df_train, test=df_test, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier,  train=df_train, test=df_test,params=gb_params)
svc = SklearnHelper(clf=SVC,  train=df_train, test=df_test, params=svc_params)

# Crear nuestras predicciones OOF de entrenamiento y prueba. Estos resultados base se usarán como nuevas características
rf_oof_train, rf_oof_test = rf.get_oof(x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = ada.get_oof(x_train, y_train, x_test)# AdaBoost 
gb_oof_train, gb_oof_test = gb.get_oof(x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = svc.get_oof(x_train, y_train, x_test)  # Support Vector Classifier

#concatenamos todos los reusltados de los oof que usamos para entrenar nuestro XGBOOST
x_train = np.concatenate(( rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


model_xgb = xgb.XGBClassifier( n_estimators= 3000, max_depth= 4, min_child_weight= 2,gamma=0.9,                        
                        subsample=0.8,colsample_bytree=0.8,objective= 'binary:logistic',nthread= -1,scale_pos_weight=1)


model_xgb.fit(x_train, y_train)
predictions = model_xgb.predict(x_test)

output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
output.to_csv('../data/submission.csv', index=False)
print("Resultados en Submission.csv")

