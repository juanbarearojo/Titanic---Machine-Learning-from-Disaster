#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: barearojo

Esta clase en Python está diseñada para extender los clasificadores de la biblioteca scikit-learn (Sklearn)

"""

from sklearn.model_selection import KFold
import numpy as np

# Class to extend the Sklearn classifier
class SklearnHelper:
    
    def __init__(self, clf, train, test, seed=95, params=None):
        self.ntrain = train.shape[0]
        self.ntest = test.shape[0]
        self.SEED = seed  # for reproducibility
        self.NFOLDS = 5  # set folds for out-of-fold prediction
        self.kf = KFold(n_splits=self.NFOLDS, shuffle=True, random_state=seed)        
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def feature_importances(self, x, y):
        print(self.clf.feature_importances_)
        
    def get_oof(self, x_train, y_train, x_test):
        oof_train = np.zeros((self.ntrain,))
        oof_test = np.zeros((self.ntest,))
        oof_test_skf = np.empty((self.NFOLDS, self.ntest))

        for i, (train_index, test_index) in enumerate(self.kf.split(x_train)):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            self.clf.fit(x_tr, y_tr)

            oof_train[test_index] = self.clf.predict(x_te)
            oof_test_skf[i, :] = self.clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)