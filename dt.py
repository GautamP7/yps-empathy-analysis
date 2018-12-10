from datasets import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import gc

model_name = 'DT.sav'

def decision_tree_train(X, Y, Xde, Yde, Xm, Ym, ps):
    print("\nTraining decision tree classifier...")
    dt = DecisionTreeClassifier(random_state=1)
    
    print('\nTuning the hyperparameters using GridSearchCV with the following parameter settings:')
    parameters = [
        {'criterion': ['gini', 'entropy'], 'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]}
    ]
    print_params(parameters)
    clf = GridSearchCV(dt, parameters, cv=ps, n_jobs=-1)
    clf.fit(Xm, Ym)
    print('\nThe best parameter settings are:\n{}'.format(clf.best_params_))
    
    print('\nPredicting decision tree classifier with the best parameter settings obtained...')
    Ypred = clf.predict(X)
    Ydepred = clf.predict(Xde)
    
    print('\ndecision tree train accuracy = {0:.6f}'.format(np.mean(Ypred == Y)))
    print('\ndecision tree dev accuracy = {0:.6f}'.format(np.mean(Ydepred == Yde)))

    print('\nSaving the model as \'{}\'...'.format(model_name))
    joblib.dump(clf, model_name)
    print('#====================================================================================================#')
    del X, Y, Xde, Yde, Xm , Ym, clf, dt, Ypred, Ydepred
    gc.collect()

def decision_tree_test(Xte, Yte):
    print('\nLoading the decision tree model...')
    clf = joblib.load(model_name)

    print('\nPredicting decision tree on test data...')
    Ytepred = clf.predict(Xte)
    print('\ndecision tree test accuracy = {0:.6f}'.format(np.mean(Ytepred == Yte)))
    print('#====================================================================================================#')
    del Xte, Yte, clf, Ytepred
    gc.collect()
