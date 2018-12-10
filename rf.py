from datasets import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import gc

model_name = 'RF.sav'

def random_forest_train(X, Y, Xde, Yde, Xm, Ym, ps):
    print("\nTraining random forest classifier...")
    rf = RandomForestClassifier(random_state=1)
    
    print('\nTuning the hyperparameters using GridSearchCV with the following parameter settings:')
    parameters = [
        {'n_estimators': [5, 10, 15], 'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 5, 7, 10, 15]}
    ]
    print_params(parameters)
    clf = GridSearchCV(rf, parameters, cv=ps, n_jobs=-1)
    clf.fit(Xm, Ym)
    print('\nThe best parameter settings are:\n{}'.format(clf.best_params_))
    
    print('\nPredicting random forest classifier with the best parameter settings obtained...')
    Ypred = clf.predict(X)
    Ydepred = clf.predict(Xde)
    
    print('\nrandom forest train accuracy = {0:.6f}'.format(np.mean(Ypred == Y)))
    print('\nrandom forest dev accuracy = {0:.6f}'.format(np.mean(Ydepred == Yde)))

    print('\nSaving the model as \'{}\'...'.format(model_name))
    joblib.dump(clf, model_name)
    print('#====================================================================================================#')
    del X, Y, Xde, Yde, Xm , Ym, clf, rf, Ypred, Ydepred
    gc.collect()

def random_forest_test(Xte, Yte):
    print('\nLoading the random forest model...')
    clf = joblib.load(model_name)

    print('\nPredicting random forest on test data...')
    Ytepred = clf.predict(Xte)
    print('\nrandom forest test accuracy = {0:.6f}'.format(np.mean(Ytepred == Yte)))
    print('#====================================================================================================#')
    del Xte, Yte, clf, Ytepred
    gc.collect()
