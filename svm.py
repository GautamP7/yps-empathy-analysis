from datasets import *
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import gc

model_name = 'SVM.sav'

def svm_train(X, Y, Xde, Yde, Xm, Ym, ps):
    print("\nTraining svm classifier...")
    svc = SVC(random_state=1)
    
    print('\nTuning the hyperparameters using GridSearchCV with the following parameter settings:')
    parameters = [
        {'C': [1, 10], 'kernel': ['rbf', 'poly'], 'gamma': [1, 10, 'scale'], 'degree': [1, 2]}                
    ]
    print_params(parameters)
    clf = GridSearchCV(svc, parameters, cv=ps)
    clf.fit(Xm, Ym)
    
    print('\nThe best parameter settings are:\n{}'.format(clf.best_params_))
    print('\nPredicting svm classifier with the best parameter settings obtained...')
    Ypred = clf.predict(X)
    Ydepred = clf.predict(Xde)
    
    print('\nsvm train accuracy = {0:.6f}'.format(np.mean(Ypred == Y)))
    print('\nsvm dev accuracy = {0:.6f}'.format(np.mean(Ydepred == Yde)))

    print('\nSaving the model as \'{}\'...'.format(model_name))
    joblib.dump(clf, model_name)
    print('#====================================================================================================#')
    del X, Y, Xde, Yde, Xm , Ym, clf, svc, Ypred, Ydepred
    gc.collect()

def svm_test(Xte, Yte):
    print('\nLoading the svm model...')
    clf = joblib.load(model_name)

    print('\nPredicting svm on test data...')
    Ytepred = clf.predict(Xte)
    print('\nsvm test accuracy = {0:.6f}'.format(np.mean(Ytepred == Yte)))
    print('#====================================================================================================#')
    del Xte, Yte, clf, Ytepred
    gc.collect()
