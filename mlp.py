from datasets import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import gc

model_name = 'MLP.sav'

def perceptron_train(X, Y, Xde, Yde, Xm, Ym, ps):
    print("\nTraining multi layer perceptron classifier...")
    mlp = MLPClassifier(max_iter=10000, random_state=1, early_stopping=True)
    
    print('\nTuning the hyperparameters using GridSearchCV with the following parameter settings:')
    parameters = [
        {'activation': ['logistic', 'tanh', 'relu'], 'learning_rate': ['constant', 'invscaling', 'adaptive'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    ]
    print_params(parameters)
    clf = GridSearchCV(mlp, parameters, cv=ps, n_jobs=-1)
    clf.fit(Xm, Ym)
    print('\nThe best parameter settings are:\n{}'.format(clf.best_params_))
    
    print('\nPredicting multilayer perceptron classifier with the best parameter settings obtained...')
    Ypred = clf.predict(X)
    Ydepred = clf.predict(Xde)
    
    print('\nmultilayer perceptron train accuracy = {0:.6f}'.format(np.mean(Ypred == Y)))
    print('\nmultilayer perceptron dev accuracy = {0:.6f}'.format(np.mean(Ydepred == Yde)))

    print('\nSaving the model as \'{}\'...'.format(model_name))
    joblib.dump(clf, model_name)
    print('#====================================================================================================#')
    del X, Y, Xde, Yde, Xm, Ym, clf, mlp, Ypred, Ydepred
    gc.collect()

def perceptron_test(Xte, Yte):
    print('\nLoading the multilayer perceptron model...')
    clf = joblib.load(model_name)

    print('\nPredicting multilayer perceptron on test data...')
    Ytepred = clf.predict(Xte)
    print('\nmultilayer perceptron test accuracy = {0:.6f}'.format(np.mean(Ytepred == Yte)))
    print('#====================================================================================================#')
    del Xte, Yte, clf, Ytepred
    gc.collect()
