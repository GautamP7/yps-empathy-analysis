from datasets import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import gc

model_name = 'MultinomialNB.sav'

def naive_bayes_train(X, Y, Xde, Yde, Xm, Ym, ps):
    print("\nTraining multinomial naives bayes classifier...")
    mnb = MultinomialNB()
    
    print('\nTuning the hyperparameters using GridSearchCV with the following parameter settings:')
    parameters = [
        {'alpha': [1, 5, 10, 15, 20, 25, 30, 40, 50]}
    ]
    print_params(parameters)
    clf = GridSearchCV(mnb, parameters, cv=ps, n_jobs=-1)
    clf.fit(Xm, Ym)
    print('\nThe best parameter settings are:\n{}'.format(clf.best_params_))
    
    print('\nPredicting multinoimal naive bayes classifier with the best parameter settings obtained...')
    Ypred = clf.predict(X)
    Ydepred = clf.predict(Xde)
    
    print('\nmultinomial naive bayes train accuracy = {0:.6f}'.format(np.mean(Ypred == Y)))
    print('\nmultinomial naive bayes dev accuracy = {0:.6f}'.format(np.mean(Ydepred == Yde)))

    print('\nSaving the model as \'{}\'...'.format(model_name))
    joblib.dump(clf, model_name)
    print('#====================================================================================================#')
    del X, Y, Xde, Yde, Xm, Ym, clf, mnb, Ypred, Ydepred
    gc.collect()

def naive_bayes_test(Xte, Yte):
    print('\nLoading the multinomial naive bayes model...')
    clf = joblib.load(model_name)

    print('\nPredicting multinomial naive bayes on test data...')
    Ytepred = clf.predict(Xte)
    print('\nmultinomial naive bayes test accuracy = {0:.6f}'.format(np.mean(Ytepred == Yte)))
    print('#====================================================================================================#')
    del Xte, Yte, clf, Ytepred
    gc.collect()
