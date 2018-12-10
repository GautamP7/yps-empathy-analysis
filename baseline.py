from datasets import *
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import gc

model_name = 'BaseLine.sav'
    
def most_frequent_train(X, Y, Xde, Yde):
    print("\nUsing most frequent classifier as the baseline classifier...")
    clf = DummyClassifier(strategy='most_frequent')
    clf.fit(X, Y)
    
    print('\nPredicting most frequent classifier...')
    Ypred = clf.predict(X)
    Ydepred = clf.predict(Xde)
    
    print('\nmost_frequent train accuracy = {0:.6f}'.format(np.mean(Ypred == Y)))
    print('\nmost_frequent dev accuracy = {0:.6f}'.format(np.mean(Ydepred == Yde)))

    print('\nSaving the model as \'{}\'...'.format(model_name))
    joblib.dump(clf, model_name)
    print('#====================================================================================================#')
    del X, Y, Xde, Yde, clf, Ypred, Ydepred
    gc.collect()

def most_frequent_test(Xte, Yte):
    print('\nLoading the most_frequent model...')
    clf = joblib.load(model_name)

    print('\nPredicting most_frequent on test data...')
    Ytepred = clf.predict(Xte)
    print('\nmost_frequent test accuracy = {0:.6f}'.format(np.mean(Ytepred == Yte)))
    print('#====================================================================================================#')
    del Xte, Yte, clf, Ytepred
    gc.collect()