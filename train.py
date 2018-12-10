from datasets import *
from baseline import *
from svm import *
from mnb import *
from mlp import *
from dt import *
from rf import *
from sklearn.externals import joblib
from sklearn.model_selection import PredefinedSplit
import numpy as np
import pandas as pd

print('Preprocessing the training and development data...')
X,   Y   = preprocess_data(YPSData.train_data)
Xde, Yde = preprocess_data(YPSData.dev_data)

print('\nLoading the selected features...')
f = joblib.load('FeatureSelection.sav')
X = X[:, f.support_]
Xde = Xde[:, f.support_]


Xm = np.concatenate((X, Xde), axis=0)
Ym = np.concatenate((Y, Yde), axis=0)
test_fold = []
for i in range(len(X)):
    test_fold.append(-1)
for i in range(len(Xde)):
    test_fold.append(0)

ps = PredefinedSplit(test_fold=test_fold)

most_frequent_train(X, Y, Xde, Yde) # Most Frequent Classifier (base line)

decision_tree_train(X, Y, Xde, Yde, Xm, Ym, ps) # Decision Tree

random_forest_train(X, Y, Xde, Yde, Xm, Ym, ps) # Random Forest

naive_bayes_train(X, Y, Xde, Yde, Xm, Ym, ps) # Multinomial Naive Bayes

perceptron_train(X, Y, Xde, Yde, Xm, Ym, ps) # Multi Layer Perceptron

svm_train(X, Y, Xde, Yde, Xm, Ym, ps) # SVM
