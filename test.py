from datasets import *
from baseline import *
from svm import *
from mnb import *
from mlp import *
from dt import *
from rf import *
from sklearn.externals import joblib

print('Preprocessing the test data...')
Xte, Yte = preprocess_data(YPSData.test_data)

print('\nLoading the selected features...')
f = joblib.load('FeatureSelection.sav')
Xte = Xte[:, f.support_]

most_frequent_test(Xte, Yte) # Most Frequent Classifier (base line)

decision_tree_test(Xte, Yte) # Decision Tree

random_forest_test(Xte, Yte) # Random Forest

naive_bayes_test(Xte, Yte) # Multinomial Naive Bayes

perceptron_test(Xte, Yte) # Multi Layer Perceptron

svm_test(Xte, Yte) # SVM
