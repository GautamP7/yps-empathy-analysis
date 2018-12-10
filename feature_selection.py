from datasets import *
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.feature_selection import RFECV
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

X,   Y   = preprocess_data(YPSData.train_data)

# Using Recursive Feature Elimination with Cross Validation
print('Performing feature selection using RFECV with Logistic Regression...\n')
clf = LogisticRegression()
rfecv = RFECV(clf, cv=10, min_features_to_select=20)
f = rfecv.fit(X, Y)
print('Number of Features: {}\n'.format(f.n_features_))
print('Selected Features: {}\n'.format(f.support_))
print('Feature Ranking: {}\n'.format(f.ranking_))
print('Saving the selected features...')
joblib.dump(f, 'FeatureSelection.sav')
