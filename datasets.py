import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

def split_data(filename):
    df = pd.read_csv(filename)
    df_mod = pd.get_dummies(df)
    train_data, test_data = train_test_split(df_mod,     test_size=0.20, random_state=1, shuffle=True)
    train_data, dev_data  = train_test_split(train_data, test_size=0.25, random_state=1, shuffle=True)
    return train_data, dev_data, test_data

def preprocess_data(data):
    missing_cols = data.columns[data.isnull().any()]
    for i in missing_cols:
        if data[i].dtype == np.dtype('O'):
            data[i].fillna(data[i].value_counts().index[0], inplace=True)
        else:
            data[i].fillna(data[i].mode().iloc[0], inplace=True)
    data.loc[data['Empathy']  < 4, 'Empathy'] = -1
    data.loc[data['Empathy'] >= 4, 'Empathy'] =  1
    X = data.drop('Empathy', axis=1).values
    Y = data['Empathy'].values
    # unique, counts = np.unique(Y, return_counts=True)
    # print(dict(zip(unique, counts)))
    return X, Y

def print_params(parameters):
    for k, v in parameters[0].items():
        print('{}:'.format(k), end=' ')
        for i in range(len(v) - 1):
            print('{}'.format(v[i]), end=', ')
        print(v[-1])

class YPSData:
    train_data, dev_data, test_data = split_data('data/responses.csv')
