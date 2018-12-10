# YPS Empathy Analysis

#### Prerequisites
----
Python3, scikit-learn, pandas, numpy

#### Dataset
----
The dataset is split into train, development and test sets in the ratio of 60:20:20 respectively.  
To change the distribution of data, change the 'test_size' and/or 'random_state' parameters of the 'train_test_split' method in `datasets.py` file.  
Make sure to run `feature_selection.py` and `train.py` in case you make changes to the distribution of the data in order to reflect the changes based on the modified data.  
Original dataset: https://www.kaggle.com/miroslavsabo/young-people-survey/

#### Models
----
Six models have been used. They are:
- Most Frequent classifier (base line classifier) -> `baseline.py`
- Decision Tree classifier -> `dt.py`
- Random Forest classifier -> `rf.py`
- Multinomial Naive Bayes classifier -> `mnb.py`
- Multilayer Perceptron classifier -> `mlp.py`
- SVM classifier -> `svm.py`

Each of the above file has a train and test method implemented.

#### Steps to run
----
For feature selection - `python3 feature_selection.py`

For training - `python3 train.py`

For testing - `python3 test.py`

> Note:  
> Running the `feature_selection.py` file selects a subset of features and saves them in `FeatureSelection.sav`  
> 
> The 'min_features_to select' and 'cv' parameters of RFECV can be changed to select a different subset of features  
>   
> To train a subset of the models, comment the calls to the train methods of the models which are unnecessary  
>   
> To test a subset of the models, comment the calls to the test methods of the models which are unnecessary  
>  
> The repo contains already selected features and trained models. Hence, the test code can be run without feature selection and training
