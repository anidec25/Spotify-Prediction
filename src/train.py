import pandas as pd
import numpy as np
import joblib
from sklearn import metrics
from sklearn import tree

import warnings as war
war.filterwarnings('ignore')


def run(fold):

    # read the training data with folds
    df = pd.read_csv(r'../data/train_folds.csv')

    # training data is where kfold is not equal to provided fold
    # also note that we rest the index
    df_train = df[df['kfold'] != fold ].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df['kfold'] == fold].reset_index(drop= True)

    # drop the target and categorical columns from dataframe and convert it to a numpy array by using .values
    # target is label column in the dataframe
    #Dropping the categorical columns
    x_train = df.drop(columns=[['song_title','artist']],axis=1,inplace=True)
    x_train = df_train.drop('target', axis=1).values
    y_train = df_train.target.values

    #similarly, for validation, we have
    x_valid = df_valid.drop('label', axis = 1).values
    y_valid = df_valid.label.values

    #initialize simple decision tree classifier from sklearn

