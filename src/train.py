#train.py

import os
import argparse
import config
import model_dispatcher

import pandas as pd
import joblib
from pandas.io import parsers
from sklearn import metrics
from sklearn import tree

import  warnings
warnings.filterwarnings('ignore')


def run(fold,model):

    warnings.filterwarnings('ignore')

    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    #Dropping the categorical columns
    df.drop(columns=['song_title','artist'],axis=1, inplace=True)

    # training data is where kfold is not equal to provided fold
    # also note that we rest the index
    df_train = df[df['kfold'] != fold ].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df['kfold'] == fold].reset_index(drop= True)

    # drop the target and categorical columns from dataframe and convert it to a numpy array by using .values
    # target is label column in the dataframe
    x_train = df_train.drop('target', axis=1).values
    y_train = df_train.target.values

    #similarly, for validation, we have
    x_valid = df_valid.drop('target', axis = 1).values
    y_valid = df_valid.target.values

    #initialize simple decision tree classifier from sklearn
    clf = model_dispatcher.models[model]
    
    #fit the model on training data
    clf.fit(x_train,y_train)

    #create predictions for validation samples
    preds = clf.predict(x_valid)

    #calculate and print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy= {accuracy}")

    #save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",type=int)
    parser.add_argument("--model",type=str)
    args = parser.parse_args()

    run(
        fold = args.fold,
        model= args.model
        )

