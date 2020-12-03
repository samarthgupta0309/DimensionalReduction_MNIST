import os

import config
import model_dispatcher

import argparse
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree


def run(fold):
    df = pd.read_csv(config.TRAINING_FILE)
    df_train = df[df.Kfold != fold].reset_index(drop = True)
    # Validation one 
    df_valid  = df[df.Kfold == fold].reset_index(drop = True)
    # now x_train and y_Train
    x_train = df_train.drop("label", axis = 1).values
    y_train = df_train.labels.values
    # x_valid and y_valid time
    x_valid = df_valid.drop("label", axis =1).values
    y_label =  df_valid.label.values
    #simple tree as classifier
    clf = model_dispatcher.models(model)
    clf.fit(x_train, y_train)
    predict_test = clf.predict(x_valid)
    # evaluation
    accuracy = metrics.accuracy_score(y_valid, predict_test)
    print(f"Fold={fold}, accuracy={accuracy}")
    # save model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"df_{fold}.bin"))
# if __name__ == "main":
#     run(fold=0)
#     run(fold=1)
#     run(fold=2)
#     run(fold=3)
#     run(fold=4)
# sometimes it is not adivisable to run multiple folds in the same script 
# so we can pass arguments in it 
# hence by installing argparse we write
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add different argument you need
    parser.add_argument("--fold", type = int)
    parser.add_argument("--model", type = str)
    args = parser.parse_args()
    run(
        fold=args.fold,
        model = args.model
    )

