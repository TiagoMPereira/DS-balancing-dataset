import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from grid_search import grid_search

DATASET_PATH = "./autobalancer_datasets/"
RESULTS_PATH = "./autobalancer_results/"
SEED = 42

def _train_test_split(X, y, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


def run_pipeline(dataset_name: str, dataset_target: str, framework_name: str):

    # =========================================================================
    # SPLIT TRAIN TEST
    full_dataset_path = DATASET_PATH+dataset_name
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    if os.path.exists(full_dataset_path+"_train.csv") and os.path.exists(full_dataset_path+"_test.csv"):
        train_data = pd.read_csv(full_dataset_path+"_train.csv")
        test_data = pd.read_csv(full_dataset_path+"_test.csv")
        X_train = train_data.drop(columns=dataset_target)
        y_train = train_data[dataset_target]
        X_test = test_data.drop(columns=dataset_target)
        y_test = test_data[dataset_target]

    else:
        data = pd.read_csv(full_dataset_path+".csv")
        X = data.drop(columns=dataset_target)
        y = data[dataset_target]

        X_train, X_test, y_train, y_test = _train_test_split(X, y)

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        train_data.to_csv(full_dataset_path+"_train.csv", index=False)
        test_data.to_csv(full_dataset_path+"_test.csv", index=False)

    # =========================================================================
    # APPLYING GRID SEARCH
    results = grid_search(
        train_dataset=train_data,
        test_dataset=test_data,
        target=dataset_target,
        dataset_name=dataset_name,
        framework_name=framework_name
    )

    # =========================================================================
    # PERSISTING RESULTS
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    with open(RESULTS_PATH+dataset_name+"_"+framework_name+"_results.json", "w") as fp:
        json.dump(results, fp)

if __name__ == "__main__":
    run_pipeline("openml_44", "class", "autogluon")
