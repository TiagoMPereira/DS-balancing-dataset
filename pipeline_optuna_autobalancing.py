import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from data_balancing.autoML_frameworks.utils import SET_SEED, GET_SEED
from optuna_search_tpe import optuna_search_tpe
from optuna_search_grid import optuna_search_grid

DATASET_PATH = "./artifacts/autobalancer_datasets/"
RESULTS_PATH = "./artifacts/autobalancer_optuna_results/"

def _train_test_split(X, y, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=GET_SEED()
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
    # APPLYING OPTUNA

    results = optuna_search_grid(
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

    results.to_csv(RESULTS_PATH+dataset_name+"_"+framework_name+"_results.csv")

if __name__ == "__main__":
    # =========================================================================
    # Reading shell variables

    print(f'sys.argv', sys.argv)
    if len(sys.argv) not in [4, 5]:
        print('Number of arguments must be 4 or 5: "dataset_name", "dataset_target", "framework_name", [seed]')
    else:
        dataset_name = str(sys.argv[1])
        dataset_target = str(sys.argv[2])
        framework_name = str(sys.argv[3])
        if len(sys.argv) == 5:
            seed = int(sys.argv[4])
            SET_SEED(seed=seed)
        
    print(f'\n**** {dataset_name:} {dataset_target:} {framework_name:} {GET_SEED()} ****\n')
    # run_pipeline("openml_44", "class", "autogluon", 42)
    run_pipeline(dataset_name, dataset_target, framework_name)
