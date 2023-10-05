import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

SEED = 42
EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60

def _train_test_split(X, y, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    data_name_target = [
        {"name": "openml_1462", "target": "Class"},
        {"name": "openml_1466", "target": "Class"},
        {"name": "openml_1479", "target": "Class"},
        {"name": "openml_1510", "target": "Class"},
        {"name": "openml_40691", "target": "class"},
        {"name": "openml_40975", "target": "class"},
    ]

    for d in data_name_target:
        dataset_name = d["name"]
        target = d["target"]

        print(dataset_name)

        data = pd.read_csv(f"./datasets/{dataset_name}.csv")


        x_cols = [col for col in data.columns if col!=target]

        X = data[x_cols]
        y = data[target]

        X_train, X_test, y_train, y_test = _train_test_split(X, y)

        train = pd.concat([X_train, y_train], axis=1)
        test = pd.concat([X_test, y_test], axis=1)
        
        train.to_csv(f"./datasets/{dataset_name}_train.csv", index=False)
        test.to_csv(f"./datasets/{dataset_name}_test.csv", index=False)
