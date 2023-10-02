import warnings

warnings.filterwarnings("ignore")

import logging
logging.basicConfig(filename='optuna_app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

import optuna
import pandas as pd

from optuna_mod.optimization.optimizer import Objective

if __name__ == "__main__":

    dataset_name = "openml_37"
    target = "class"

    train_dataset = pd.read_csv(f"./datasets/{dataset_name}_train.csv")
    test_dataset = pd.read_csv(f"./datasets/{dataset_name}_test.csv")

    o_methods = [
        "adasyn", "ctgan", "copulagan", "fastml",
        "gaussiancopula", "random", "smote", "tvae"
    ]
    u_methods = ["random"]
    o_thresh = [0, 0.25, 0.5, 1, 5, "auto"]
    u_thresh = [0, 0.05, 0.1, 0.2, 0.3, "auto"]

    _id = "optuna_test"

    objective = Objective(
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        target = target,
        oversampling_methods = o_methods,
        undersampling_methods = u_methods,
        oversampling_thresholds = o_thresh,
        undersampling_thresholds = u_thresh,
        project_id = _id,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print(study.best_trial)

    # print(study.get_trials())