from typing import Dict, List, Union

import optuna
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score
import logging

from optuna_mod.utils.load_methods import get_over_method, get_under_method
from optuna_mod.utils.sampling_strategies import (oversampling_strategy,
                                                  undersampling_strategy)

SEED = 42
EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60


class Objective:
    def __init__(
        self,
        train_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        target: str,
        oversampling_methods: List[str],
        undersampling_methods: List[str],
        oversampling_thresholds: List[Union[int, float, str]],
        undersampling_thresholds: List[Union[int, float, str]],
        project_id: str,
    ):
        self.oversampling_methods = oversampling_methods
        self.undersampling_methods = undersampling_methods
        self.oversampling_thresholds = oversampling_thresholds
        self.undersampling_thresholds = undersampling_thresholds
        self.project_id = project_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.target = target

        # Original class occurences
        self.class_occurences = {
            k: int(v)
            for k, v in dict(self.train_dataset[self.target].value_counts()).items()
        }

        # Models path
        self.save_models_path = "./models/"

    def _fit_eval(self, data: pd.DataFrame) -> float:

        X_train = data.drop(columns=self.target)
        y_train = data[self.target]

        X_test = self.test_dataset.drop(columns=self.target)
        y_test = self.test_dataset[self.target]

        train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
        test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

        clf = TabularPredictor(eval_metric='f1_weighted', label='class')

        clf = clf.fit(time_limit=EXEC_TIME_SECONDS, train_data=train_df)

        y_test = test_df['class'].values
        y_pred = clf.predict(test_df)

        return float(f1_score(y_test, y_pred, average="weighted"))


    def __call__(self, trial):

        # Select undersampling method
        under_method_name = trial.suggest_categorical(
            "undersampling_method", self.undersampling_methods)

        # Select oversampling method
        over_method_name = trial.suggest_categorical(
            "oversampling_method", self.oversampling_methods)

        # Select undersampling threshold -> Categorical 
        under_threshold = trial.suggest_categorical(
            "undersampling_threshold", self.undersampling_thresholds)

        # Select oversampling threshold -> Categorical 
        over_threshold = trial.suggest_categorical(
            "oversampling_threshold", self.oversampling_thresholds)
        
        _description = (f"{over_method_name}: {over_threshold} == {under_method_name}: {under_threshold}")
        print(_description)

        # Get undersampling model
        under_method = get_under_method(
            dataset=self.train_dataset,
            target=self.target,
            method_name=under_method_name,
            project_id=self.project_id,
            base_path=self.save_models_path
        )

        # Get oversampling model
        over_method = get_over_method(
            dataset=self.train_dataset,
            target=self.target,
            method_name=over_method_name,
            project_id=self.project_id,
            base_path=self.save_models_path
        )

        try:
            balanced_data = self.train_dataset.copy()

            # Generate oversampled data
            if over_threshold != 0:
                # Get oversampling strategy
                over_strategy = oversampling_strategy(
                    self.class_occurences, over_threshold
                )
                balanced_data = over_method.resample(over_strategy)

            # Generate undersampled data
            if under_threshold != 0:
                # Get undersampling strategy
                under_strategy = undersampling_strategy(
                    self.class_occurences, under_threshold
                )
                balanced_data = under_method.resample(
                    under_strategy, balanced_data
                )

            score = self._fit_eval(balanced_data)            

        except Exception as e:
            logging.error(f"\nFAIL ON \n{_description}\n{e}")
            score = 0.0

        return score


if __name__ == "__main__":

    dataset_name = "synthetic_dataset2"
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
    study.optimize(objective, n_trials=10)

    print(study.best_trial)

    print(study.get_trials())
