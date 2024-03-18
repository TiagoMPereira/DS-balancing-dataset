from typing import Dict, List, Union

import optuna
import pandas as pd
from sklearn.metrics import f1_score
import logging

from data_balancing.autoML_frameworks import (automl_autogluon,
                                              automl_autokeras,
                                              automl_autopytorch,
                                              automl_autosklearn,
                                              automl_evalml,
                                              automl_flaml,
                                              automl_gama,
                                              automl_h2o,
                                              automl_lightautoml,
                                              automl_tpot)
from data_balancing.utils.load_methods import get_over_method, get_under_method
from data_balancing.utils.sampling_strategies import (oversampling_strategy,
                                                  undersampling_strategy)


FRAMEWORKS = {
    "autogluon": automl_autogluon,
    "autokeras": automl_autokeras,
    "autopytorch": automl_autopytorch,
    "autosklearn": automl_autosklearn,
    "evalml": automl_evalml,
    "flaml": automl_flaml,
    "gama": automl_gama,
    "h2o": automl_h2o,
    "lightautoml": automl_lightautoml,
    "tpot": automl_tpot
}


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
        framework_name: str,
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
        self.framework_name = framework_name

        # Original class occurences
        self.class_occurences = {
            k: int(v)
            for k, v in dict(self.train_dataset[self.target].value_counts()).items()
        }

        # Models path
        self.save_models_path = "./artifacts/optuna_models/"

    def _fit_eval(self, balanced_data: pd.DataFrame) -> float:

        # SCORE
        X_train = balanced_data.drop(columns=self.target)
        y_train = balanced_data[self.target]

        X_test = self.test_dataset.drop(columns=self.target)
        y_test = self.test_dataset[self.target]
        score = FRAMEWORKS[self.framework_name].fit_eval(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test
        )

        return float(score.get("f1_score_weighted", -1))


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
            score = -1.0

        return score
