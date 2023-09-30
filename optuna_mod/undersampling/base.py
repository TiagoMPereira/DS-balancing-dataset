import pickle as pkl
from typing import Dict

import pandas as pd
from imblearn.under_sampling.base import BaseUnderSampler


class IUndersampling(object):

    def __init__(self, data: pd.DataFrame, target: str):
        '''
        Args:
            - data: pd.DataFrame | Data to fit the model and to be resampled
            - target: str        | name of the target column
        '''
        pass

    def _create_model(self):
        pass

    def _get_conditions(self, conditions: Dict[str, int]):
        '''
        Args:
            - conditions: Dict[str, int] | number of rows (values) to be
            dropped for each class (keys)
        '''
        pass

    def fit(self):
        pass

    def resample(self, occurences: Dict[str, int]) -> pd.DataFrame:
        '''
        Args:
            - occurences: Dict[str, int] | number of rows (values) to be
            dropped for each class (keys)
        Return:
            - pandas DataFrame with the original data + generated data
        '''
        pass

    def save(self, path: str):
        '''
        Args:
            - path: str | path to save the current object
        '''
        pass

    @staticmethod
    def load(path: str):
        '''
        Args:
            - path: str | path to load the object
        '''
        pass


class ImblearnUndersampling(IUndersampling):

    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target
        self.random_state = 42

    def _create_model(self):
        self.model: BaseUnderSampler = None

    def _get_conditions(self, conditions: Dict[str, int]) -> Dict[str, int]:

        # conditions has the number of rows to be dropped
        # The Imblearn submodules require the total number of rows after the
        # sampling. So the sampling strategy must be TOTAL - CONDITIONS

        current_values = dict(self.y.value_counts())

        return {
            class_: int(current_values[class_] - conditions[class_])
            for class_ in conditions.keys()
        }

    def fit(self):
        self.X = self.data.drop(columns=self.target)
        self.y = self.data[self.target]

    def resample(self, occurences: Dict[str, int]) -> pd.DataFrame:

        total_rows = self._get_conditions(occurences)
        self.model.sampling_strategy = total_rows

        # The resampled data consists of the original data + the resampled data
        X_resampled, y_resampled = self.model.fit_resample(
            X=self.X, y=self.y
        )

        resampled_data = pd.concat([X_resampled, y_resampled], axis=1)
        return resampled_data

    def save(self, path: str):
        with open(path, "wb") as fp:
            pkl.dump(self, fp)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as fp:
            obj = pkl.load(fp)

        return obj
