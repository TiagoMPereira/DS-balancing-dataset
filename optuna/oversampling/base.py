import pickle as pkl
from typing import Dict, List

import pandas as pd
from imblearn.over_sampling.base import BaseOverSampler
from sdv.sampling import Condition
from sdv.single_table.base import BaseSingleTableSynthesizer

from metadata.sdv_metadata import SDVMetadata


class IOversampling(object):

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
            generated for each class (keys)
        '''
        pass

    def fit(self):
        pass

    def resample(self, occurences: Dict[str, int]) -> pd.DataFrame:
        '''
        Args:
            - occurences: Dict[str, int] | number of rows (values) to be
            generated for each class (keys)
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


class SDVOversampling(IOversampling):

    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target
        self.locales = ['en-us']
        self.verbose = False
        self.cuda = False
        self.fit_time = None
        self.sample_time = None
        self.fit_memo = None
        self.sample_memo = None

    def _create_model(self):
        metadata_obj = SDVMetadata()
        metadata_obj.create_from_df(self.data)
        self.metadata = metadata_obj.metadata
        self.model: BaseSingleTableSynthesizer = None

    def _get_conditions(self, conditions: Dict[str, int]) -> List[Condition]:
        conditions_list = []
        for class_, n_rows in conditions.items():
            if n_rows == 0:
                continue
            c = Condition(
                num_rows=n_rows,
                column_values={self.target: class_}
            )
            conditions_list.append(c)

        return conditions_list

    def fit(self):
        self.model.fit(self.data)

    def resample(self, occurences: Dict[str, int]) -> pd.DataFrame:
        conditions = self._get_conditions(occurences)
        generated_data = self.model.sample_from_conditions(conditions)
        
        # The generated_data is (as it says) only the new rows, so it's
        # necessary to append it to original data

        resampled_data = pd.concat([self.data, generated_data], axis=0)

        return resampled_data

    def save(self, path: str):
        with open(path, "wb") as fp:
            pkl.dump(self, fp)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as fp:
            obj = pkl.load(fp)

        return obj


class ImblearnOversampling(IOversampling):

    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target
        self.random_state = 42

    def _create_model(self):
        self.model: BaseOverSampler = None

    def _get_conditions(self, conditions: Dict[str, int]) -> Dict[str, int]:

        # conditions has the number of rows to be generated
        # The Imblearn submodules require the total number of rows after the
        # sampling. So the sampling strategy must be TOTAL + CONDITIONS

        current_values = dict(self.y.value_counts())

        return {
            class_: int(current_values[class_] + conditions[class_])
            for class_ in conditions.keys()
        }

    def fit(self):
        self.X = self.data.drop(columns=self.target)
        self.y = self.data[self.target]

    def resample(self, occurences: Dict[str, int]) -> pd.DataFrame:

        total_rows = self._get_conditions(occurences)
        self.model.sampling_strategy = total_rows

        # The resampled data consists of the original data + the resampled data
        X_resampled, y_resampled = self.synthesizer.fit_resample(
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
