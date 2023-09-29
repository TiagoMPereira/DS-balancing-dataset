from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
import pandas as pd
import pickle as pkl


class BaseSynthesizer(object):

    def __init__(self, metadata: SingleTableMetadata, name: str):
        self.metadata = metadata
        self.name = name
        self.locales = ['en-us']
        self.verbose = False
        self.cuda = False
        self.fit_time = None
        self.sample_time = None
        self.fit_memo = None
        self.sample_memo = None

    def _create_models(self):
        self.synthesizer = None

    def _create_conditions(self, target: str, occurences: dict):
        conditions = []
        for class_, n_rows in occurences.items():
            if n_rows == 0:
                continue
            c = Condition(
                num_rows=n_rows,
                column_values={target: class_}
            )
            conditions.append(c)

        return conditions

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        self.data = data
        self.synthesizer.fit(data)

    def sample(self, n_rows: int) -> pd.DataFrame:
        generated_data = self.synthesizer.sample(num_rows=n_rows)
        return generated_data
    
    def sample_from_conditions(self, occurences: dict, **kwargs) -> pd.DataFrame:
        target = kwargs["target"]
        conditions = self._create_conditions(target, occurences)

        generated_data = self.synthesizer.sample_from_conditions(conditions)
        return generated_data

    def save(self, name: str):
        with open(name, "wb") as fp:
            pkl.dump(self, fp)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as fp:
            syn = pkl.load(fp)

        return syn
