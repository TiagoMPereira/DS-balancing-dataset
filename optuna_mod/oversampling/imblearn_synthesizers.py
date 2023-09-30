from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from optuna_mod.oversampling.base import ImblearnOversampling
import pandas as pd


class ADASYNSynthesizer(ImblearnOversampling):
    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = ADASYN(
            sampling_strategy="auto",
            random_state=self.random_state
        )


class RandomSynthesizer(ImblearnOversampling):
    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = RandomOverSampler(
            sampling_strategy="auto",
            random_state=self.random_state
        )


class SMOTESynthesizer(ImblearnOversampling):

    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = SMOTE(
            sampling_strategy="auto",
            random_state=self.random_state
        )



    

    
