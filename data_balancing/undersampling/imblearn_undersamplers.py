import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

from data_balancing.undersampling.base import ImblearnUndersampling

class RandomUnder(ImblearnUndersampling):
    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = RandomUnderSampler(
            sampling_strategy="auto",
            random_state=self.random_state
        )


class ClusterCentroidsUnder(ImblearnUndersampling):
    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = ClusterCentroids(
            sampling_strategy="auto",
            random_state=self.random_state
        )
