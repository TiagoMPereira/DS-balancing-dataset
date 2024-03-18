import pandas as pd
from imblearn.under_sampling import (ClusterCentroids,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     InstanceHardnessThreshold,
                                     NearMiss,
                                     OneSidedSelection,
                                     RandomUnderSampler,
                                     TomekLinks)

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


class CondensedNearestNeighbourUnder(ImblearnUndersampling):
    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = CondensedNearestNeighbour(
            sampling_strategy="auto",
            random_state=self.random_state,
            n_jobs=-1
        )

    def _set_sample_strategy(self, occurences):
        samples = [k for k, v in occurences.items() if v > 0]
        self.model.sampling_strategy = samples


class EditedNearestNeighboursUnder(ImblearnUndersampling):
    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = EditedNearestNeighbours(
            sampling_strategy="auto",
            n_jobs=-1
        )

    def _set_sample_strategy(self, occurences):
        samples = [k for k, v in occurences.items() if v > 0]
        self.model.sampling_strategy = samples


class InstanceHardnessThresholdUnder(ImblearnUndersampling):
    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = InstanceHardnessThreshold(
            sampling_strategy="auto",
            random_state=self.random_state,
            n_jobs=-1
        )


class NearMissUnder(ImblearnUndersampling):
    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = NearMiss(
            sampling_strategy="auto",
            n_jobs=-1
        )


class OneSidedSelectionUnder(ImblearnUndersampling):
    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = OneSidedSelection(
            sampling_strategy="auto",
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def _set_sample_strategy(self, occurences):
        samples = [k for k, v in occurences.items() if v > 0]
        self.model.sampling_strategy = samples


class TomekLinksUnder(ImblearnUndersampling):
    def __init__(self, data: pd.DataFrame, target: str):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        self.model = TomekLinks(
            sampling_strategy="auto",
            n_jobs=-1
        )

    def _set_sample_strategy(self, occurences):
        samples = [k for k, v in occurences.items() if v > 0]
        self.model.sampling_strategy = samples
