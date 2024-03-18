from sdv.lite import SingleTablePreset
from sdv.single_table import (CopulaGANSynthesizer, CTGANSynthesizer,
                              GaussianCopulaSynthesizer, TVAESynthesizer)

from data_balancing.autoML_frameworks.utils import GET_SEED
from data_balancing.oversampling.base import SDVOversampling

import numpy as np
import torch

np.random.seed(GET_SEED())
torch.manual_seed(GET_SEED())


class CopulaGAN(SDVOversampling):

    def __init__(
        self,
        data,
        target,
    ):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        super()._create_model()
        self.model = CopulaGANSynthesizer(
            metadata=self.metadata,
            locales=self.locales,
            verbose=self.verbose,
            cuda=self.cuda
        )


class CTGAN(SDVOversampling):

    def __init__(
        self,
        data,
        target,
    ):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        super()._create_model()
        self.model = CTGANSynthesizer(
            metadata=self.metadata,
            locales=self.locales,
            verbose=self.verbose,
            cuda=self.cuda
        )


class FASTML(SDVOversampling):

    def __init__(self, data, target):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        super()._create_model()
        self.model = SingleTablePreset(
            metadata=self.metadata,
            name="FAST_ML",
            locales=self.locales
        )


class GaussianCopula(SDVOversampling):

    def __init__(
        self,
        data,
        target,
    ):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        super()._create_model()
        self.model = GaussianCopulaSynthesizer(
            metadata=self.metadata,
            locales=self.locales,
        )


class TVAE(SDVOversampling):

    def __init__(
        self,
        data,
        target
    ):
        super().__init__(data=data, target=target)
        self._create_model()

    def _create_model(self):
        super()._create_model()
        self.model = TVAESynthesizer(
            metadata=self.metadata,
            cuda=self.cuda
        )
