from sdv.lite import SingleTablePreset
from sdv.single_table import (CopulaGANSynthesizer, CTGANSynthesizer,
                              GaussianCopulaSynthesizer, TVAESynthesizer)
from optuna_mod.oversampling.base import SDVOversampling


class CopulaGAN(SDVOversampling):

    def __init__(
        self,
        data,
        target,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        numerical_distributions: dict = {},
        default_distribution: str = "beta",
        epochs: int = 300
    ):
        super().__init__(data=data, target=target)
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.numerical_distributions = numerical_distributions
        self.default_distribution = default_distribution
        self.epochs = epochs
        self._create_model()

    def _create_model(self):
        super()._create_model()
        self.model = CopulaGANSynthesizer(
            metadata=self.metadata,
            locales=self.locales,
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            numerical_distributions=self.numerical_distributions,
            default_distribution=self.default_distribution,
            epochs=self.epochs,
            verbose=self.verbose,
            cuda=self.cuda
        )


class CTGAN(SDVOversampling):

    def __init__(
        self,
        data,
        target,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        epochs: int = 300
    ):
        super().__init__(data=data, target=target)
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.epochs = epochs
        self._create_model()

    def _create_model(self):
        super()._create_model()
        self.model = CTGANSynthesizer(
            metadata=self.metadata,
            locales=self.locales,
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            epochs=self.epochs,
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
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        numerical_distributions: dict = None,
        default_distribution: str = "beta"
    ):
        super().__init__(data=data, target=target)
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.numerical_distributions = numerical_distributions
        self.default_distribution = default_distribution
        self._create_model()

    def _create_model(self):
        super()._create_model()
        self.model = GaussianCopulaSynthesizer(
            metadata=self.metadata,
            locales=self.locales,
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            numerical_distributions=self.numerical_distributions,
            default_distribution=self.default_distribution
        )


class TVAE(SDVOversampling):

    def __init__(
        self,
        data,
        target,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        epochs: int = 300
    ):
        super().__init__(data=data, target=target)
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.epochs = epochs
        self._create_model()

    def _create_model(self):
        super()._create_model()
        self.model = TVAESynthesizer(
            metadata=self.metadata,
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            epochs=self.epochs,
            cuda=self.cuda
        )
