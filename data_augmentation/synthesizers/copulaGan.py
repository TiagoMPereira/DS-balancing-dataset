from data_augmentation.synthesizers import BaseSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CopulaGANSynthesizer


class CopulaGAN(BaseSynthesizer):

    def __init__(
        self,
        metadata: SingleTableMetadata,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        numerical_distributions: dict = {},
        default_distribution: str = "beta",
        epochs: int = 300
    ):
        name = "COPULA_GAN"
        super().__init__(metadata, name=name)
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.numerical_distributions = numerical_distributions
        self.default_distribution = default_distribution
        self.epochs = epochs
        self._create_model()

    def _create_model(self):
        self.synthesizer = CopulaGANSynthesizer(
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