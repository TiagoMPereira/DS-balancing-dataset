from data_augmentation.synthesizers import BaseSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


class GaussianCopula(BaseSynthesizer):

    def __init__(
        self,
        metadata: SingleTableMetadata,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        numerical_distributions: dict = None,
        default_distribution: str = "beta"
    ):
        name = "GAUSSIAN_COPULA"
        super().__init__(metadata, name=name)
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.numerical_distributions = numerical_distributions
        self.default_distribution = default_distribution
        self._create_model()

    def _create_model(self):
        self.synthesizer = GaussianCopulaSynthesizer(
            metadata=self.metadata,
            locales=self.locales,
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            numerical_distributions=self.numerical_distributions,
            default_distribution=self.default_distribution
        )
