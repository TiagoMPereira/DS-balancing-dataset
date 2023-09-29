from data_augmentation.synthesizers import BaseSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


class CTGAN(BaseSynthesizer):

    def __init__(
        self,
        metadata: SingleTableMetadata,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        epochs: int = 300
    ):
        name = "CTGAN"
        super().__init__(metadata, name=name)
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.epochs = epochs
        self._create_model()

    def _create_model(self):
        self.synthesizer = CTGANSynthesizer(
            metadata=self.metadata,
            locales=self.locales,
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            epochs=self.epochs,
            verbose=self.verbose,
            cuda=self.cuda
        )
