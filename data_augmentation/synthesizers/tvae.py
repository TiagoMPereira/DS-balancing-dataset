from data_augmentation.synthesizers import BaseSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer


class TVAE(BaseSynthesizer):

    def __init__(
        self,
        metadata: SingleTableMetadata,
        enforce_min_max_values: bool = True,
        enforce_rounding: bool = True,
        epochs: int = 300
    ):
        name = "TVAE"
        super().__init__(metadata, name=name)
        self.enforce_min_max_values = enforce_min_max_values
        self.enforce_rounding = enforce_rounding
        self.epochs = epochs
        self._create_model()

    def _create_model(self):
        self.synthesizer = TVAESynthesizer(
            metadata=self.metadata,
            enforce_min_max_values=self.enforce_min_max_values,
            enforce_rounding=self.enforce_rounding,
            epochs=self.epochs,
            cuda=self.cuda
        )