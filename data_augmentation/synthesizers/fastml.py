from sdv.lite import SingleTablePreset
from data_augmentation.synthesizers import BaseSynthesizer
from sdv.metadata import SingleTableMetadata


class FASTML(BaseSynthesizer):

    def __init__(self, metadata: SingleTableMetadata):
        name = "FAST_ML"
        super().__init__(metadata, name=name)
        self._create_model()

    def _create_model(self):
        self.synthesizer = SingleTablePreset(
            metadata=self.metadata,
            name=self.name,
            locales=self.locales
        )
