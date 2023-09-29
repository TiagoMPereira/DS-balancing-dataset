from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import pickle as pkl


class RandomUnder():

    def __init__(self):
        name = "RANDOM_UNDERSAMPLER"
        self.random_state = 42
        self._create_model()

    def _create_model(self):
        self.synthesizer = RandomUnderSampler(
            sampling_strategy="auto",
            random_state=self.random_state
        )

    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        self.X = kwargs.get("X", [])
        self.y = kwargs.get("y", "")
        self.data = data

    def sample(self, n_rows: int) -> pd.DataFrame:
        return pd.DataFrame()
    
    def _get_sample_strategy(self, n_drop: dict) -> dict:
        current_values = dict(self.data[self.y].value_counts())

        return {
            class_: int(current_values[class_] - n_drop[class_])
            for class_ in n_drop.keys()
        }
    
    def sample_from_conditions(self, conditions: dict) -> pd.DataFrame:

        # Conditions tem a quantidade de dados a serem apagados
        # O input deve ser a quantidade de dados ao fim do processamento
        # logo: total - conditions

        final_rows = self._get_sample_strategy(conditions)

        self.synthesizer.sampling_strategy = final_rows

        X_resampled, y_resampled = self.synthesizer.fit_resample(
            X=self.data[self.X], y=self.data[self.y]
        )

        generated_data = pd.concat([X_resampled, y_resampled], axis=1)
        generated_data = generated_data.reset_index(drop=True)
        return generated_data

    def save(self, name: str):
        with open(name, "wb") as fp:
            pkl.dump(self, fp)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as fp:
            syn = pkl.load(fp)

        return syn