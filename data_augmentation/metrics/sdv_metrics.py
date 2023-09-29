import pandas as pd
import numpy as np
from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity
from sdmetrics.single_column import KSComplement, TVComplement


def evaluate_single_columns(
    original_dataset: pd.DataFrame,
    preocessed_dataset: pd.DataFrame,
    target: str,
    continuous_columns: list
):
    
    discrete_columns = [
        col for col in original_dataset.columns
        if col not in continuous_columns and col != target
    ]

    _classes = original_dataset[target].unique()

    discrete_metric = []
    continuous_metric = []

    for _class in _classes:
        for col in continuous_columns:
            try:
                _ks = KSComplement.compute(
                    real_data=original_dataset.loc[original_dataset[target] == _class, col],
                    synthetic_data=preocessed_dataset.loc[preocessed_dataset[target] == _class, col]
                )
            except:
                _ks = 0
            continuous_metric.append(_ks)

        for col in discrete_columns:
            try:
                _tv = TVComplement.compute(
                    real_data=original_dataset.loc[original_dataset[target] == _class, col],
                    synthetic_data=preocessed_dataset.loc[preocessed_dataset[target] == _class, col]
                )
            except:
                _tv = 0
            discrete_metric.append(_tv)

    return {
        "ks": np.median(continuous_metric),
        "tv": np.median(discrete_metric),
    }


def evaluate_pairs_columns(
    original_dataset: pd.DataFrame,
    preocessed_dataset: pd.DataFrame,
    target: str,
    continuous_columns: list
):
    
    discrete_columns = [
        col for col in original_dataset.columns
        if col not in continuous_columns and col != target
    ]

    _classes = original_dataset[target].unique()

    discrete_metric = []
    continuous_metric = []

    for _class in _classes:

        for i in range(len(continuous_columns)):
            for j in range(i, len(continuous_columns)):
                if i == j:
                    continue
                try:
                    _current_correlation = CorrelationSimilarity.compute(
                        real_data=original_dataset.loc[original_dataset[target] == _class, [continuous_columns[i], continuous_columns[j]]],
                        synthetic_data=preocessed_dataset.loc[preocessed_dataset[target] == _class, [continuous_columns[i], continuous_columns[j]]],
                        coefficient='Pearson'
                    )
                except:
                    _current_correlation = 0
                continuous_metric.append(_current_correlation)

        for i in range(len(discrete_columns)):
            for j in range(i, len(discrete_columns)):
                if i == j:
                    continue
                try:
                    _current_contingency = ContingencySimilarity.compute(
                        real_data=original_dataset.loc[original_dataset[target] == _class, [discrete_columns[i], discrete_columns[j]]],
                        synthetic_data=preocessed_dataset.loc[preocessed_dataset[target] == _class, [discrete_columns[i], discrete_columns[j]]],
                    )
                except:
                    _current_contingency = 0
                discrete_metric.append(_current_contingency)

    return {
        "correlation": np.median(continuous_metric),
        "contingency": np.median(discrete_metric),
    }

if __name__ == "__main__":
    data = pd.read_csv("./datasets/synthetic_dataset2.csv")
    generated = pd.read_csv("./results/synthetic_dataset2-u_random_0-o_adasyn_0.5.csv")

    continuous = ["col_0", "col_1", "col_2", "col_3", "col_4", "col_5"]
    print(evaluate_single_columns(data, generated, "class", continuous))
    print(evaluate_pairs_columns(data, generated, "class", continuous))
