import pandas as pd
from data_balancing.autoML_frameworks import (automl_autogluon,
                                              automl_autokeras,
                                              automl_autopytorch,
                                              automl_autosklearn,
                                              automl_evalml,
                                              automl_flaml,
                                              automl_gama,
                                              automl_h2o,
                                              automl_lightautoml,
                                              automl_tpot)
from data_balancing.oversampling import (CTGAN,
                                         FASTML,
                                         TVAE,
                                         ADASYNSynthesizer,
                                         CopulaGAN,
                                         GaussianCopula,
                                         RandomSynthesizer,
                                         SMOTESynthesizer)
from data_balancing.undersampling import RandomUnder
from data_balancing.utils.load_methods import (get_over_method,
                                               get_under_method)
from data_balancing.utils.sampling_strategies import (oversampling_strategy,
                                                      undersampling_strategy)


UNDERSAMPLING_METHODS = {
    "random": RandomUnder
}
OVERSAMPLING_METHODS = {
    "adasyn": ADASYNSynthesizer,
    "ctgan": CTGAN,
    "copulagan": CopulaGAN,
    "fastml": FASTML,
    "gaussiancopula": GaussianCopula,
    "random": RandomSynthesizer,
    "smote": SMOTESynthesizer,
    "tvae": TVAE
}

UNDERSAMPLING_THRESHOLDS = [0, 0.0625, 0.125, 0.25, 0.5, "auto"]
OVERSAMPLING_THRESHOLDS = [0, 0.25, 0.5, 1, 2, "auto"]

FRAMEWORKS = {
    "autogluon": automl_autogluon,
    "autokeras": automl_autokeras,
    "autopytorch": automl_autopytorch,
    "autosklearn": automl_autosklearn,
    "evalml": automl_evalml,
    "flaml": automl_flaml,
    "gama": automl_gama,
    "h2o": automl_h2o,
    "lightautoml": automl_lightautoml,
    "tpot": automl_tpot
}


def grid_search(
    train_dataset: pd.DataFrame, test_dataset: pd.DataFrame,
    target: str, dataset_name: str, framework_name: str
):
    total_results = []
    
    class_occurences = {
        k: int(v)
        for k, v in dict(train_dataset[target].value_counts()).items()
    }

    for over_method_name in OVERSAMPLING_METHODS.keys():
        over_method = get_over_method(
            dataset=train_dataset,
            target=target,
            method_name=over_method_name,
            project_id=dataset_name
        )

        for over_threshold in OVERSAMPLING_THRESHOLDS:

            for under_method_name in UNDERSAMPLING_METHODS.keys():
                under_method = get_under_method(
                    dataset=train_dataset,
                    target=target,
                    method_name=under_method_name,
                    project_id=dataset_name
                )

                for under_threshold in UNDERSAMPLING_THRESHOLDS:
                    _id = (f"{dataset_name}_{framework_name}_"
                           f"{over_method_name}-{over_threshold}_"
                           f"{under_method_name}-{under_threshold}")
                    print(_id)
                    try:
                        balanced_data = train_dataset.copy()

                        # Generate oversampled data
                        if over_threshold != 0:
                            # Get oversampling strategy
                            over_strategy = oversampling_strategy(
                                class_occurences, over_threshold
                            )
                            balanced_data = over_method.resample(over_strategy)

                        # Generate undersampled data
                        if under_threshold != 0:
                            # Get undersampling strategy
                            under_strategy = undersampling_strategy(
                                class_occurences, under_threshold
                            )
                            balanced_data = under_method.resample(
                                under_strategy, balanced_data
                            )

                        # SCORE
                        X_train = balanced_data.drop(columns=target)
                        y_train = balanced_data[target]

                        X_test = test_dataset.drop(columns=target)
                        y_test = test_dataset[target]
                        score = FRAMEWORKS[framework_name].fit_eval(
                            X_train=X_train, X_test=X_test,
                            y_train=y_train, y_test=y_test
                        )
                    except:
                        score = {}
                    
                    report = {
                        "_id": _id,
                        "dataset_name": dataset_name,
                        "framework_name": framework_name,
                        "oversampling-method": over_method_name,
                        "oversampling-threshold": over_threshold,
                        "undersampling-method": under_method_name,
                        "undersampling-threshold": under_threshold,
                        "score": score
                    }
                    total_results.append(report)

    return total_results
