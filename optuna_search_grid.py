import pandas as pd
import optuna

from data_balancing.autoML_frameworks.utils import GET_SEED
from data_balancing.optimization.optimizer import Objective


def optuna_search_grid(
    train_dataset: pd.DataFrame, test_dataset: pd.DataFrame,
    target: str, dataset_name: str, framework_name: str
) -> pd.DataFrame:

    o_methods = [
        "adasyn", "ctgan", "copulagan", "fastml",
        "gaussiancopula", "random", "smote", "tvae"
    ]
    u_methods = [
        "random", "clustercentroid", "condensednn", "editednn",
        "instancehardness", "nearmiss", "onesidedselection", "tomeklinks"
    ]
    o_thresh = [0, 0.25, 0.5, 1, 2, "auto"]
    u_thresh = [0, 0.0625, 0.125, 0.25, 0.5, "auto"]

    _id = f"optuna_{dataset_name}_{framework_name}"

    objective = Objective(
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        target = target,
        oversampling_methods = o_methods,
        undersampling_methods = u_methods,
        oversampling_thresholds = o_thresh,
        undersampling_thresholds = u_thresh,
        framework_name = framework_name,
        project_id = _id
    )

    search_space = {
        "oversampling_method": o_methods,
        "undersampling_method": u_methods,
        "oversampling_threshold": o_thresh,
        "undersampling_threshold": u_thresh,
    }

    n_trials = len(o_methods) * len(u_methods) * len(o_thresh) * len(u_thresh)

    study = optuna.create_study(
        study_name=_id,
        direction="maximize",
        storage=f"sqlite:///artifacts/optuna_dbs/{_id}_{GET_SEED()}_{n_trials}.db",
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner(),
        sampler=optuna.samplers.GridSampler(seed=GET_SEED(), search_space=search_space)
    )

    study.optimize(objective, n_trials=n_trials, catch=(Exception,))

    total_results = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    return total_results
