import pandas as pd
import optuna

from data_balancing.autoML_frameworks.utils import N_TRIALS, SEED
from data_balancing.optimization.optimizer import Objective


def optuna_search_tpe(
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
        project_id = _id,
    )

    study = optuna.create_study(
        study_name=_id,
        direction="maximize",
        storage=f"sqlite:///artifacts/optuna_dbs/{_id}.db",
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner(),
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    # study.optimize(objective, timeout=60)
    study.optimize(objective, timeout=60*60, n_trials=N_TRIALS)

    total_results = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    return total_results
