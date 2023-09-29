import optuna
import pandas as pd
import random
from data_augmentation.utils.sampling_strategies import (
    oversampling_strategy, undersampling_strategy)
import os


DATASET_NAME = "openml_37"
DATASET_TARGET = "class"
DATASET_PATH = "./datasets/"

OVERSAMPLING_METHODS = [
    "adasyn", "ctgan", "copulagan", "fastml",
    "gaussiancopula", "random", "smote", "tvae"
]

UNDERSAMPLING_METHODS = [
    "random"
]

OVERSAMPLING_THRESHOLDS = [0, 0.25, 0.5, 1, 5, "auto"]
UNDERSAMPLING_THRESHOLDS = [0, 0.05, 0.1, 0.2, 0.3, "auto"]

def get_under_method(
    id_: str, method_name: str, method, dataset: pd.DataFrame,
    x_col: list, y_col: str, **kwargs
):
    models_path = "./models/"+id_+"/undersampling/"
    method_path = models_path+method_name+".uds"
    
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    if os.path.exists(method_path):
        return method.load(method_path)
    else:
        under_method = method(**kwargs)
        under_method.fit(dataset, X=x_col, y=y_col)
        under_method.save(method_path)
        return under_method

def get_over_method(
    id_: str, method_name: str, method, dataset: pd.DataFrame,
    method_kwargs = {}, fit_kwargs = {}
):
    models_path = "./models/"+id_+"/oversampling/"
    method_path = models_path+method_name+".ovs"
    
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    if os.path.exists(method_path):
        return method.load(method_path)
    else:
        under_method = method(**method_kwargs)
        under_method.fit(dataset, **fit_kwargs)
        under_method.save(method_path)
        return under_method


def objective(trial):

    # Carregar datasets de treino e teste (SEPARADOS)
    train_dataset = pd.read_csv(DATASET_PATH+DATASET_NAME+"_train.csv")
    test_dataset = pd.read_csv(DATASET_PATH+DATASET_NAME+"_test.csv")

    # Separar X e Y
    X_train = train_dataset.drop(columns=DATASET_TARGET)
    X_test = test_dataset.drop(columns=DATASET_TARGET)

    y_train = train_dataset[DATASET_TARGET]
    y_test = test_dataset[DATASET_TARGET]

    # Selecionar método de undersampling
    under_method = trial.suggest_categorical(
        "undersampling_method", UNDERSAMPLING_METHODS)

    # Selecionar método de oversampling
    over_method = trial.suggest_categorical(
        "oversampling_method", OVERSAMPLING_METHODS)

    # Selecionar limite de undersampling -> Categórico
    under_threshold = trial.suggest_categorical(
        "undersampling_threshold", UNDERSAMPLING_THRESHOLDS)

    # Selecionar limite de oversampling -> Categórico
    over_threshold = trial.suggest_categorical(
        "oversampling_threshold", OVERSAMPLING_THRESHOLDS)

    # TODO: Aplicar undersampling
    # TODO: Aplicar oversampling

    # TODO: Criar modelo do AutoGluon

    # TODO: Predict do X de teste

    # TODO: Avaliar métrica do predict com Y teste

    # TODO: Retornar avaliação


    return random.randint(0, 100)


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print(study.best_trial)