import warnings

warnings.filterwarnings("ignore")

import json
import logging
from datetime import datetime

import pandas as pd
import os

from data_augmentation.metadata import SDVMetadata
from data_augmentation.synthesizers import (CTGAN, FASTML, TVAE,
                                            ADASYNSynthesizer, CopulaGAN,
                                            GaussianCopula, RandomSynthesizer,
                                            SMOTESynthesizer)
from data_augmentation.undersampling import RandomUnder
from data_augmentation.utils.sampling_strategies import (
    oversampling_strategy, undersampling_strategy)

logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_PATH = "./results/"

def get_under_method(
    id_: str, method_name: str, method: RandomUnder, dataset: pd.DataFrame,
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

def apply_balancing(
    dataset: pd.DataFrame,
    target: str,
    available_under_methods: dict,
    available_under_thresholds: list,
    available_over_methods: dict,
    available_over_thresholds: list,
    id_: str = ""
):
    
    y_col = target
    x_col = [col for col in dataset.columns if col != y_col]

    classes_occurences = dict(dataset[target].value_counts())
    classes_occurences = {k: int(v) for k, v in classes_occurences.items()}

    metadata_methods = ["ctgan", "fastml", "tvae", "copulagan", "gaussiancopula"]
    metadata = SDVMetadata()
    metadata.create_from_df(dataset)

    for under_t in available_under_thresholds:
        under_strategy = undersampling_strategy(classes_occurences, under_t)

        for over_t in available_over_thresholds:
            over_strategy = oversampling_strategy(classes_occurences, over_t)

            for under_m in available_under_methods.keys():
                under_method = get_under_method(
                    id_, under_m, available_under_methods[under_m],
                    dataset, x_col, y_col
                )

                if under_t == 0:
                    undersampled_data = dataset.copy()
                else:
                    undersampled_data = under_method.sample_from_conditions(under_strategy)

                for over_m in oversampling_methods.keys():
                    description = (f"Oversampling: {over_m} {over_t}\n"
                                   f"Undersampling: {under_m} {under_t}\n")
                    
                    print(description)
                    try:
                        if over_m in metadata_methods:
                            over_kwargs = {
                                "method_kwargs": {"metadata": metadata.metadata},
                            }

                        else:
                            over_kwargs = {
                                "fit_kwargs": {"X": x_col, "y": y_col}
                            }

                        over_method = get_over_method(
                            id_, over_m, available_over_methods[over_m],
                            dataset, **over_kwargs
                        )

                        if over_t == 0:
                            oversampled_data = pd.DataFrame()
                        else:
                            oversampled_data = over_method.sample_from_conditions(over_strategy, target=target)

                        balanced_data = pd.concat([undersampled_data, oversampled_data], axis=0)

                        balanced_data.sort_values(by=target, inplace=True)

                        balanced_data.to_csv(f"{BASE_PATH}{id_}-u_{under_m}_{under_t}-o_{over_m}_{over_t}.csv", index=False)
                        
                        balanced_target = {k: int(v) for k, v in dict(balanced_data[target].value_counts()).items()}
                    except Exception as e:
                        logging.error(f"FAIL ON \n{description}\n{e}")
                        balanced_target={}
                        pd.DataFrame().to_csv(f"{BASE_PATH}{id_}-u_{under_m}_{under_t}-o_{over_m}_{over_t}.csv", index=False)


                    result = "Original: "+json.dumps(classes_occurences)+"\nBalanced: "+json.dumps(balanced_target)+"\n\n"
                    print(result)
                    with open(f"{BASE_PATH}{id_}.txt", "a") as f:
                        f.write(description)
                        f.write(result)


if __name__ == "__main__":

    dataset_name = "openml_44"
    target = "class"
    oversampling_thresholds = [0, 0.25, 0.5, 1, 5, "auto"]
    undersampling_thresholds = [0, 0.05, 0.1, 0.2, 0.3, "auto"]

    oversampling_methods = {
        "adasyn": ADASYNSynthesizer,
        "ctgan": CTGAN,
        "copulagan": CopulaGAN,
        "fastml": FASTML,
        "gaussiancopula": GaussianCopula,
        "random": RandomSynthesizer,
        "smote": SMOTESynthesizer,
        "tvae": TVAE
    }

    undersampling_methods = {
        "random": RandomUnder
    }

    dataset = pd.read_csv(f"datasets/{dataset_name}_train.csv")

    id_ = dataset_name
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

    with open(f"{BASE_PATH}{id_}.txt", "a") as f:
        f.write("\n\n==============================================\n")
        f.write(date_time)

    apply_balancing(
        dataset, target,
        undersampling_methods, undersampling_thresholds,
        oversampling_methods, oversampling_thresholds,
        id_=id_
    )