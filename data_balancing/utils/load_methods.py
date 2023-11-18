import os
from typing import Any, Dict

import pandas as pd

from data_balancing.oversampling import (CTGAN, FASTML, TVAE, ADASYNSynthesizer,
                                     CopulaGAN, GaussianCopula, IOversampling,
                                     RandomSynthesizer, SMOTESynthesizer)
from data_balancing.undersampling import (ClusterCentroidsUnder,
                                          CondensedNearestNeighbourUnder,
                                          EditedNearestNeighboursUnder,
                                          InstanceHardnessThresholdUnder,
                                          NearMissUnder,
                                          OneSidedSelectionUnder,
                                          RandomUnder,
                                          TomekLinksUnder)
from data_balancing.undersampling import IUndersampling, RandomUnder


def get_over_method(
    dataset: pd.DataFrame,
    target: str,
    method_name: str,
    project_id: str,
    base_path: str = "autobalancer_models/",
):
    
    # METHODS DICT
    methods: Dict[str, IOversampling] = {
        "adasyn": ADASYNSynthesizer,
        "copulagan": CopulaGAN,
        "ctgan": CTGAN,
        "fastml": FASTML,
        "gaussiancopula": GaussianCopula,
        "random": RandomSynthesizer,
        "smote": SMOTESynthesizer,
        "tvae": TVAE,
    }

    if not base_path:
        _method = methods.get(method_name)
        if not _method:
            raise ValueError(f"Invalid method name: {method_name}")
        
        ovr_method = _method(data=dataset, target=target)
        ovr_method.fit()
        return ovr_method

    models_path = base_path+project_id+"/oversampling/"
    method_path = models_path+method_name+".ovs"

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if os.path.exists(method_path):
        _method = methods.get(method_name)
        if not _method:
            raise ValueError(f"Invalid method name: {method_name}")
        return _method.load(method_path)
    else:
        _method = methods.get(method_name)
        if not _method:
            raise ValueError(f"Invalid method name: {method_name}")
        ovr_method = _method(data=dataset, target=target)
        ovr_method.fit()
        ovr_method.save(method_path)
        return ovr_method
    

def get_under_method(
    dataset: pd.DataFrame,
    target: str,
    method_name: str,
    project_id: str,
    base_path: str = "autobalancer_models/"
):
    
    # METHODS DICT
    methods: Dict[str, IUndersampling] = {
    "random": RandomUnder,
    "clustercentroid": ClusterCentroidsUnder,
    "condensednn": CondensedNearestNeighbourUnder,
    "editednn": EditedNearestNeighboursUnder,
    "instancehardness": InstanceHardnessThresholdUnder,
    "nearmiss": NearMissUnder,
    "onesidedselection": OneSidedSelectionUnder,
    "tomeklinks": TomekLinksUnder,
}

    if not base_path:
        _method = methods.get(method_name)
        if not _method:
            raise ValueError(f"Invalid method name: {method_name}")
        
        und_method = _method(data=dataset, target=target)
        und_method.fit()
        return und_method

    models_path = base_path+project_id+"/undersampling/"
    method_path = models_path+method_name+".uds"

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if os.path.exists(method_path):
        _method = methods.get(method_name)
        if not _method:
            raise ValueError(f"Invalid method name: {method_name}")
        return _method.load(method_path)
    else:
        _method = methods.get(method_name)
        if not _method:
            raise ValueError(f"Invalid method name: {method_name}")
        und_method = _method(data=dataset, target=target)
        und_method.fit()
        und_method.save(method_path)
        return und_method
