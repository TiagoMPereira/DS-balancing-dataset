import pandas as pd
from data_balancing.autoML_frameworks.utils import eval, infer_task_type, EXEC_TIME_SECONDS, EXEC_TIME_MINUTES, GET_SEED
import random
import numpy as np
import torch


random.seed(GET_SEED())
np.random.seed(GET_SEED())
torch.manual_seed(GET_SEED())
    

def fit_eval(X_train, X_test, y_train, y_test):
    from autoPyTorch.api.tabular_classification import TabularClassificationTask

    clf = TabularClassificationTask(seed=GET_SEED())

    clf.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimize_metric='accuracy',
        budget_type='runtime',
        total_walltime_limit=EXEC_TIME_SECONDS,
        func_eval_time_limit_secs=EXEC_TIME_SECONDS//10,
        memory_limit=None
    )

    y_pred = clf.predict(X_test)

    results = eval(y_test, y_pred)
    return results
