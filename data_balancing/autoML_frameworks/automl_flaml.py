import pandas as pd
from data_balancing.autoML_frameworks.utils import eval, infer_task_type, EXEC_TIME_SECONDS, EXEC_TIME_MINUTES, SEED
import random
import numpy as np


random.seed(SEED)
np.random.seed(SEED)
    

def fit_eval(X_train, X_test, y_train, y_test):
    from flaml import AutoML

    clf = AutoML()
    print(f'1 -> {clf=}')

    clf.fit(X_train, y_train, metric="accuracy", task="classification", time_budget=EXEC_TIME_SECONDS)
    print(f'2 -> {clf=}')

    y_pred = clf.predict(X_test)
    print(f'3 -> {y_pred=}')

    results = eval(y_test, y_pred)
    return results
