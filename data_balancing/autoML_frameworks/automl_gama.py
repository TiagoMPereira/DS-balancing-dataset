import pandas as pd
from data_balancing.autoML_frameworks.utils import eval, infer_task_type, EXEC_TIME_SECONDS, EXEC_TIME_MINUTES, SEED
import random
import numpy as np


random.seed(SEED)
np.random.seed(SEED)
    

def fit_eval(X_train, X_test, y_train, y_test):
    from gama import GamaClassifier

    clf = GamaClassifier(max_total_time=EXEC_TIME_SECONDS, store="nothing")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = eval(y_test, y_pred)
    return results
