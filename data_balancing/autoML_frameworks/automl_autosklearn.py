import pandas as pd
from data_balancing.autoML_frameworks.utils import eval, infer_task_type, EXEC_TIME_SECONDS, EXEC_TIME_MINUTES, GET_SEED
import random
import numpy as np


random.seed(GET_SEED())
np.random.seed(GET_SEED())
    

def fit_eval(X_train, X_test, y_train, y_test):
    from autosklearn.classification import AutoSklearnClassifier

    clf = AutoSklearnClassifier(memory_limit=None,
                                time_left_for_this_task=EXEC_TIME_SECONDS, 
                                resampling_strategy="cv", 
                                resampling_strategy_arguments={"folds": 5}, 
                                seed=GET_SEED())

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = eval(y_test, y_pred)
    return results
