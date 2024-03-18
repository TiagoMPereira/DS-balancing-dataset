import pandas as pd
from data_balancing.autoML_frameworks.utils import eval, infer_task_type, EXEC_TIME_SECONDS, EXEC_TIME_MINUTES, GET_SEED
import random
import numpy as np


random.seed(GET_SEED())
np.random.seed(GET_SEED())


def fit_eval(X_train, X_test, y_train, y_test):
    import autokeras as ak

    multi_label = False

    autokeras = ak.StructuredDataClassifier(multi_label=multi_label,
                                            max_trials=3,
                                            overwrite=True,
                                            seed=GET_SEED())

    autokeras.fit(X_train, y_train, epochs=1000)

    if multi_label:
        y_pred = autokeras.predict(X_test).astype(int)
    else:
        y_pred = autokeras.predict(X_test).astype(int).flatten()

    results = eval(y_test, y_pred)
    return results
