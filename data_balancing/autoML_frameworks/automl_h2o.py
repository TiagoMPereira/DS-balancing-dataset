import pandas as pd
from data_balancing.autoML_frameworks.utils import eval, infer_task_type, EXEC_TIME_SECONDS, EXEC_TIME_MINUTES, SEED
import random
import numpy as np


random.seed(SEED)
np.random.seed(SEED)
    

def fit_eval(X_train, X_test, y_train, y_test):
    from h2o.sklearn import H2OAutoMLClassifier

    train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
    test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

    clf = H2OAutoMLClassifier(max_runtime_secs=EXEC_TIME_SECONDS, nfolds=5, seed=SEED, sort_metric='accuracy')

    clf.fit(train_df.drop('class', axis=1).values, train_df['class'].values)

    y_pred = clf.predict(X_test)

    results = eval(y_test, y_pred)
    return results
