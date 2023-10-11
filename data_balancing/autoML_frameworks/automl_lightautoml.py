import numpy as np
import pandas as pd

from data_balancing.autoML_frameworks.utils import eval, infer_task_type

SEED = 42
EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
    

def fit_eval(X_train, X_test, y_train, y_test):
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
    train_df = train_df.rename(columns={i:str(i) for i in train_df.columns})
    test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()
    test_df = test_df.rename(columns={i:str(i) for i in test_df.columns})

    clf = TabularAutoML(task=Task(infer_task_type(y_test), metric='accuracy'), timeout=EXEC_TIME_SECONDS)

    feature_names = [str(i) for i in train_df.columns]
    clf.fit_predict(train_df, roles={'target': 'class'})

    y_test = test_df['class']
    y_pred = np.argmax(clf.predict(test_df).data, axis=1)

    results = eval(y_test, y_pred)
    return results
