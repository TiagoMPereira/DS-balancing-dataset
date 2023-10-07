from autoPyTorch.api.tabular_classification import TabularClassificationTask
from data_balancing.autoML_frameworks.utils import eval

SEED = 42
EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
    

def fit_eval(X_train, X_test, y_train, y_test):

    clf = TabularClassificationTask(seed=SEED)

    clf.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimize_metric='accuracy',
        budget_type='runtime',
        total_walltime_limit=EXEC_TIME_SECONDS,
        func_eval_time_limit_secs=EXEC_TIME_SECONDS/10,
        memory_limit=8192
    )

    y_pred = clf.predict(X_test)

    results = eval(y_test, y_pred)
    return results
