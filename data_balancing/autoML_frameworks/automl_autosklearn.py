
from data_balancing.autoML_frameworks.utils import eval

SEED = 42
EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
    

def fit_eval(X_train, X_test, y_train, y_test):
    from autosklearn.classification import AutoSklearnClassifier

    clf = AutoSklearnClassifier(time_left_for_this_task=EXEC_TIME_SECONDS, resampling_strategy="cv", resampling_strategy_arguments={"folds": 5}, seed=SEED)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = eval(y_test, y_pred)
    return results
