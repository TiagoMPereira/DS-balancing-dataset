from flaml import AutoML
from data_balancing.autoML_frameworks.utils import eval

SEED = 42
EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
    

def fit_eval(X_train, X_test, y_train, y_test):

    clf = AutoML()

    clf.fit(X_train, y_train, metric="accuracy", task="classification", time_budget=EXEC_TIME_SECONDS)

    y_pred = clf.predict(X_test)

    results = eval(y_test, y_pred)
    return results
