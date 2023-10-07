import autokeras as ak
from data_balancing.autoML_frameworks.utils import eval

SEED = 42
EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
    

def fit_eval(X_train, X_test, y_train, y_test):

    multi_label = False

    autokeras = ak.StructuredDataClassifier(multi_label=multi_label, max_trials=3, overwrite=True, seed=SEED)

    autokeras.fit(X_train, y_train, epochs=1000)

    if multi_label:
        y_pred = autokeras.predict(X_test).astype(int)
    else:
        y_pred = autokeras.predict(X_test).astype(int).flatten()

    results = eval(y_test, y_pred)
    return results
