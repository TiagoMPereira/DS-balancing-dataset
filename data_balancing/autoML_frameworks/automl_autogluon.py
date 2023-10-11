import pandas as pd
from data_balancing.autoML_frameworks.utils import eval

SEED = 42
EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
    

def fit_eval(X_train, X_test, y_train, y_test):
    from autogluon.tabular import TabularPredictor

    train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
    test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

    clf = TabularPredictor(eval_metric='accuracy', label='class')

    clf = clf.fit(time_limit=EXEC_TIME_SECONDS, train_data=train_df)

    y_test = test_df['class'].values
    y_pred = clf.predict(test_df)

    results = eval(y_test, y_pred)
    return results
