import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

SEED = 42
EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60

def _eval(y_test, y_pred):
    def _calculate_score(metric, y_true, y_pred, **kwargs):
        try:
            return float(metric(y_true, y_pred, **kwargs))
        except:
            return float(-1.0)
        
    results = {}
    results.update({"accuracy_score":                           _calculate_score(accuracy_score, y_test, y_pred)})
    results.update({"average_precision_score":                  _calculate_score(average_precision_score, y_test, y_pred)})
    results.update({"balanced_accuracy_score":                  _calculate_score(balanced_accuracy_score, y_test, y_pred)})
    results.update({"cohen_kappa_score":                        _calculate_score(cohen_kappa_score, y_test, y_pred)})
    results.update({"f1_score_macro":                           _calculate_score(f1_score, y_test, y_pred, average="macro")})
    results.update({"f1_score_micro":                           _calculate_score(f1_score, y_test, y_pred, average="micro")})
    results.update({"f1_score_weighted":                        _calculate_score(f1_score, y_test, y_pred, average="weighted")})
    results.update({"matthews_corrcoef":                        _calculate_score(matthews_corrcoef, y_test, y_pred)})
    results.update({"precision_score":                          _calculate_score(precision_score, y_test, y_pred)})
    results.update({"recall_score":                             _calculate_score(recall_score, y_test, y_pred)})
    results.update({"roc_auc_score":                            _calculate_score(roc_auc_score, y_test, y_pred)})
    results.update({"coverage_error":                           _calculate_score(coverage_error, y_test, y_pred)})
    results.update({"label_ranking_average_precision_score":    _calculate_score(label_ranking_average_precision_score, y_test, y_pred)})
    results.update({"label_ranking_loss":                       _calculate_score(label_ranking_loss, y_test, y_pred)})

    return results
    

def fit_eval(X_train, X_test, y_train, y_test):

    train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
    test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

    clf = TabularPredictor(eval_metric='accuracy', label='class')

    clf = clf.fit(time_limit=EXEC_TIME_SECONDS, train_data=train_df)

    y_test = test_df['class'].values
    y_pred = clf.predict(test_df)

    results = _eval(y_test, y_pred)
    return results
