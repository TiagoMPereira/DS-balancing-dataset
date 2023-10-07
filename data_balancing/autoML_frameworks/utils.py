from sklearn.metrics import *

def eval(y_test, y_pred):
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


def infer_task_type(y_test):
    num_classes = len(set(y_test))
    if num_classes == 1:
        raise Exception('Malformed data set; num_classes == 1')
    elif num_classes == 2:
        task_type = 'binary'
    else:
        task_type = 'multiclass'
    return task_type