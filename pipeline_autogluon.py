from autoGluon import autogluon
import json
import pandas as pd


if __name__ == "__main__":

    print("LETS GO _ 44") # 19:30
    dataset_name = "openml_44"
    target = "class"
    datasets_path = "./results/"
    metrics_path = "./metrics/"
    base_dataset_path = "./datasets/"

    over_methods = [
        "adasyn", "ctgan", "copulagan", "fastml",
        "gaussiancopula", "random", "smote", "tvae"
    ]
    under_methods =["random"]
    over_thresh = [0, 0.25, 0.5, 1, 5, "auto"]
    under_thresh = [0, 0.05, 0.1, 0.2, 0.3, "auto"]


    for u_method in under_methods:
        for u_thresh in under_thresh:
            for o_method in over_methods:
                for o_thresh in over_thresh:

                    full_dataset_name = f"{dataset_name}-u_{u_method}_{u_thresh}-o_{o_method}_{o_thresh}" 
                    print(full_dataset_name)
                    
                    try:
                        dataset_train = pd.read_csv(datasets_path+full_dataset_name+".csv")
                        dataset_test = pd.read_csv(base_dataset_path+dataset_name+"_test.csv")
                        x_cols = [col for col in dataset_train.columns if col!=target]

                        X_train = dataset_train[x_cols]
                        y_train = dataset_train[target]

                        X_test = dataset_test[x_cols]
                        y_test = dataset_test[target]

                        results = autogluon.fit_eval(X_train, X_test, y_train, y_test)
                    except:
                        results = {}

                    with open(metrics_path+full_dataset_name+".json", "w") as fp:
                        json.dump(results, fp)