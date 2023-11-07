#!/bin/bash

# Datasets
# - binary:
#   - 37        diabetes                        768x9x2
#   - 44        spambase            		    4601x58x2
#   - 1462      banknote-authentication         1372x6x2
#   - 1479      hill-valley         		    1212x101x2
#   - 1510      wdbc                		    569x31x2
# - multiclass:
#   - 23        contraceptive-method-choice	    1473x10x10
#   - 181       yeast				            1484x9x10
#   - 1466      cardiotocography         	    2126x24x10
#   - 40691     wine-quality        		    1599x12x6
#   - 40975     car      		                1728x7x4


password=LINUX_SUDO_PASSWORD
datasets=("openml_37" "openml_44" "openml_1462" "openml_1479" "openml_1510" "openml_23" "openml_181" "openml_1466" "openml_40691" "openml_40975")
targets=("class" "class" "Class" "Class" "Class" "Contraceptive_method_used" "class_protein_localization" "Class" "class" "class")

echo Script execution started at $(date).

# Preparation
echo ======== Preparation ========
echo Started cleaning files from previous executions at $(date).
rm -rf __pycache* &> /dev/null
rm -rf artifacts/optuna_models/* &> /dev/null
rm -rf autobalancer_models* &> /dev/null
rm -rf autobalancer_results* &> /dev/null
rm -rf autobalancer_optuna_results* &> /dev/null
rm -rf Autogluon* &> /dev/null
rf -rf gama* &> /dev/null
rm -rf results* &> /dev/null
rm -rf structured* &> /dev/null
rm -rf venv-* &> /dev/null
rm artifacts/autobalancer_datasets/*_train.csv &> /dev/null
rm artifacts/autobalancer_datasets/*_test.csv &> /dev/null
rm artifacts/exec_logs/* &> /dev/null
rm artifacts/optuna_dbs/* &> /dev/null
rm artifacts/sdv_cache/* &> /dev/null
echo Finished cleaning files from previous executions at $(date).

for ((i=0; i<${#datasets[@]}; i++)); do 

    dataset_name="${datasets[i]}"
    target_name="${targets[i]}"

    echo ======== Processing ========
    echo Started processing dataset $id at $(date).

    # AutoGluon
    echo ======== AutoGluon ========
    python -m venv venv-autogluon
    source ./venv-autogluon/bin/activate
    python -m pip install --upgrade pip
    python -m pip install --upgrade setuptools wheel future autogluon
    python -m pip install optuna imbalanced-learn sdmetrics sdv torch
    python ./pipeline_optuna_autobalancing.py $dataset_name $target_name autogluon

    # # AutoKeras
    # echo ======== AutoKeras ========
    # python -m venv venv-autokeras
    # source ./venv-autokeras/bin/activate
    # python -m pip install --upgrade pip
    # python -m pip install --upgrade setuptools wheel future autokeras
    # python -m pip install optuna imbalanced-learn sdmetrics sdv scikit-learn
    # python ./pipeline_optuna_autobalancing.py $dataset_name $target_name autokeras

    # # AutoPyTorch
    # echo ======== AutoPyTorch ========
    # python -m venv venv-autopytorch
    # source ./venv-autopytorch/bin/activate
    # echo $password | sudo -S apt-get install ffmpeg libsm6 libxext6 -y
    # python -m pip install --upgrade pip
    # python -m pip install --upgrade setuptools wheel future autoPyTorch
    # python -m pip install optuna imbalanced-learn sdmetrics sdv swig torch
    # python ./pipeline_optuna_autobalancing.py $dataset_name $target_name autopytorch

    # # AutoSklearn
    # echo ======== AutoSklearn ========
    # python -m venv venv-autosklearn
    # source ./venv-autosklearn/bin/activate
    # echo $password | sudo -S apt-get install build-essential swig python3-dev -y
    # python -m pip install --upgrade pip
    # python -m pip install --upgrade setuptools wheel future auto-sklearn
    # python -m pip install optuna imbalanced-learn sdmetrics sdv
    # python ./pipeline_optuna_autobalancing.py $dataset_name $target_name autosklearn

    # # EvalML
    # echo ======== EvalML ========
    # python -m venv venv-evalml
    # source ./venv-evalml/bin/activate
    # python -m pip install --upgrade pip
    # python -m pip install --upgrade setuptools wheel future evalml
    # python -m pip install optuna imbalanced-learn sdmetrics sdv
    # python ./pipeline_optuna_autobalancing.py $dataset_name $target_name evalml

    # # FLAML
    # echo ======== FLAML ========
    # python -m venv venv-flaml
    # source ./venv-flaml/bin/activate
    # python -m pip install --upgrade pip
    # python -m pip install --upgrade setuptools wheel future flaml[automl]
    # python -m pip install optuna imbalanced-learn sdmetrics sdv
    # python ./pipeline_optuna_autobalancing.py $dataset_name $target_name flaml

    # # GAMA
    # echo ======== GAMA ========
    # python -m venv venv-gama
    # source ./venv-gama/bin/activate
    # python -m pip install --upgrade pip
    # python -m pip install --upgrade setuptools wheel future gama
    # python -m pip install optuna imbalanced-learn sdmetrics sdv
    # sed -i 's/ SCORERS/ _SCORERS/' ./venv-gama/lib/python3.8/site-packages/gama/utilities/metrics.py
    # python ./pipeline_optuna_autobalancing.py $dataset_name $target_name gama

    # # H2O
    # echo ======== H2O ========
    # python -m venv venv-h2o
    # source ./venv-h2o/bin/activate
    # echo $password | sudo -S apt-get install default-jre -y
    # python -m pip install --upgrade pip
    # python -m pip install --upgrade setuptools wheel future h2o
    # python -m pip install optuna imbalanced-learn sdmetrics sdv requests tabulate scikit-learn pandas
    # python ./pipeline_optuna_autobalancing.py $dataset_name $target_name h2o

    # # LightAutoML
    # echo ======== LightAutoML ========
    # python -m venv venv-lightautoml
    # source ./venv-lightautoml/bin/activate
    # python -m pip install --upgrade pip
    # python -m pip install --upgrade setuptools wheel future lightautoml
    # python -m pip install optuna imbalanced-learn sdmetrics sdv
    # python ./pipeline_optuna_autobalancing.py $dataset_name $target_name lightautoml

    # # TPOT
    # echo ======== TPOT ========
    # python -m venv venv-tpot
    # source ./venv-tpot/bin/activate
    # python -m pip install --upgrade pip
    # python -m pip install --upgrade setuptools wheel future tpot
    # python -m pip install optuna imbalanced-learn sdmetrics sdv deap update_checker tqdm stopit xgboost torch
    # python ./pipeline_optuna_autobalancing.py $dataset_name $target_name tpot

    echo Finished processing dataset $dataset_name at $(date).

done

echo Script execution finished at $(date).
