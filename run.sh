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


datasets=("openml_37" "openml_44" "openml_1462" "openml_1479" "openml_1510" "openml_23" "openml_181" "openml_1466" "openml_40691" "openml_40975")
targets=("class" "class" "Class" "Class" "Class" "Contraceptive_method_used" "class_protein_localization" "Class" "class" "class")

echo Script execution started at $(date).

# Preparation
echo ======== Preparation ========
echo Started cleaning files from previous executions at $(date).
rm -rf __pycache* &> /dev/null
rm -rf Autogluon* &> /dev/null
rf -rf gama_* &> /dev/null
rm -rf structured* &> /dev/null
rm -rf venv-* &> /dev/null
rm *.log &> /dev/null
rm results/* &> /dev/null
echo Finished cleaning files from previous executions at $(date).

for ((i=0; i<${#datasets[@]}; i++)); do 

    dataset_name="${datasets[i]}"
    target_name="${targets[i]}"

    echo ======== Processing ========
    echo Started processing dataset $id at $(date).

    # AutoGluon
    echo ======== AutoGluon ========
    python3.8 -m venv venv-autogluon
    source ./venv-autogluon/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel imbalanced-learn==0.11.0 sdmetrics==0.11.1 sdv==1.4.0 torch==1.12+cpu torchvision==0.13.0+cpu torchtext==0.13.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html autogluon
    python3.8 ./pipeline_autobalancing.py $dataset_name $target_name autogluon

    # AutoKeras
    echo ======== AutoKeras ========
    python3.8 -m venv venv-autokeras
    source ./venv-autokeras/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel imbalanced-learn==0.11.0 sdmetrics==0.11.1 sdv==1.4.0 git+https://github.com/keras-team/keras-tuner.git scikit-learn autokeras
    python3.8 ./pipeline_autobalancing.py $dataset_name $target_name autokeras

    # # AutoPyTorch
    echo ======== AutoPyTorch ========
    python3.8 -m venv venv-autopytorch
    source ./venv-autopytorch/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel imbalanced-learn==0.11.0 sdmetrics==0.11.1 sdv==1.4.0 swig torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu autoPyTorch
    python3.8 ./pipeline_autobalancing.py $dataset_name $target_name autopytorch

    # # AutoSklearn
    echo ======== AutoSklearn ========
    python3.8 -m venv venv-autosklearn
    source ./venv-autosklearn/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel imbalanced-learn==0.11.0 sdmetrics==0.11.1 sdv==1.4.0 auto-sklearn
    python3.8 ./pipeline_autobalancing.py $dataset_name $target_name autosklearn

    # # EvalML
    echo ======== EvalML ========
    python3.8 -m venv venv-evalml
    source ./venv-evalml/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel imbalanced-learn==0.11.0 sdmetrics==0.11.1 sdv==1.4.0 evalml
    python3.8 ./pipeline_autobalancing.py $dataset_name $target_name evalml

    # # FLAML
    echo ======== FLAML ========
    python3.8 -m venv venv-flaml
    source ./venv-flaml/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel imbalanced-learn==0.11.0 sdmetrics==0.11.1 sdv==1.4.0 flaml
    python3.8 ./pipeline_autobalancing.py $dataset_name $target_name flaml

    # # GAMA
    echo ======== GAMA ========
    python3.8 -m venv venv-gama
    source ./venv-gama/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel imbalanced-learn==0.11.0 sdmetrics==0.11.1 sdv==1.4.0 gama
    python3.8 ./pipeline_autobalancing.py $dataset_name $target_name gama

    # # H2O
    echo ======== H2O ========
    python3.8 -m venv venv-h2o
    source ./venv-h2o/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel imbalanced-learn==0.11.0 sdmetrics==0.11.1 sdv==1.4.0 requests tabulate future scikit-learn pandas h2o
    python3.8 ./pipeline_autobalancing.py $dataset_name $target_name h2o

    # # LightAutoML
    echo ======== LightAutoML ========
    python3.8 -m venv venv-lightautoml
    source ./venv-lightautoml/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel imbalanced-learn==0.11.0 sdmetrics==0.11.1 sdv==1.4.0 lightautoml
    python3.8 ./pipeline_autobalancing.py $dataset_name $target_name lightautoml

    # # TPOT
    echo ======== TPOT ========
    python3.8 -m venv venv-tpot
    source ./venv-tpot/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel imbalanced-learn==0.11.0 sdmetrics==0.11.1 sdv==1.4.0 deap update_checker tqdm stopit xgboost torch tpot
    python3.8 ./pipeline_autobalancing.py $dataset_name $target_name tpot

    echo Finished processing dataset $dataset_name at $(date).

done

echo Script execution finished at $(date).
