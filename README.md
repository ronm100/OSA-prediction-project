# OSA-prediction-project
This project provides the infrastructure needed to train a neural network clasiffier of OSA(Obstructive Sleep Apnea) sevrity of patients using the SPO_2 signals
of the SHHS1 dataset.

# Files:
##  data_categorization.py:
Contains functions that plot some different distributions of our dataset.

##  data_processing.py :
Contains functions that apply some preprocessing to the input signals.

##  data_categorization.py:
Contains functions that plot some different distributions of our dataset.

##  edf_pre_proccecing.py:
Contains functions that convert raw edf input into a csv form, and match labels to inputs.

##  hyperparam_opt.py:
Contains infrastructure to optimize the hyperparametes of our neural networks using the optuna lib.

##  model_testing.py:
Contains infrastructure to test a model using our test set, with logging.

##  nn_training.py:
Contains infrastructure to train different architectures using our dataset.
