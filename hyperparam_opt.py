import pyedflib
import numpy as np
import csv
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pydot import *
import optuna
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import sys
# from keras import backend as K

ROOT = Path('../../../../..')
LOG_BASE_DIR = ROOT.joinpath('databases/ronmaishlos@staff.technion.ac.il/logs')
CSV_DIR = ROOT.joinpath('databases/ronmaishlos@staff.technion.ac.il/processed_data_as_csv')
STFT_DIR = CSV_DIR.joinpath('stft')

def conv2d_block(input_layer, filters, kernel_size=3, dropout=0.25, pooling='max', pool_size=(1,2)):
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(input_layer)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.AveragePooling2D(pool_size)(conv) if pooling == 'avg' else keras.layers.MaxPooling2D(pool_size)(conv)
    conv = keras.layers.Dropout(dropout)(conv)
    return conv

def conv1d_block(input_layer, filters, kernel_size=3, dropout=0.25, pooling='max', pool_size=2):
    conv = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=2)(input_layer)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.ReLU()(conv)
    conv = keras.layers.AveragePooling1D(pool_size, padding='same')(conv) if pooling == 'avg' else keras.layers.MaxPooling1D(pool_size, padding='same')(conv)
    conv = keras.layers.Dropout(dropout)(conv)
    return conv

def make_model(orig_input_shape, stft_input_shape, trial):
    num_classes = 4
    orig_input_layer = keras.layers.Input(orig_input_shape)
    stft_input_layer = keras.layers.Input(stft_input_shape)

    p1 = trial.suggest_float("dropout_1", 0.0, 0.55) 
    p2 = trial.suggest_float("dropout_2", 0.0, 0.55) 

    # STFT - CNN:
    conv2d_1 = conv2d_block(filters=4, kernel_size=(3,9), dropout=p1, pooling='avg', input_layer=stft_input_layer)
    conv2d_2 = conv2d_block(filters=8, kernel_size=(3,9), dropout=p2, input_layer=conv2d_1)
    conv2d_3 = conv2d_block(filters=16, kernel_size=(3,5), dropout=p2,input_layer=conv2d_2)
    conv2d_4 = conv2d_block(filters=32, pool_size=2, dropout=p2,input_layer=conv2d_3)
    conv2d_5 = conv2d_block(filters=64, pool_size=2, dropout=p2,input_layer=conv2d_4)
    conv2d_6 = conv2d_block(filters=128, pool_size=2, dropout=p2,input_layer=conv2d_5)
    conv2d_7 = conv2d_block(filters=256, pool_size=2, dropout=p2,input_layer=conv2d_6)
    conv2d_8 = conv2d_block(filters=256, pool_size=2, dropout=p2,input_layer=conv2d_7)
    stft_flat = keras.layers.Flatten()(conv2d_8)
    stft_flat = keras.layers.Dense(128, activation="softmax")(stft_flat)

    # Duplo - RNN:
    conv_lstm1 = conv1d_block(orig_input_layer, filters=32, kernel_size=9, dropout=p1, pooling='avg')
    conv_lstm2 = conv1d_block(conv_lstm1, filters=64,dropout=p2, kernel_size=3)
    lstm_1 = keras.layers.LSTM(64)(conv_lstm2)
    lstm_1 = keras.layers.Reshape((64, -1))(lstm_1)
    conv_lstm3 = conv1d_block(lstm_1, filters=128, kernel_size=3,dropout=p2, pooling='avg')
    conv_lstm4 = conv1d_block(conv_lstm3, filters=128,dropout=p2, kernel_size=3)
    lstm_2 = keras.layers.LSTM(128)(conv_lstm4)

    # Duplo CNN:
    conv1 = conv1d_block(orig_input_layer, filters=16, kernel_size=9, dropout=p1, pooling='avg', pool_size=4)
    conv2 = conv1d_block(conv1, filters=16, kernel_size=3,dropout=p2, pool_size=4)
    conv3 = conv1d_block(conv2, filters=32, kernel_size=3,dropout=p2, pool_size=4)
    conv4 = conv1d_block(conv3, filters=32,dropout=p2, kernel_size=3)
    conv5 = conv1d_block(conv4, filters=64,dropout=p2, kernel_size=3, pool_size=4)
    conv6 = conv1d_block(conv5, filters=64,dropout=p2, kernel_size=3)
    conv7 = conv1d_block(conv6, filters=128, dropout=p2,kernel_size=3)
    conv8 = conv1d_block(conv7, filters=256, dropout=p2,kernel_size=3)
    cnn_flat = keras.layers.Flatten()(conv8)
    cnn_flat = keras.layers.Dense(128, activation="softmax")(cnn_flat)

    concatted = keras.layers.Concatenate()([cnn_flat, stft_flat, lstm_2])
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(concatted)

    return keras.models.Model(inputs=[orig_input_layer, stft_input_layer], outputs=output_layer)

def get_training_val_data(stft_name: str):
    sample_length = 21600
    num_of_samples = 5755
    stft_input_dir = STFT_DIR.joinpath(stft_name)

    x = np.array(pd.read_csv(CSV_DIR.joinpath('x_train.csv'), nrows = num_of_samples))[:, 0:sample_length]
    y = np.array(pd.read_csv(CSV_DIR.joinpath('y_train.csv'), nrows = num_of_samples))[0:num_of_samples,1]
    x = x.reshape((x.shape[0], x.shape[1], 1))
    y = y.reshape(num_of_samples, -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25, random_state = 21)

    stft_train = np.load(stft_input_dir.joinpath('x_t'), allow_pickle=True)
    stft_val = np.load(stft_input_dir.joinpath('x_v'), allow_pickle=True)
    
    return x_train, x_val, stft_train, stft_val, y_train, y_val


def train_model(x_train, x_val, stft_train, stft_val, y_train, y_val, log_dir, trial):

    training_inputs = [x_train, stft_train]
    validation_inputs = [x_val, stft_val]
    model = make_model(orig_input_shape=x_train.shape[1:], stft_input_shape=stft_train.shape[1:], trial=trial)
    keras.utils.plot_model(model, to_file = log_dir.joinpath("architecture.png"), show_shapes=True)
    epochs = 1000
    batch_size = 32
    callbacks = [keras.callbacks.ModelCheckpoint(log_dir.joinpath("best_model.h5"), save_best_only=True, monitor="val_sparse_categorical_accuracy"),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=0.00000001),
                 keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=20, verbose=1, restore_best_weights=True, mode='max'),
                 optuna.integration.TFKerasPruningCallback(trial, "val_loss")]

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True) # log=True, will use log scale to interplolate b
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(keras.optimizers, optimizer_name)(learning_rate=lr)
    model.compile (optimizer=optimizer ,loss="sparse_categorical_crossentropy",
                   metrics=['sparse_categorical_accuracy'])

    history = model.fit (training_inputs, y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks, validation_data=(validation_inputs, y_val), verbose=1,)

    # eval_results = model.evaluate(validation_inputs, y_val)
    y_pred_probs = model.predict(validation_inputs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    f1 = "%.3f" % f1_score(y_val, y_pred, average='macro')
    (f1_0,f1_1,f1_2, f1_3) = f1_score(y_val, y_pred, average=None)
    f1_0 = "%.3f" % f1_0
    f1_1 = "%.3f" % f1_1
    f1_2 = "%.3f" % f1_2
    f1_3 = "%.3f" % f1_3
    confusion_m = confusion_matrix(y_val, y_pred)
    auc_score = "%.3f" % roc_auc_score(y_val, y_pred_probs, average='macro', multi_class='ovr')
    acc = accuracy_score(y_val, y_pred)
    recall ="%.3f" % recall_score(y_val, y_pred, average='macro')
    (recall_0, recall_1, recall_2, recall_3) = recall_score(y_val, y_pred, average=None)
    recall_0 = "%.3f" % recall_0
    recall_1 = "%.3f" % recall_1
    recall_2 = "%.3f" % recall_2
    recall_3 = "%.3f" % recall_3


    print(f'f1 score is: {f1}' )
    print(f'roc_auc_score is: {auc_score}')
    print(f'accuracy score is: {"%.3f" % acc}')
    print(f'recall_score is: {recall}')
    print(f'confusion matrix is:\n {confusion_m}')
    print(f'f1_0 score is: {f1_0}' )
    print(f'f1_1 score is: {f1_1}' )
    print(f'f1_2 score is: {f1_2}' )
    print(f'f1_3 score is: {f1_3}' )
    print(f'recall_0 score is: {recall_0}')
    print(f'recall_1 score is: {recall_1}')
    print(f'recall_2 score is: {recall_2}')
    print(f'recall_3 score is: {recall_3}')

    return (history, f1, auc_score, acc, recall, f1_0, f1_1, f1_2, f1_3, recall_0, recall_1, recall_2, recall_3, confusion_m)

def log_results(results, log_dir):
    history, f1, auc_score, acc, recall, f1_0, f1_1, f1_2, f1_3, recall_0, recall_1, recall_2, recall_3, confusion_m = results
    for metric in ["sparse_categorical_accuracy", "loss"]:
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric])
        plt.title("model " + metric)
        plt.ylabel(metric, fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend(["train", "val"], loc="best")
        plt.savefig(log_dir.joinpath(metric + f'_figure.png'))
        plt.close()

    with open(log_dir.joinpath('metrics_log.txt'), 'w') as file:
        file.write('\n')
        file.write(model_name + stft_name + '\n')
        file.write('f1 score is: ' + str(f1) + '\n')
        file.write('roc_auc_score is: ' + str(auc_score) + '\n')
        file.write('accuracy score is: ' + str(acc) + '\n')
        file.write('recall_score is: ' + str(recall) + '\n')
        file.write('f1_0 score is: ' + str(f1_0) + '\n')
        file.write('f1_1 score is: ' + str(f1_1) + '\n')
        file.write('f1_2 score is: ' + str(f1_2) + '\n')
        file.write('f1_3 score is: ' + str(f1_3) + '\n')
        file.write('recall_0 score is: ' + str(recall_0) + '\n')
        file.write('recall_1 score is: ' + str(recall_1) + '\n')
        file.write('recall_2 score is: ' + str(recall_2) + '\n')
        file.write('recall_3 score is: ' + str(recall_3) + '\n')
        file.write('confusion_matrix:\n')
        for line in range(4):
            file.write(f'{confusion_m[line]}\n')

    
def objective(trial):
    log_dir = LOG_BASE_DIR.joinpath(model_name).joinpath(str(trial.number)) #joinpath('trial.number')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    results = train_model(x_train, x_val, stft_train, stft_val, y_train, y_val, log_dir, trial)
    log_results(results, log_dir)
    return results[3] # acc

if __name__ == '__main__':
    # stfts = [(128, 128), (128,16), (128, 64), (128, 8),(64, 50), (64, 32), (64, 8)]
    stfts = [(128, 128)]
    model_name = 'optuna_1'
    stft_name = f'stft_128_128'
    x_train, x_val, stft_train, stft_val, y_train, y_val = get_training_val_data(stft_name)
    # print(f'acc = {objective(None)}')
    # train_model(x_train, x_val, stft_train, stft_val, y_train, y_val, override_logs=True)

    

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.plot_param_importances(study)