import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pydot import *
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split

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

def make_model_DuPLO_STFT(orig_input_shape, stft_input_shape):
    num_classes = 4
    orig_input_layer = keras.layers.Input(orig_input_shape)
    stft_input_layer = keras.layers.Input(stft_input_shape)

    # STFT - CNN:
    conv2d_1 = conv2d_block(filters=4, kernel_size=(3,9), dropout=0.1, pooling='avg', input_layer=stft_input_layer)
    conv2d_2 = conv2d_block(filters=8, kernel_size=(3,9), input_layer=conv2d_1)
    conv2d_3 = conv2d_block(filters=16, kernel_size=(3,5), input_layer=conv2d_2)
    conv2d_4 = conv2d_block(filters=32, pool_size=2, input_layer=conv2d_3)
    conv2d_5 = conv2d_block(filters=64, pool_size=2, input_layer=conv2d_4)
    conv2d_6 = conv2d_block(filters=128, pool_size=2, input_layer=conv2d_5)
    conv2d_7 = conv2d_block(filters=256, pool_size=2, input_layer=conv2d_6)
    conv2d_8 = conv2d_block(filters=256, pool_size=2, input_layer=conv2d_7)
    stft_flat = keras.layers.Flatten()(conv2d_8)
    stft_flat = keras.layers.Dense(128, activation="softmax")(stft_flat)

    # Duplo - RNN:
    conv_lstm1 = conv1d_block(orig_input_layer, filters=32, kernel_size=9, dropout=0.1, pooling='avg')
    conv_lstm2 = conv1d_block(conv_lstm1, filters=64, kernel_size=3)
    lstm_1 = keras.layers.LSTM(64)(conv_lstm2)
    lstm_1 = keras.layers.Reshape((64, -1))(lstm_1)
    conv_lstm3 = conv1d_block(lstm_1, filters=128, kernel_size=3, pooling='avg')
    conv_lstm4 = conv1d_block(conv_lstm3, filters=128, kernel_size=3)
    lstm_2 = keras.layers.LSTM(128)(conv_lstm4)

    # Duplo CNN:
    conv1 = conv1d_block(orig_input_layer, filters=16, kernel_size=9, dropout=0.1, pooling='avg', pool_size=4)
    conv2 = conv1d_block(conv1, filters=16, kernel_size=3, pool_size=4)
    conv3 = conv1d_block(conv2, filters=32, kernel_size=3, pool_size=4)
    conv4 = conv1d_block(conv3, filters=32, kernel_size=3)
    conv5 = conv1d_block(conv4, filters=64, kernel_size=3, pool_size=4)
    conv6 = conv1d_block(conv5, filters=64, kernel_size=3)
    conv7 = conv1d_block(conv6, filters=128, kernel_size=3)
    conv8 = conv1d_block(conv7, filters=256, kernel_size=3)
    cnn_flat = keras.layers.Flatten()(conv8)
    cnn_flat = keras.layers.Dense(128, activation="softmax")(cnn_flat)

    concatted = keras.layers.Concatenate()([cnn_flat, stft_flat, lstm_2])
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(concatted)

    return keras.models.Model(inputs=[orig_input_layer, stft_input_layer], outputs=output_layer)

def make_model_DuPLO(orig_input_shape):
    num_classes = 4
    orig_input_layer = keras.layers.Input(orig_input_shape)

    # Duplo - RNN:
    conv_lstm1 = conv1d_block(orig_input_layer, filters=32, kernel_size=9, dropout=0.1, pooling='avg')
    conv_lstm2 = conv1d_block(conv_lstm1, filters=64, kernel_size=3)
    lstm_1 = keras.layers.LSTM(64)(conv_lstm2)
    lstm_1 = keras.layers.Reshape((64, -1))(lstm_1)
    conv_lstm3 = conv1d_block(lstm_1, filters=128, kernel_size=3, pooling='avg')
    conv_lstm4 = conv1d_block(conv_lstm3, filters=128, kernel_size=3)
    lstm_2 = keras.layers.LSTM(128)(conv_lstm4)

    # Duplo CNN:
    conv1 = conv1d_block(orig_input_layer, filters=16, kernel_size=9, dropout=0.1, pooling='avg', pool_size=4)
    conv2 = conv1d_block(conv1, filters=16, kernel_size=3, pool_size=4)
    conv3 = conv1d_block(conv2, filters=32, kernel_size=3, pool_size=4)
    conv4 = conv1d_block(conv3, filters=32, kernel_size=3)
    conv5 = conv1d_block(conv4, filters=64, kernel_size=3, pool_size=4)
    conv6 = conv1d_block(conv5, filters=64, kernel_size=3)
    conv7 = conv1d_block(conv6, filters=128, kernel_size=3)
    conv8 = conv1d_block(conv7, filters=256, kernel_size=3)
    cnn_flat = keras.layers.Flatten()(conv8)
    cnn_flat = keras.layers.Dense(128, activation="softmax")(cnn_flat)

    concatted = keras.layers.Concatenate()([cnn_flat, lstm_2])
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(concatted)

    return keras.models.Model(inputs=orig_input_layer, outputs=output_layer)

def make_model_CNN(orig_input_shape):
    num_classes = 4
    orig_input_layer = keras.layers.Input(orig_input_shape)

    # Duplo CNN:
    conv1 = conv1d_block(orig_input_layer, filters=16, kernel_size=9, dropout=0.1, pooling='avg', pool_size=4)
    conv2 = conv1d_block(conv1, filters=16, kernel_size=3, pool_size=4)
    conv3 = conv1d_block(conv2, filters=32, kernel_size=3, pool_size=4)
    conv4 = conv1d_block(conv3, filters=32, kernel_size=3)
    conv5 = conv1d_block(conv4, filters=64, kernel_size=3, pool_size=4)
    conv6 = conv1d_block(conv5, filters=64, kernel_size=3)
    conv7 = conv1d_block(conv6, filters=128, kernel_size=3)
    conv8 = conv1d_block(conv7, filters=256, kernel_size=3)
    cnn_flat = keras.layers.Flatten()(conv8)
    cnn_flat = keras.layers.Dense(128, activation="softmax")(cnn_flat)
    cnn_flat = keras.layers.GlobalAveragePooling1D()(cnn_flat)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(cnn_flat)

    return keras.models.Model(inputs=orig_input_layer, outputs=output_layer)

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


def train_model(x_train, x_val, stft_train, stft_val, y_train, y_val, override_logs = False):
    log_dir = LOG_BASE_DIR.joinpath(model_name).joinpath(stft_name)
    if os.path.exists(log_dir):
        if not override_logs:
            raise ValueError('Trying to overrun existing model results')
        else:
            print(f'Overriding {log_dir}')
    else:
        os.makedirs(log_dir)

    training_inputs = [x_train, stft_train]
    validation_inputs = [x_val, stft_val]
    model = make_model_DuPLO_STFT(orig_input_shape=x_train.shape[1:], stft_input_shape=stft_train.shape[1:])
    keras.utils.plot_model(model, to_file = log_dir.joinpath("architecture.png"), show_shapes=True)
    epochs = 1000
    batch_size = 32
    callbacks = [keras.callbacks.ModelCheckpoint(log_dir.joinpath("best_model.h5"), save_best_only=True, monitor="val_sparse_categorical_accuracy"),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=0.000001),
                 keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=20, verbose=1, start_from_epoch=100, restore_best_weights=True, mode='max'),]

    model.compile (optimizer="adam",loss="sparse_categorical_crossentropy",
                   metrics=['sparse_categorical_accuracy'])

    history = model.fit (training_inputs, y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks, validation_data=(validation_inputs, y_val), verbose=1,)

    model = keras.models.load_model(log_dir.joinpath("best_model.h5"))
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
    acc ="%.3f" % accuracy_score(y_val, y_pred)
    recall ="%.3f" % recall_score(y_val, y_pred, average='macro')
    (recall_0, recall_1, recall_2, recall_3) = recall_score(y_val, y_pred, average=None)
    recll_0 = "%.3f" % recall_0
    recll_1 = "%.3f" % recall_1
    recll_2 = "%.3f" % recall_2
    recll_3 = "%.3f" % recall_3


    print(f'f1 score is: {f1}' )
    print(f'roc_auc_score is: {auc_score}')
    print(f'accuracy score is: {acc}')
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


    for metric in ["sparse_categorical_accuracy", "loss"]:
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric])
        plt.title("model " + metric)
        plt.ylabel(metric, fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend(["train", "val"], loc="best")
        plt.savefig(log_dir.joinpath(metric + '_figure.png'))
        plt.close()




    with open(log_dir.joinpath('metrics_log.txt'), 'w+') as file:
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
    
if __name__ == '__main__':
    stfts = [(128, 128)]
    model_name = '3_branches_2nd_try'
    for n_fft, win_len in stfts:
        stft_name = f'stft_{n_fft}_{win_len}'
        x_train, x_val, stft_train, stft_val, y_train, y_val = get_training_val_data(stft_name)
        train_model(x_train, x_val, stft_train, stft_val, y_train, y_val, override_logs=True)