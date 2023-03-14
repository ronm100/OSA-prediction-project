import pyedflib
import numpy as np
from pathlib import Path
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pydot import *
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from librosa import stft
import time



CSV_DIR = '../../../../databases/ronmaishlos@staff.technion.ac.il/processed_data_as_csv'
ROOT = Path('../../../../..')
STFT_DIR = ROOT.joinpath(Path('databases/ronmaishlos@staff.technion.ac.il/processed_data_as_csv/stft'))

def ahi_to_label(ahi):
    if ahi < 5:
        return 0
    elif ahi < 15:
        return 1
    elif ahi < 30:
        return 2
    else:
        return 3


def get_labels(path):
    data = pd.read_csv(path)
    ahi = data['ahi_a0h3a']
    return list(map(ahi_to_label, ahi))


def edf_get_oximetry(edf_path):
    edf = pyedflib.EdfReader(edf_path)
    i_position = np.where(np.array(edf.getSignalLabels()) == 'SaO2')[0][0]
    position = edf.readSignal(i_position)
    signal = np.array(position).astype(float)
    return signal


def conv2d_block(input_layer, filters, kernel_size, pooling='max', padding='valid'):
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding=padding)(
        input_layer)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.AveragePooling2D()(conv) if pooling == 'avg' else keras.layers.MaxPooling2D()(conv)
    conv = keras.layers.Dropout(0.2)(conv)

    return conv


def make_model(input_shape):
    num_classes = 4
    input_layer = keras.layers.Input(input_shape)

    conv_1 = conv2d_block(filters=4, kernel_size=3, pooling='avg', padding='same', input_layer=input_layer)
    conv_2 = conv2d_block(filters=8, kernel_size=3, padding='same', input_layer=conv_1)
    conv_3 = conv2d_block(filters=16, kernel_size=(4, 2), input_layer=conv_2)
    conv_4 = conv2d_block(filters=32, kernel_size=(4, 2), input_layer=conv_3)
    # conv_5 = conv2d_block(filters=64, kernel_size=(3, 2), input_layer=conv_4)
    # conv_6 = conv2d_block(filters=128, kernel_size=(3, 2), input_layer=conv_5)
    # conv_7 = conv2d_block(filters=256, kernel_size=(3, 2), input_layer=conv_6)

    gap = keras.layers.GlobalAveragePooling2D()(conv_3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def get_training_data(data_dir: Path):
    x_train = np.load(data_dir.joinpath('x_t'), allow_pickle=True)
    x_val = np.load(data_dir.joinpath('x_v'), allow_pickle=True)
    y_train = np.load(data_dir.joinpath('y_t'), allow_pickle=True)
    y_val = np.load(data_dir.joinpath('y_v'), allow_pickle=True)
    
    return x_train, x_val, y_train, y_val


def train_model(x_train, x_val, y_train, y_val):
    log_dir = ROOT.joinpath('databases/ronmaishlos@staff.technion.ac.il/logs/stft_on_2d_conv/' + model_name)
    if os.path.exists(log_dir):
        raise ValueError('Trying to overrun existing model results')
    os.makedirs(log_dir)

    model = make_model(input_shape=x_train.shape[1:])
    keras.utils.plot_model(model, to_file = log_dir.joinpath("architecture.png"), show_shapes=True)
    epochs = 1000
    batch_size = 32
    callbacks = [keras.callbacks.ModelCheckpoint(log_dir.joinpath("best_model.h5"), save_best_only=True, monitor="val_loss"),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=0.00001),
                 keras.callbacks.EarlyStopping(monitor="val_loss", patience=18, verbose=1),]

    model.compile (optimizer="adam",loss="sparse_categorical_crossentropy",
                   metrics=['sparse_categorical_accuracy'])

    history = model.fit (x_train, y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks, validation_data=(x_val, y_val), verbose=1,)

    # loss, accuracy, f1_score, precision, recall = model.evaluate(x_train, y_train, verbose=0)
    y_pred_probs = model.predict(x_val)
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




    with open(log_dir.joinpath('matrics_log.txt'), 'w+') as file:
        file.write(model_name + '\n')
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


    pd.DataFrame(confusion_m).to_csv(log_dir.joinpath('confusion_matrix.csv'))

if __name__ == '__main__':
    stfts = [(128, 128), (128,16), (128, 64), (128, 8),(64, 50), (64, 32), (64, 8)]
    for n_fft, win_len in stfts:
        model_name = f'stft_{n_fft}_{win_len}'
        data_dir = STFT_DIR.joinpath(model_name)
        x_train, x_val, y_train, y_val = get_training_data(data_dir)
        train_model(x_train, x_val, y_train, y_val)