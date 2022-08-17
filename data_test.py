import pyedflib
import numpy as np
import csv
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pydot import *
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
# from keras import backend as K

SIGNAL_DIR = './signals'
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


def make_model(input_shape):
    num_classes = 4
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv1 = keras.layers.MaxPooling1D(2, padding='same')(conv1)

    conv2 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2 = keras.layers.MaxPooling1D(6, padding='same')(conv2)

    conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    conv3 = keras.layers.MaxPooling1D(6, padding='same')(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


if __name__ == '__main__':
    x_train = []
    semple_length = 21600
    num_of_semples = 20
    y_train = get_labels('./shhs1-dataset-0.14.0.csv')[0:num_of_semples]
    y_train = np.array(y_train)
    y_train = y_train.reshape(num_of_semples, -1)

    all_paths = os.listdir(SIGNAL_DIR)
    all_paths.sort()
    paths = all_paths[0:num_of_semples]
    for path in paths:
        temp = edf_get_oximetry(SIGNAL_DIR + '/' + path)[0:semple_length]
        edf_index = int(path[7:11])
        if np.shape(temp) == (semple_length,):
            x_train.append(temp)
        else:
            y_train = np.delete(y_train, edf_index - 1, 0)
    x_train = np.stack(x_train, axis=0)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    model = make_model(input_shape=x_train.shape[1:])
    keras.utils.plot_model(model, show_shapes=True)

    epochs = 4
    batch_size = 5

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=['sparse_categorical_accuracy']
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    # loss, accuracy, f1_score, precision, recall = model.evaluate(x_train, y_train, verbose=0)
    y_pred_probs = model.predict(x_train)
    y_pred = np.argmax(y_pred_probs, axis=1)
    f1 = f1_score(y_train, y_pred, average='macro')
    confusion_m = confusion_matrix(y_train, y_pred)
    auc_score = roc_auc_score(y_train, y_pred_probs, average='macro', multi_class='ovr')
    acc = accuracy_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred, average='macro')
    print(f'f1 score is: {f1}')
    print(f'roc_auc_score is: {auc_score}')
    print(f'accuracy score is: {acc}')
    print(f'recall_score is: {recall}')
    print(f'confusion matrix is:\n {confusion_m}')


    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()
