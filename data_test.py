import pyedflib
import numpy as np
import csv
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pydot import *
import tensorflow_addons as tfa

# from keras import backend as K


def sensitivity(y_true, y_pred): # TP/(TP+FN)
    true_positives = keras.backend.sum(keras.backend.cast(y_true * y_pred, 'float'), axis = 0)
    possible_positives = keras.backend.sum(keras.backend.cast(y_true, 'float'), axis = 0)
    sensitivity = true_positives / (possible_positives + keras.backend.epsilon())
    return sensitivity


def precision(y_true, y_pred):  # TP/(TP+FP)
    true_positives = keras.backend.sum(keras.backend.cast(y_true * y_pred,'float'), axis = 0)
    predicted_positives = keras.backend.sum(keras.backend.cast(y_pred,'float'), axis = 0)
    precision = true_positives / (predicted_positives + keras.backend.epsilon())
    return precision


def f1(y_true, y_pred): # 2*(precision * sensitivity) / (precision + sensitivity) = TP/(TP+(FP+FN)/2)
    pre = precision(y_true, y_pred)
    se = sensitivity(y_true, y_pred)
    return 2 * ((pre * se) / (pre + se + keras.backend.epsilon()))

def specificity(y_true, y_pred):  # TN/(TN+FP)
    true_negatives = keras.backend.sum(keras.backend.cast((1-y_true) * (1-y_pred),'float'), axis = 0)
    possible_negatives = keras.backend.sum(keras.backend.cast((1-y_true),'float'), axis = 0)
    specificity = true_negatives / (possible_negatives + keras.backend.epsilon())
    return specificity


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
    for i in range(1, num_of_semples + 1):
        path = './signals/shhs1-2' + str(i).zfill(5) + '.edf'
        temp = edf_get_oximetry(path)[0:semple_length]
        if np.shape(temp) == (semple_length,):
            x_train.append(temp)
        else:
            y_train = np.delete(y_train, i - 1, 0)
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
        metrics=[tfa.metrics.F1Score(average='macro',num_classes=4), sensitivity, specificity, 'accuracy'],
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
    ret = model.evaluate(x_train, y_train, verbose=0)
    print(f'ret = {ret} \n names = {model.metrics_names}')
    print(history.history.keys())

    metric = "tfa.metrics.F1Score"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

    metric = "sensitivity"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

    metric = "specificity"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

    metric = 'acc'
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

