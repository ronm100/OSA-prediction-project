import pyedflib
import numpy as np
import csv
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from pandas import *
from pydot import *

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
    data = read_csv(path)
    ahi = data['ahi_a0h3a']
    return list(map(ahi_to_label, ahi))

def edf_get_oximetry(edf_path):
    edf = pyedflib.EdfReader(edf_path)
    i_position = np.where(np.array(edf.getSignalLabels()) == 'SaO2')[0][0]
    position = edf.readSignal(i_position)
    signal = np.array(position).astype(float)
    return signal

def make_model(input_shape):
    num_classes  = 4
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv1 = keras.layers.MaxPooling1D(6,padding='same')(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2 = keras.layers.MaxPooling1D(4,padding='same')(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    conv3 = keras.layers.MaxPooling1D(3,padding='same')(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)



def main():
    x_train = []
    for i in range(1, 21):
        path = './signals/shhs1-2000' + str(i).zfill(2) + '.edf'
        x_train.append(edf_get_oximetry(path)[0:21600])
    #x_train = edf_get_oximetry('./signals/shhs1-200001.edf')
    x_train = np.stack(x_train, axis=0)
    x_train = x_train.reshape(-1,20,21600)
    print(x_train.shape)
    y_train = get_labels('./shhs1-dataset-0.14.0.csv')[0:20]
    y_train = np.array(y_train)
    y_train = y_train.reshape(-1,20)
    print(y_train.shape)
    model = make_model(input_shape=(1,21600))
    keras.utils.plot_model(model, show_shapes=True)
main()