import pyedflib
import numpy as np
import csv
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from pydot import *
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import sys
# from keras import backend as K

MODEL_NAME = 'data_test_1'
LOG_DIR = '../../../../databases/aviv.ish@staff.technion.ac.il/' + MODEL_NAME
CSV_DIR = '../../../../databases/aviv.ish@staff.technion.ac.il/data_as_csv'
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

    semple_length = 21600
    num_of_semples = 5755
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    # else:
    #     print("files already exists, please delete the" + MODEL_NAME + "directory and try again")
    #     sys.exit()
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x = pd.read_csv(CSV_DIR + '/' + 'x_train.csv', nrows = num_of_semples)
    x = np.array(x)
    x = x[:,0:21600]
    x = x.reshape((x.shape[0], x.shape[1], 1))
    y = pd.read_csv(CSV_DIR + '/' + 'y_train.csv', nrows = num_of_semples)
    y = np.array(y)[0:num_of_semples,1]
    y = y.reshape(num_of_semples, -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
    model = make_model(input_shape=x_train.shape[1:])
    keras.utils.plot_model(model, to_file = LOG_DIR + '/' + MODEL_NAME+ "_architecture.png", show_shapes=True)
    epochs = 100
    batch_size = 32
    callbacks = [keras.callbacks.ModelCheckpoint(LOG_DIR + '/' + MODEL_NAME + "_best_model.h5", save_best_only=True, monitor="val_loss"),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=0.0001),
                 keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, verbose=1),]

    model.compile (optimizer="adam",loss="sparse_categorical_crossentropy",
                   metrics=['sparse_categorical_accuracy'])

    history = model.fit (x_train, y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks, validation_split=0.2, verbose=1,)

    # loss, accuracy, f1_score, precision, recall = model.evaluate(x_train, y_train, verbose=0)
    y_pred_probs = model.predict(x_train)
    y_pred = np.argmax(y_pred_probs, axis=1)
    f1 = "%.3f" % f1_score(y_train, y_pred, average='macro')
    confusion_m = confusion_matrix(y_train, y_pred)
    auc_score = "%.3f" % roc_auc_score(y_train, y_pred_probs, average='macro', multi_class='ovr')
    acc ="%.3f" % accuracy_score(y_train, y_pred)
    recall ="%.3f" % recall_score(y_train, y_pred, average='macro')
    print(f'f1 score is: {f1}' )
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
    plt.savefig(LOG_DIR + '/' + metric + '_figure.png')
    plt.close()

    metric = "loss"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(LOG_DIR + '/' + metric + '_figure.png')
    plt.close()



    with open(LOG_DIR + '/' + MODEL_NAME + '_matrics_log.txt', 'w+') as file:
        file.write(MODEL_NAME + '\n')
        file.write('f1 score is: ' + str(f1) + '\n')
        file.write('roc_auc_score is: ' + str(auc_score) + '\n')
        file.write('accuracy score is: ' + str(acc) + '\n')
        file.write('recall_score is: ' + str(recall) + '\n')

    pd.DataFrame(confusion_m).to_csv(LOG_DIR + '/' + MODEL_NAME + '_confusion_matrix.csv')
