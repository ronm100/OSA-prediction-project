import pyedflib
import numpy as np
import csv
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import pydot as pydot
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import sys
# from keras import backend as K

MODEL_NAME = 'data_test_ResNet5'
LOG_DIR = '../../../../databases/aviv.ish@staff.technion.ac.il/' + MODEL_NAME
CSV_DIR = '../../../../databases/aviv.ish@staff.technion.ac.il/processed_data_as_csv'
kernel_regularizer = keras.regularizers.L1(l1=0.1)
kernel_initializer = keras.initializers.GlorotNormal()

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

def conv_block(input, filters, Kernel_size, strides, pool_size, pool_stride):
    conv = keras.layers.Conv1D(filters=filters, kernel_size=Kernel_size, padding="same", strides=strides)(input)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.LeakyReLU()(conv)
    conv = keras.layers.MaxPooling1D(pool_size, strides=pool_stride, padding='same')(conv)
    conv = keras.layers.Dropout(0.2)(conv)
    return(conv)

def basic_conv_block (input, filters, stride, Relu):
    conv = keras.layers.Conv1D(filters=filters, kernel_size=5, padding="same", strides=stride)(input)
    conv = keras.layers.BatchNormalization()(conv)
    if (Relu == 0):
        return(conv)
    else:
        conv = keras.layers.LeakyReLU()(conv)
        return (conv)


def Res_conv_block(input, filters, strides, num_of_double_layers, shortcut):

    conv1 = basic_conv_block(input, filters, strides, 1)
    conv2 = basic_conv_block(conv1, filters, 1, 1)
    if (shortcut==1):
        input = keras.layers.Conv1D(filters=filters, kernel_size=2, strides=2, padding="same")(input)
    conv2 = keras.layers.LeakyReLU()(conv2 + input)

    conv3 = basic_conv_block(conv2, filters, 1, 1)
    conv4 = basic_conv_block(conv3, filters, 1, 0)
    conv4 = keras.layers.LeakyReLU()(conv4 + conv2)
    if (num_of_double_layers == 2):
        conv4 = keras.layers.Dropout(0.2)(conv4)
        return (conv4)

    conv5 = basic_conv_block(conv4, filters, 1, 1)
    conv6 = basic_conv_block(conv5, filters, 1, 0)
    conv6 = keras.layers.LeakyReLU()(conv6 + conv4)
    if (num_of_double_layers == 3):
        conv6 = keras.layers.Dropout(0.2)(conv6)
        return (conv6)

    conv7 = basic_conv_block(conv6, filters, 1, 1)
    conv8 = basic_conv_block(conv7, filters, 1, 0)
    conv8 = keras.layers.LeakyReLU()(conv8 + conv6)
    if (num_of_double_layers == 4):
        conv8 = keras.layers.Dropout(0.2)(conv8)
        return (conv8)

    conv9 = basic_conv_block(conv8, filters, 1, 1)
    conv10 = basic_conv_block(conv9, filters, 1, 0)
    conv10 = keras.layers.LeakyReLU()(conv10 + conv8)
    if (num_of_double_layers == 5):
        conv10 = keras.layers.Dropout(0.2)(conv10)
        return (conv10)


    conv11 = basic_conv_block(conv10, filters, 1, 1)
    conv12 = basic_conv_block(conv11, filters, 1, 0)
    conv12 = keras.layers.LeakyReLU()(conv12 + conv10)
    if (num_of_double_layers == 6):
        conv12 = keras.layers.Dropout(0.2)(conv12)
        return (conv12)
    else:
        print("Illegal number of double layers, please enter a number between 2 and 6")

def make_model(input_shape):
    num_classes = 4
    input_layer = keras.layers.Input(input_shape)

    conv0 = conv_block(input_layer, 128 , Kernel_size=3, strides=1, pool_size=2, pool_stride=2 )

    conv1 = Res_conv_block(conv0, 128, strides=1, num_of_double_layers=3, shortcut=0)

    conv2 = Res_conv_block(conv1, 128, strides=2, num_of_double_layers=2, shortcut=1)

    conv3 = Res_conv_block(conv2, 128, strides=2, num_of_double_layers=2, shortcut=1)

    conv3 = conv_block(conv3, 128, Kernel_size=3, strides=1, pool_size=2, pool_stride=2)
    conv3 = conv_block(conv3, 128, Kernel_size=3, strides=1, pool_size=2, pool_stride=2)
    conv3 = conv_block(conv3, 128, Kernel_size=3, strides=1, pool_size=2, pool_stride=2)
    conv3 = conv_block(conv3, 128, Kernel_size=3, strides=1, pool_size=2, pool_stride=2)
    conv3 = conv_block(conv3, 128, Kernel_size=3, strides=1, pool_size=2, pool_stride=2)
    #
    # conv4 = Res_conv_block(conv3, 512, strides=2, num_of_double_layers=2, shortcut=1)



    # conv1 = conv_block(conv0, 64, 3, "same", 0, 1)
    # conv2 = conv_block(conv1, 64 ,3 ,"same", 0, 1)
    # conv3 = conv_block(conv2 + conv0, 64 ,3 ,"same", 0, 1)
    #
    # conv4 = conv_block(conv3, 64 ,3 ,"same", 0, 1)
    # conv5 = conv_block(conv4, 64 ,3 ,"same", 0, 1)
    # conv6 = conv_block(conv5 + conv3, 64 ,3 ,"same", 0, 1)
    #
    # conv7 = conv_block(conv6, 128 ,3 ,"same", 0, 1)
    # conv8 = conv_block(conv7, 128 ,3 ,"same", 0, 2)
    # conv8 = conv_block(conv8 + conv6, 128, 3, "same", 0, 2)
    #
    # conv8 = conv_block(conv7, 128, 3, "same", 0.2, 2)
    # conv8 = conv_block(conv7, 128, 3, "same", 0.2, 2)
    # conv8 = conv_block(conv7, 128, 3, "same", 0.2, 2)



    # conv1 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(input_layer)
    # conv1 = keras.layers.BatchNormalization()(conv1)
    # conv1 = keras.layers.LeakyReLU()(conv1)
    # conv1 = keras.layers.MaxPooling1D(2, padding='same')(conv1)
    # conv1 = keras.layers.Dropout(0.2)(conv1)
    #
    # conv2 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv1)
    # conv2 = keras.layers.BatchNormalization()(conv2)
    # conv2 = keras.layers.LeakyReLU()(conv2)
    # conv2 = keras.layers.MaxPooling1D(2, padding='same')(conv2)
    # conv2 = keras.layers.Dropout(0.2)(conv2)
    #
    # conv3= keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(conv2)
    # conv3 = keras.layers.BatchNormalization()(conv3)
    # conv3 = keras.layers.LeakyReLU()(conv3)
    # conv3 = keras.layers.MaxPooling1D(2, padding='same')(conv3)
    # conv3 = keras.layers.Dropout(0.2)(conv3)
    #
    # conv4 = keras.layers.Conv1D(filters=128, kernel_size=5, padding="same")(conv3)
    # conv4 = keras.layers.BatchNormalization()(conv4)
    # conv4 = keras.layers.LeakyReLU()(conv4)
    # conv4 = keras.layers.MaxPooling1D(2, padding='same')(conv4)
    # conv4 = keras.layers.Dropout(0.2)(conv4)
    #
    # conv5 = keras.layers.Conv1D(filters=128, kernel_size=7, padding="same")(conv4)
    # conv5 = keras.layers.BatchNormalization()(conv5)
    # conv5 = keras.layers.LeakyReLU()(conv5)
    # conv5 = keras.layers.Dropout(0.2)(conv5)
    #
    # conv6 = keras.layers.Conv1D(filters=128, kernel_size=7, padding="same")(conv5+conv4)
    # conv6 = keras.layers.BatchNormalization()(conv6)
    # conv6 = keras.layers.LeakyReLU()(conv6)
    # conv6 = keras.layers.Dropout(0.2)(conv6)
    #
    # conv7 = keras.layers.Conv1D(filters=128, kernel_size=9, padding="same")(conv6+conv5)
    # conv7 = keras.layers.BatchNormalization()(conv7)
    # conv7 = keras.layers.LeakyReLU()(conv7)
    # conv7 = keras.layers.MaxPooling1D(2, padding='same')(conv7)
    # conv7 = keras.layers.Dropout(0.2)(conv7)
    #
    # conv8 = keras.layers.Conv1D(filters=128, kernel_size=9, padding="same")(conv7)
    # conv8 = keras.layers.BatchNormalization()(conv8)
    # conv8 = keras.layers.LeakyReLU()(conv8)
    # conv8 = keras.layers.MaxPooling1D(2, padding='same')(conv8)
    # conv8 = keras.layers.Dropout(0.2)(conv8)
    # #
    # flattened = keras.layers.Flatten()(conv8)
    #
    # gap = keras.layers.Dense(500, activation="relu")(flattened)
    conv3 = keras.layers.AveragePooling1D(2, strides=2, padding='same')(conv3)
    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    # output_layer = keras.layers.Dense(256, activation="softmax")(gap)
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
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.25, random_state = 21)

    model = make_model(input_shape=x_train.shape[1:])

    keras.utils.plot_model(model, to_file = LOG_DIR + '/' + MODEL_NAME+ "_architecture.png", show_shapes=True)

    epochs = 1000
    batch_size = 32
    callbacks = [keras.callbacks.ModelCheckpoint(LOG_DIR + '/' + MODEL_NAME + "_best_model.h5", save_best_only=True, monitor="val_loss"),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=0.00001),
                 keras.callbacks.EarlyStopping(monitor="val_loss", patience=18, verbose=1),]


    model.compile (optimizer="adam",loss="sparse_categorical_crossentropy",
                   metrics=['sparse_categorical_accuracy'])

    history = model.fit (x_train, y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=callbacks, validation_data=(x_val, y_val), verbose=1)

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
        file.write('f1_0 score is: ' + str(f1_0) + '\n')
        file.write('f1_1 score is: ' + str(f1_1) + '\n')
        file.write('f1_2 score is: ' + str(f1_2) + '\n')
        file.write('f1_3 score is: ' + str(f1_3) + '\n')
        file.write('recall_0 score is: ' + str(recall_0) + '\n')
        file.write('recall_1 score is: ' + str(recall_1) + '\n')
        file.write('recall_2 score is: ' + str(recall_2) + '\n')
        file.write('recall_3 score is: ' + str(recall_3) + '\n')


    pd.DataFrame(confusion_m).to_csv(LOG_DIR + '/' + MODEL_NAME + '_confusion_matrix.csv')

