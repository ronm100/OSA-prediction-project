import pyedflib
import numpy as np
import os
import pandas as pd
import sys

SIGNAL_DIR = '../../../../databases/aviv.ish@staff.technion.ac.il/edf'
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
    try:
        edf = pyedflib.EdfReader(edf_path)
    except:
        return np.array([0,0])
    i_position = np.where(np.array(edf.getSignalLabels()) == 'SaO2')[0][0]
    position = edf.readSignal(i_position)
    signal = np.array(position).astype(float)
    return signal

if __name__ == '__main__':
    x_train = []
    y_train = []
    semple_length = 21600
    num_of_semples = 5790
    y_train_full = get_labels('./shhs1-dataset-0.14.0.csv')[0:num_of_semples]
    y_train_full = np.array(y_train_full)
    y_train_full = y_train_full.reshape(num_of_semples, -1)

    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
    else:
        print("files already exists, please delete the 'data_as_csv' directory and try again")
        sys.exit()
    all_paths = os.listdir(SIGNAL_DIR)
    all_paths.sort()
    paths = all_paths[0:num_of_semples]
    x_indexes = []
    for path in paths:
        temp = edf_get_oximetry(SIGNAL_DIR + '/' + path)[0:semple_length]
        if temp.all == 0:
            continue
        edf_index = int(path[7:12])
        if np.shape(temp) == (semple_length,):
            x_train.append(temp)
            x_indexes.append(edf_index - 1)
            y_train.append(y_train_full[edf_index - 1])
            if (((edf_index-1) % 10) == 0):
                print('semple number = ' + str(edf_index - 1))
    y_train = np.stack(y_train, axis=0)
    y_train.reshape(num_of_semples, -1)
    x_train = np.stack(x_train, axis=0)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
    pd.DataFrame(x_train).to_csv(CSV_DIR + '/' + 'x_train.csv')
    pd.DataFrame(y_train).to_csv(CSV_DIR + '/' + 'y_train.csv')
    pd.DataFrame(x_indexes).to_csv(CSV_DIR + '/' + 'valid_semples_inexes.csv')
