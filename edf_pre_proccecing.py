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
        return None
    i_position = np.where(np.array(edf.getSignalLabels()) == 'SaO2')[0][0]
    position = edf.readSignal(i_position)
    signal = np.array(position).astype(float)
    return signal


if __name__ == '__main__':
    x_train = []
    semple_length = 21600
    num_of_semples = 5804
    y_train = get_labels('./shhs1-dataset-0.14.0.csv')[0:num_of_semples]
    y_train = np.array(y_train)
    y_train = y_train.reshape(num_of_semples, -1)

    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
    else:
        print("files already exists, please delete the 'data_as_csv' directory and try again")
        sys.exit()
    all_paths = os.listdir(SIGNAL_DIR)
    all_paths.sort()
    paths = all_paths[:]
    x_indexes = []
    skipped_indexes = []
    last_index = 0
    for path in paths:
        temp = edf_get_oximetry(SIGNAL_DIR + '/' + path)
        if temp is None:
            continue
        edf_index = int(path[7:12])
        if np.shape(temp)[0] >= semple_length:
            temp = temp[0:semple_length]
            x_train.append(temp)
            x_indexes.append(edf_index - 1)
            if (edf_index - 1) != last_index:
                for i in range(last_index, edf_index-1):
                    skipped_indexes.append(i)
                    print(f'skipped unvalid semple: ' + str(i))
            if ((edf_index-1) % 10) == 0:
                print('semple number = ' + str(edf_index - 1))
            last_index = edf_index
    pd.DataFrame(y_train).to_csv(CSV_DIR + '/' + 'y_train_full.csv')
    y_train = y_train[x_indexes]
    x_train = np.stack(x_train, axis=0)
    pd.DataFrame(x_train).to_csv(CSV_DIR + '/' + 'x_train.csv')
    pd.DataFrame(y_train).to_csv(CSV_DIR + '/' + 'y_train.csv')
    pd.DataFrame(x_indexes).to_csv(CSV_DIR + '/' + 'valid_semples_inedex.csv')
    pd.DataFrame(skipped_indexes).to_csv(CSV_DIR + '/' + 'skipped_semples.csv')
    x_indexes = [x_indexes,y_train]
    pd.DataFrame(x_indexes).to_csv(CSV_DIR + '/' + 'valid_semples_inedex_with_lables.csv')
