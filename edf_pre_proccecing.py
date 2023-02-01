import pyedflib
import numpy as np
import os
import pandas as pd
import sys
from librosa import stft
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path('../../../../..')
SIGNAL_DIR = '../../../../databases/aviv.ish@staff.technion.ac.il/edf'
CSV_DIR = ROOT.joinpath(Path('databases/ronmaishlos@staff.technion.ac.il/processed_data_as_csv'))
STFT_DIR = CSV_DIR.joinpath(Path('stft'))

num_of_samples = 5755


def apply_stft(x_train, x_val, n_fft, win_length):
    train_stft, val_stft = list(), list()
    i = 0
    # t_1, t_2 = time.time(), time.time()
    for train_sample in x_train:
        train_sample = train_sample.reshape(train_sample.shape[0], )
        train_stft.append(abs(stft(train_sample, n_fft=n_fft, win_length=win_length)))
        # i += 1
        # t_2 = t_1
        # t_1 = time.time()
        # if i % 200 != 0:
        #     print(f'time_delta: {t_1 - t_2}, i / 200 = {i / 200}')

    for val_sample in x_val:
        val_sample = val_sample.reshape(val_sample.shape[0], )
        val_stft.append(abs(stft(val_sample, n_fft=n_fft, win_length=win_length)))

    return np.expand_dims(np.array(train_stft), axis=3), np.expand_dims(np.array(val_stft), axis=3)


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


def edf_to_csv():
    x_train = []
    semple_length = 21600
    num_of_semples = 5800
    y_train = get_labels('./shhs1-dataset-0.14.0.csv')
    y_train = np.array(y_train)
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
                for i in range(last_index, edf_index - 1):
                    skipped_indexes.append(i)
                    print(f'skipped unvalid semple: ' + str(i))
            if ((edf_index - 1) % 10) == 0:
                print('semple number = ' + str(edf_index - 1))
            last_index = edf_index
    pd.DataFrame(y_train).to_csv(CSV_DIR + '/' + 'y_train_full.csv')
    y_train = y_train[x_indexes]
    x_train = np.stack(x_train, axis=0)
    pd.DataFrame(x_train).to_csv(CSV_DIR + '/' + 'x_train.csv')
    pd.DataFrame(y_train).to_csv(CSV_DIR + '/' + 'y_train.csv')
    pd.DataFrame(x_indexes).to_csv(CSV_DIR + '/' + 'valid_semples_inedex.csv')
    pd.DataFrame(skipped_indexes).to_csv(CSV_DIR + '/' + 'skipped_semples.csv')
    x_indexes = [x_indexes, y_train]
    pd.DataFrame(x_indexes).to_csv(CSV_DIR + '/' + 'valid_semples_inedex_with_lables.csv')


def compute_and_save_dft(dir_path):
    x_train = pd.read_csv(dir_path + 'x_train.csv')
    x_train_fft = np.fft.rfft(x_train, axis=-1)
    x_train_fft_norm = np.abs(x_train_fft) ** 2
    x_train_fft_tensor = np.array((x_train_fft.real, x_train_fft.imag))  # should be tensor of shape (x,x,2)
    pd.DataFrame(x_train_fft_norm).to_csv(dir_path + 'x_train_fft_norm.csv')
    pd.DataFrame(x_train_fft.real).to_csv(dir_path + 'x_train_fft_real.csv')
    pd.DataFrame(x_train_fft.imag).to_csv(dir_path + 'x_train_fft_imag.csv')


def compute_and_save_stft(dir_path, n_fft, window_length):
    x = pd.read_csv(dir_path.joinpath('x_train.csv'))
    x = np.array(x)
    x = x[:, 0:21600]
    x = x.reshape((x.shape[0], x.shape[1], 1))
    y = pd.read_csv(dir_path.joinpath('y_train.csv'), nrows=num_of_samples)
    y = np.array(y)[0:num_of_samples, 1]
    y = y.reshape(num_of_samples, -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=21)
    x_train_stft, x_val_stft = apply_stft(x_train, x_val, n_fft, window_length)

    # Save data:
    dir_path = dir_path.joinpath(Path('stft'))
    new_dir = dir_path.joinpath(Path(f'stft_{n_fft}_{window_length}'))
    x_t_path = new_dir.joinpath(Path('x_t'))
    x_v_path = new_dir.joinpath(Path('x_v'))
    y_t_path = new_dir.joinpath(Path('y_t'))
    y_v_path = new_dir.joinpath(Path('y_v'))
    if not Path.exists(new_dir):
        Path.mkdir(new_dir)

    print(f'stft shape = {x_train_stft.shape}')
    with open(x_t_path, 'wb') as x_t_file:
        np.save(x_t_file, x_train_stft)
    with open(x_v_path, 'wb') as x_v_file:
        np.save(x_v_file, x_val_stft)
    with open(y_t_path, 'wb') as y_t_file:
        np.save(y_t_file, y_train)
    with open(y_v_path, 'wb') as y_v_file:
        np.save(y_v_file, y_test)

if __name__ == '__main__':
    # stft_pairs = [(128, 16), (128, 8), (256, 16), (256, 8), (512, 8)]
    stft_pairs = [(128, 128), (128, 64), (64, 50), (64, 32), (64, 8),]
    for n_fft, window_length in stft_pairs:
        compute_and_save_stft(CSV_DIR, n_fft, window_length)
        print(f'{n_fft}, {window_length} finished')
