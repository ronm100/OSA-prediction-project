from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path('../../../../..')
LOG_BASE_DIR = ROOT.joinpath('databases/ronmaishlos@staff.technion.ac.il/logs')
MODEL_DIR = LOG_BASE_DIR.joinpath('optuna_multiparam/31')
MODEL_PATH = MODEL_DIR.joinpath('best_model.h5')
CSV_DIR = ROOT.joinpath('databases/ronmaishlos@staff.technion.ac.il/processed_data_as_csv')
STFT_DIR = CSV_DIR.joinpath('stft')
num_of_samples = 5755


def get_test_data(stft_name: str):
    sample_length = 21600
    num_of_samples = 5755
    stft_input_dir = STFT_DIR.joinpath(stft_name)

    x = np.array(pd.read_csv(CSV_DIR.joinpath('x_train.csv'), nrows=num_of_samples))[:, 0:sample_length]
    y = np.array(pd.read_csv(CSV_DIR.joinpath('y_train.csv'), nrows=num_of_samples))[0:num_of_samples, 1]
    x = x.reshape((x.shape[0], x.shape[1], 1))
    y = y.reshape(num_of_samples, -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    stft_test = np.load(stft_input_dir.joinpath('x_test'), allow_pickle=True)

    return x_test, stft_test[0], y_test

def log_results(results):
    f1, auc_score, acc, recall, f1_0, f1_1, f1_2, f1_3, recall_0, recall_1, recall_2, recall_3, confusion_m = results

    with open(MODEL_DIR.joinpath('metrics_log_test_set.txt'), 'w') as file:
        file.write('\n')
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
        file.write('confusion_matrix:\n')
        for line in range(4):
            file.write(f'{confusion_m[line]}\n')

if __name__ == '__main__':
    # Load required model
    model = keras.models.load_model(MODEL_PATH)

    stft_name = f'stft_128_128'
    x_test, stft_test, y_test = get_test_data(stft_name)

    y_pred_probs = model.predict([x_test, stft_test])
    y_pred = np.argmax(y_pred_probs, axis=1)
    f1 = "%.3f" % f1_score(y_test, y_pred, average='macro')
    (f1_0, f1_1, f1_2, f1_3) = f1_score(y_test, y_pred, average=None)
    f1_0 = "%.3f" % f1_0
    f1_1 = "%.3f" % f1_1
    f1_2 = "%.3f" % f1_2
    f1_3 = "%.3f" % f1_3
    confusion_m = confusion_matrix(y_test, y_pred)
    auc_score = "%.3f" % roc_auc_score(y_test, y_pred_probs, average='macro', multi_class='ovr')
    acc = "%.3f" % accuracy_score(y_test, y_pred)
    recall = "%.3f" % recall_score(y_test, y_pred, average='macro')
    (recall_0, recall_1, recall_2, recall_3) = recall_score(y_test, y_pred, average=None)
    recll_0 = "%.3f" % recall_0
    recll_1 = "%.3f" % recall_1
    recll_2 = "%.3f" % recall_2
    recll_3 = "%.3f" % recall_3

    print(f'f1 score is: {f1}')
    print(f'roc_auc_score is: {auc_score}')
    print(f'accuracy score is: {acc}')
    print(f'recall_score is: {recall}')
    print(f'confusion matrix is:\n {confusion_m}')
    print(f'f1_0 score is: {f1_0}')
    print(f'f1_1 score is: {f1_1}')
    print(f'f1_2 score is: {f1_2}')
    print(f'f1_3 score is: {f1_3}')
    print(f'recall_0 score is: {recall_0}')
    print(f'recall_1 score is: {recall_1}')
    print(f'recall_2 score is: {recall_2}')
    print(f'recall_3 score is: {recall_3}')

    results = f1, auc_score, acc, recall, f1_0, f1_1, f1_2, f1_3, recall_0, recall_1, recall_2, recall_3, confusion_m
    log_results(results)
