from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
import numpy as np

# MODEL_PATH = '../../../../databases/ronmaishlos@staff.technion.ac.il/logs/2_LSTMs_Size_128_each/2_LSTMs_Size_128_each_best_model.h5'
MODEL_PATH = '../../../../databases/aviv.ish@staff.technion.ac.il/data_test_8/data_test_8_best_model.h5'
CSV_DIR = '../../../../databases/ronmaishlos@staff.technion.ac.il/processed_data_as_csv'
num_of_samples = 5755

if __name__ == '__main__':
    # Load required model
    model = keras.models.load_model(MODEL_PATH)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x = pd.read_csv(CSV_DIR + '/' + 'x_train.csv', nrows=num_of_samples)
    x = np.array(x)
    x = x[:, 0:21600]
    x = x.reshape((x.shape[0], x.shape[1], 1))
    y = pd.read_csv(CSV_DIR + '/' + 'y_train.csv', nrows=num_of_samples)
    y = np.array(y)[0:num_of_samples, 1]
    y = y.reshape(num_of_samples, -1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    y_pred_probs = model.predict(x_test)
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
