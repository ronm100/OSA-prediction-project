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
from scipy.signal import savgol_filter


CSV_DIR = '../../../../databases/aviv.ish@staff.technion.ac.il/data_as_csv_final'
OUTPUT_CSV_DIR = '../../../../databases/aviv.ish@staff.technion.ac.il/processed_data_as_csv'


if __name__ == '__main__':

    semple_length = 21600


    x = pd.read_csv(CSV_DIR + '/' + 'x_train.csv')
    x = np.array(x)
    x = x[:,0:21600]
    x = np.around(x, decimals=5)

    for i in range (0, np.shape(x)[0]):
        x[i,:] = np.where(x[i,:]>75, x[i,:], np.mean(x[i,:]))
        x[i,:] = savgol_filter(x[i,:], 3, 2)
        if (i % 10) == 0:
            print(i)

    x = np.around(x, decimals=4)
    print(x.shape)
    num_of_semples = x.shape[0]

    y = pd.read_csv(CSV_DIR + '/' + 'y_train.csv')
    y = np.array(y)[0:num_of_semples,1]

    if not os.path.exists(OUTPUT_CSV_DIR):
        os.makedirs(OUTPUT_CSV_DIR)

    pd.DataFrame(x).to_csv(OUTPUT_CSV_DIR + '/' + 'x_train.csv')
    pd.DataFrame(y).to_csv(OUTPUT_CSV_DIR + '/' + 'y_train.csv')
