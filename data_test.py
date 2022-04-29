import pyedflib
import numpy as np
import csv
import  os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32520, 10)
        self.fc2 = nn.ReLU()
        self.softmax = nn.Softmax(4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)

def edf_get_oximetry(edf_path):
    edf = pyedflib.EdfReader(edf_path)
    i_position = np.where(np.array(edf.getSignalLabels()) == 'SaO2')[0][0]
    position = edf.readSignal(i_position)
    signal = np.array(position).astype(np.float)

    return signal


def main():
    signal = edf_get_oximetry('./signals/shhs1-200001.edf')
    print(len(signal))
    print((signal))
main()