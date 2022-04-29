import pyedflib
import numpy as np
import csv


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