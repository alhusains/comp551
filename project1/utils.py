import csv

import numpy as np


def iono_data(data_path, test_size=0.2, shuffle=False):
    X = []
    y = []

    # load data
    with open(data_path, mode='r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            try:
                X.append([float(e) for e in line[:-1]])
                y.append([1 if line[-1] == 'g' else 0])
            except ValueError:
                pass

    X = np.array(X)
    y = np.array(y)

    # shuffle data
    if shuffle:
        s = np.random.RandomState(11).permutation(X.shape[0])
        X = X[s]
        y = y[s]

    # train-test split
    i = int((1 - test_size) * X.shape[0])
    X_train = X[:i]
    y_train = y[:i]
    X_test = X[i:]
    y_test = y[i:]

    return X_train, y_train, X_test, y_test


def wine_data(data_path, test_size=0.2, shuffle=False):
    X = []
    y = []

    # load data
    with open(data_path, mode='r') as f:
        reader = csv.reader(f, delimiter=';')
        for line in reader:
            try:
                X.append([float(e) for e in line[:-1]])
                y.append([1 if int(line[-1]) >= 7 else 0])
            except ValueError:
                pass

    X = np.array(X)
    y = np.array(y)

    # shuffle data
    if shuffle:
        s = np.random.RandomState(11).permutation(X.shape[0])
        X = X[s]
        y = y[s]

    # train-test split
    i = int((1 - test_size) * X.shape[0])
    X_train = X[:i]
    y_train = y[:i]
    X_test = X[i:]
    y_test = y[i:]

    return X_train, y_train, X_test, y_test
