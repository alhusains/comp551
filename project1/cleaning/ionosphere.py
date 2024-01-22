import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.model_selection import train_test_split


def iono_data(data_path, test_size=0.2, shuffle=False):
    data = pd.read_csv(data_path + 'ionosphere_data.csv', header=None)

    print('data shape')
    print(data.shape)

    #Reformat labels
    data.iloc[:,34]=data.iloc[:,34].map({'g': 1, 'b': 0})

    # drop unuseful columns (2nd column all zeros)
    data = data.drop(data.columns[[1]], axis=1)
    print(data.shape)

    #split data to train and test
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=shuffle, random_state=11)

    #save the numpy arrays
    xtrain = np.array(train_data.drop(train_data.columns[(train_data.shape[1]-1)], axis=1))
    ytrain = np.array(train_data.iloc[:,(train_data.shape[1]-1)])
    print(xtrain.shape)
    print(ytrain)
    xtest = np.array(test_data.drop(test_data.columns[(train_data.shape[1]-1)], axis=1))
    ytest = np.array(test_data.iloc[:,(train_data.shape[1]-1)])

    ###Apply statistics for better understanding of the data
    p = sns.countplot(data=train_data, x=data.columns[(train_data.shape[1]-1)])
    plt.show()

    corr = data.corr()
    sns.heatmap(corr)
    print(corr)

    described = data.describe()
    print(described)

    x = data.loc[:, 2]
    y = data.loc[:, 3]
    plt.scatter(x, y)
    plt.show()

    return xtrain, ytrain, xtest, ytest


if __name__ == '__main__':
    data_path = 'C:/Users/alhus/Documents/McGill/Winter 2020/Applied Machine Learning/Project 1/'
    xtrain, ytrain, xtest, ytest = iono_data(data_path)
