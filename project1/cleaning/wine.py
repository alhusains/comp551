import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def wine_data(data_path, test_size=0.2, shuffle=False):
    data = pd.read_csv(data_path + 'winequality-red.csv', sep=';')
    print('data shape')
    print(data.shape)

    # understand data types
    data.info()

    # Reformat the labels column
    for i in range(0, 1599):
        if data.iloc[i, 11] >= 7:
            data.iloc[i, 11] = 1
        else:
            data.iloc[i, 11] = 0

    data.to_csv(data_path + 'updated_train_wine.csv', index=None, header=True)

    # calculate correlation between features and drop unnecessary features (with corr >0.9)
    corr = data.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = data.columns[columns]
    data = data[selected_columns]
    print(data.shape)


    #split data to train and test
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=shuffle, random_state=11)

    # save the numpy arrays
    xtrain = np.array(train_data.drop(columns=['quality']))
    ytrain = np.array(train_data['quality'])
    print(xtrain.shape)
    print(ytrain)
    xtest = np.array((test_data.drop(columns=['quality'])))
    ytest = np.array((test_data['quality']))

    ###Apply some statistics to understand the data better
    # counting the income distribution
    p = sns.countplot(data=data, x='quality')
    plt.show()

    # statistical description of the data from pandas
    described = data.describe()
    print(described)

    # plotting features against each other
    plt.scatter(data['fixed acidity'], data['volatile acidity'])
    plt.show()

    # data correlation plots
    corr = data.corr()
    print(corr)

    # histogram of age
    hist = data.hist(column='fixed acidity', bins=5)
    plt.show(hist)
    return xtrain, ytrain, xtest, ytest


if __name__ == '__main__':
    data_path = 'C:/Users/alhus/Documents/McGill/Winter 2020/Applied Machine Learning/Project 1/'
    xtrain, ytrain, xtest, ytest = wine_data(data_path)
