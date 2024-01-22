import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def breast_data(data_path, test_size=0.2, shuffle=False):
    data = pd.read_csv(data_path + 'wdbc_data.csv', names=header)
    print('data shape')
    print(data.shape)

    # understand data types
    data.info()

    # Reformat the labels column
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # drop unuseful columns (id column and last empty column)
    data = data.drop(data.columns[[0]], axis=1)
    print(data.shape)

    # save the updated csv
    data.to_csv(data_path + 'updated_train_breast.csv', index=None, header=True)

    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=shuffle, random_state=11)

    # save the numpy arrays
    xtrain = np.array(train_data.drop(columns=['diagnosis']))
    ytrain = np.array(train_data['diagnosis'])
    print(xtrain.shape)
    print(ytrain)
    xtest = np.array((test_data.drop(columns=['diagnosis'])))
    ytest = np.array((test_data['diagnosis']))

    ###Apply some statistics to understand the data better
    # counting the income distribution
    p = sns.countplot(data=data, x='diagnosis')
    plt.show()

    # statistical description of the data from pandas
    described = data.describe()
    print(described)

    #scatter plots and density plots of some features
    subset=data.iloc[:, 1:4]
    scatter_matrix(subset, alpha=0.2, figsize=(4, 4), diagonal='kde')

    # data correlation plots
    corr= data.corr()
    #sns.heatmap(corr, center=0)
    print(corr)

    # histogram of age
    hist = data.hist(column='radius_mean', bins=5)
    plt.show(hist)
    return xtrain, ytrain, xtest, ytest


header = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
          "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
          "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
          "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
          "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
          "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]

if __name__ == '__main__':
    data_path = 'C:/Users/alhus/Documents/McGill/Winter 2020/Applied Machine Learning/Project 1/'

    #xtrain, ytrain, xtest, ytest = breast_data(data_path)
    breast_data(data_path)
