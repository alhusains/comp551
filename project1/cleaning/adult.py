import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def adult_data(data_path, test_size=0.2, shuffle=False):
    train_data = pd.read_csv(data_path + 'adult_train.csv', names=header)
    test_data = pd.read_csv(data_path + 'adult_test.csv', comment = '|', names = header) #Comment = '|' to ignore first line

    data = pd.concat([train_data,test_data])

    print(data.shape)

    #In the provided data, '?' refers to a missing value, replace every '?' with NaN
    data = data.replace(' ?', np.nan)
    #drop rows with NaN
    data = data.dropna(axis=0)
    data = data.drop(data.columns[[2, 13]], axis=1)
    print(data.shape)

    #understand data types
    train_data.info()

    #edit the income column
    data['income']=data['income'].map({' <=50K': 0, ' >50K': 1, ' <=50K.': 0, ' >50K.': 1})


    #save the updated csv
    data.to_csv (data_path + 'updated_train_adult.csv', index = None, header=True)

    # One_hot encoding
    one_hot = pd.get_dummies(data, prefix_sep='_', drop_first=True)
    print(one_hot.shape)
    one_hot.to_csv (data_path + 'encoded_train_adult.csv', index = None, header=True)

    #split data to train and test
    train_data, test_data = train_test_split(one_hot, test_size=test_size, shuffle=shuffle, random_state=11)

    #save the numpy arrays
    xtrain = np.array(train_data.drop(columns=['income']))
    ytrain = np.array(train_data['income'])
    print(xtrain.shape)
    print(ytrain)
    xtest = np.array((test_data.drop(columns=['income'])))
    ytest = np.array((test_data['income']))


    ###Apply some statistics to understand the data better
    #counting the income distribution
    p = sns.countplot(data=one_hot, x='income')
    plt.show()

    #statistical description of the data from pandas
    described = one_hot.describe()
    print(described)

    #plotting features against each other
    plt.scatter(one_hot['age'], one_hot['hours_per_week'])
    plt.show()

    #data correlation plots
    corr = one_hot.corr()
    print(corr)

    #histogram of age
    hist = one_hot.hist(column='age', bins=5)
    plt.show(hist)
    return xtrain, ytrain, xtest, ytest


header = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship",
          "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]

if __name__ == '__main__':
    data_path = 'C:/Users/alhus/Documents/McGill/Winter 2020/Applied Machine Learning/Project 1/'
    xtrain, ytrain, xtest, ytest = adult_data(data_path)
