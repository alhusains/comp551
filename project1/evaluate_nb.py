from cleaning.adult import adult_data
from cleaning.breast_cancer import breast_data
from cleaning.ionosphere import iono_data
from cleaning.wine import wine_data
from evaluation import evaluate_naive_bayes

data_path = 'datasets/'
test_size = 0.2
shuffle = True
prompt = '\nPress enter to continue...'

print('\nAdult Dataset:')
X_train, y_train, X_test, y_test = adult_data(data_path, test_size, shuffle)
evaluate_naive_bayes(X_train, y_train, X_test, y_test)

input(prompt)

print('\nIonosphere Dataset:')
X_train, y_train, X_test, y_test = iono_data(data_path, test_size, shuffle)
evaluate_naive_bayes(X_train, y_train, X_test, y_test)

input(prompt)

print('\nWine Quality Dataset:')
X_train, y_train, X_test, y_test = wine_data(data_path, test_size, shuffle)
evaluate_naive_bayes(X_train, y_train, X_test, y_test)

input(prompt)

print('\nBreast Cancer Dataset:')
X_train, y_train, X_test, y_test = breast_data(data_path, test_size, shuffle)
evaluate_naive_bayes(X_train, y_train, X_test, y_test)
