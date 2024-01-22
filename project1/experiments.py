from cleaning.adult import adult_data
from cleaning.breast_cancer import breast_data
from cleaning.ionosphere import iono_data
from cleaning.wine import wine_data
from evaluation import evaluate, evaluate_by_dataset_size, evaluate_log_reg_learning_rate

data_path = 'datasets/'
test_size = 0.1
shuffle = True
prompt = '\nPress enter to continue...'

print('\nAdult Dataset:')
X_train, y_train, X_test, y_test = adult_data(data_path, test_size, shuffle)
evaluate(X_train, y_train, X_test, y_test)
evaluate_by_dataset_size(X_train, y_train, X_test, y_test, 5)
evaluate_log_reg_learning_rate(X_train, y_train, X_test, y_test, [0.1, 0.05, 0.01], 25)

input(prompt)

print('\nIonosphere Dataset:')
X_train, y_train, X_test, y_test = iono_data(data_path, test_size, shuffle)
evaluate(X_train, y_train, X_test, y_test)
evaluate_by_dataset_size(X_train, y_train, X_test, y_test, 5)
evaluate_log_reg_learning_rate(X_train, y_train, X_test, y_test, [0.03, 0.015, 0.01], 25)

input(prompt)

print('\nWine Quality Dataset:')
X_train, y_train, X_test, y_test = wine_data(data_path, test_size, shuffle)
evaluate(X_train, y_train, X_test, y_test)
evaluate_by_dataset_size(X_train, y_train, X_test, y_test, 5)
evaluate_log_reg_learning_rate(X_train, y_train, X_test, y_test, [0.01, 0.002, 0.001], 25)

input(prompt)

print('\nBreast Cancer Dataset:')
X_train, y_train, X_test, y_test = breast_data(data_path, test_size, shuffle)
evaluate(X_train, y_train, X_test, y_test)
evaluate_by_dataset_size(X_train, y_train, X_test, y_test, 5)
evaluate_log_reg_learning_rate(X_train, y_train, X_test, y_test, [0.1, 0.04, 0.01], 25)
