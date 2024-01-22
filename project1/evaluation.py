import itertools

import numpy as np

from logistic_regression import LogisticRegression
from naive_bayes import NaiveBayes


def evaluate_acc(y, yh):
    """
    Computes the accuracy of a binary classifier.

    Arguments:
        y: N x 1 NumPy array (N actual labels)
        yh: N x 1 NumPy array (N predicted labels)

    Returns:
        accuracy as a floating-point number between 0 and 1
    """
    acc = float((np.dot(y.T, yh) + np.dot(1 - y.T, 1 - yh)) / y.shape[0])
    return acc


def k_fold_cross_validation_tune_hyperparams(k, X_train, y_train, clf_init, clf_params):
    best_mean_acc = 0
    best_stdev_acc = 0
    best_params = None

    for param_perm in itertools.product(*clf_params.values()):
        params = dict(zip(clf_params.keys(), param_perm))
        mean_acc, stdev_acc = k_fold_cross_validation(k, X_train, y_train, clf_init, params)
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_stdev_acc = stdev_acc
            best_params = params

    return best_params, best_mean_acc, best_stdev_acc


def k_fold_cross_validation(k, X_train, y_train, clf_init, clf_params):
    datasets = k_fold_cross_validation_helper(k, X_train, y_train)

    accuracies = []

    for dataset in datasets:
        X_tr, y_tr, X_val, y_val = dataset

        clf = clf_init(X_tr, y_tr)
        clf.fit(X_tr, y_tr, **clf_params)
        yh = clf.predict(X_val)
        acc = evaluate_acc(y_val, yh)

        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    stdev_acc = np.std(accuracies)

    return mean_acc, stdev_acc


def k_fold_cross_validation_helper(k, X_train, y_train):
    N = X_train.shape[0]
    m = int(N / k)

    datasets = []

    for i in range(k):
        s1 = slice(0, i * m)
        s2 = slice(i * m, (i + 1) * m)
        s3 = slice((i + 1) * m, N)

        X_tr = np.concatenate((X_train[s1], X_train[s3]))
        y_tr = np.concatenate((y_train[s1], y_train[s3]))
        X_val = X_train[s2]
        y_val = y_train[s2]

        datasets.append((X_tr, y_tr, X_val, y_val))

    return datasets


def evaluate_logistic_regression(X_train, y_train, X_test, y_test):
    clf_init = lambda X_tr, y_tr: LogisticRegression(X_tr.shape[1])

    # 5-fold cross-validation
    clf_params = {'learning_rate': [0.1, 0.05, 0.01], 'max_iter': [1000, 2000], 'tolerance': [0.01, 0.001, 0.0001]}
    best_params, mean_acc, stdev_acc = k_fold_cross_validation_tune_hyperparams(5, X_train, y_train, clf_init, clf_params)
    print('\n==== logistic regression: 5-fold cross-validation ====')
    print('best hyperparameters:', best_params)
    print('mean accuracy: {:.1f}% ({})'.format(100 * mean_acc, mean_acc))
    print('standard deviation: {:.1f}% ({})'.format(100 * stdev_acc, stdev_acc))

    # accuracy on test dataset
    clf = clf_init(X_train, y_train)
    clf.fit(X_train, y_train, **best_params)
    yh = clf.predict(X_test)
    acc = evaluate_acc(y_test, yh)
    print('\n==== logistic regression: accuracy on test dataset ====')
    print('accuracy: {:.1f}% ({})'.format(100 * acc, acc))


def evaluate_naive_bayes(X_train, y_train, X_test, y_test):
    clf_init = lambda X_tr, y_tr: NaiveBayes(X_tr.shape[1])

    # 5-fold cross-validation
    mean_acc, stdev_acc = k_fold_cross_validation(5, X_train, y_train, clf_init, {})
    print('\n==== naive bayes: 5-fold cross-validation ====')
    print('mean accuracy: {:.1f}% ({})'.format(100 * mean_acc, mean_acc))
    print('standard deviation: {:.1f}% ({})'.format(100 * stdev_acc, stdev_acc))

    # accuracy on test dataset
    clf = clf_init(X_train, y_train)
    clf.fit(X_train, y_train)
    yh = clf.predict(X_test)
    acc = evaluate_acc(y_test, yh)
    print('\n==== naive bayes: accuracy on test dataset ====')
    print('accuracy: {:.1f}% ({})'.format(100 * acc, acc))


def evaluate(X_train, y_train, X_test, y_test):
    # logistic regression
    evaluate_logistic_regression(X_train, y_train, X_test, y_test)

    # naive bayes
    evaluate_naive_bayes(X_train, y_train, X_test, y_test)


def evaluate_by_dataset_size(X_train, y_train, X_test, y_test, k):
    # vary training set size
    m = int(X_train.shape[0] / k)
    for i in range(k):
        j = (i + 1) * m
        X_tr = X_train[:j]
        y_tr = y_train[:j]

        print('\nTraining Set Size:', j)
        evaluate(X_tr, y_tr, X_test, y_test)


def evaluate_log_reg_learning_rate(X_train, y_train, X_test, y_test, learning_rates, iter_collect_w):
    print('\n==== logistic regression: learning rates ====')
    clf_init = lambda X_tr, y_tr: LogisticRegression(X_tr.shape[1])

    for lr in learning_rates:
        print('\nlearning rate:', lr)

        # 5-fold cross-validation
        clf_params = {'learning_rate': [lr], 'max_iter': [1000, 2000], 'tolerance': [0.01, 0.001, 0.0001]}
        best_params, _, _ = k_fold_cross_validation_tune_hyperparams(5, X_train, y_train, clf_init, clf_params)

        clf = clf_init(X_train, y_train)
        weights = clf.fit(X_train, y_train, **best_params, iter_collect_w=iter_collect_w)

        for w in weights:
            clf.w = w[1]
            yh = clf.predict(X_test)
            acc = evaluate_acc(y_test, yh)
            print('accuracy after iteration {}: {:.1f}% ({})'.format(w[0], 100 * acc, acc))
