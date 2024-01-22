import numpy as np


class LogisticRegression:

    def __init__(self, D, standardize=True):
        """
        Initializes the model parameters.

        Arguments:
            D: number of features
            standardize: whether to standardize features
        """
        self.w = np.zeros((D + 1, 1))
        self.standardize = standardize

    def fit(self, X, y, *, learning_rate=0.01, max_iter=1000, tolerance=0.0001, iter_collect_w=0):
        """
        Trains the model on the given dataset using batch gradient descent.

        Training stops after reaching a given number of iterations or tolerance, whichever comes first.

        Arguments:
            X: N x D NumPy array (N examples, D features)
            y: N x 1 NumPy array (N labels)
            learning_rate: learning rate of gradient descent
            max_iter: maximum number of iterations of gradient descent
            tolerance: threshold for change in cost needed to stop gradient descent
            iter_collect_w: number of iterations after which to collect weights

        Returns:
            list of (iteration_number, model_weights) pairs, collected every iter_collect_w iterations
        """
        N = X.shape[0]
        X = add_dummy_feature(standardize_features(X) if self.standardize else X)
        y = y.reshape((N, 1))

        weights = []

        # gradient descent
        cost = np.inf
        for i in range(1, max_iter + 1):
            # compute activation
            z = np.dot(X, self.w)
            a = sigmoid(z)

            # compute cost
            J = np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z)))
            if np.abs(cost - J) <= tolerance:
                break
            cost = J

            # compute gradient
            dw = np.dot(X.T, a - y) / N

            # update weights
            self.w -= learning_rate * dw

            # collect weights
            if iter_collect_w > 0 and i % iter_collect_w == 0:
                weights.append((i, np.copy(self.w)))

        return weights

    def predict(self, X):
        """
        Predicts the labels (0 or 1) for the given dataset using the model parameters.

        Arguments:
            X: N x D NumPy array (N examples, D features)

        Returns:
            N x 1 NumPy array (N labels)
        """
        X = add_dummy_feature(standardize_features(X) if self.standardize else X)

        z = np.dot(X, self.w)
        a = sigmoid(z)

        yh = np.where(a >= 0.5, 1, 0)
        return yh


def add_dummy_feature(X):
    """
    Adds a dummy feature with value 1 to the beginning of each feature vector.

    Arguments:
        X: N x D NumPy array (N examples, D features)

    Returns:
        N x (D + 1) NumPy array (N examples, D + 1 features)
    """
    X_new = np.ones((X.shape[0], X.shape[1] + 1))
    X_new[:, 1:] = X
    return X_new


def standardize_features(X):
    """
    Standardize features by subtracting the mean and dividing by the standard deviation.

    Arguments:
        X: N x D NumPy array (N examples, D features)

    Returns:
        N x D NumPy array
    """
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    stdev[stdev == 0] = 1
    X_new = (X - mean) / stdev
    return X_new


def sigmoid(z):
    """
    Computes the sigmoid/logistic function.

    Arguments:
        z: scalar or NumPy array

    Returns:
        scalar or NumPy array
    """
    a = 1 / (1 + np.exp(-z))
    return a
