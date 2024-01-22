import numpy as np

class NaiveBayes:

    def __init__(self, D):
        self.log_pri = None
        self.y1 = None
        self.mu, self.s = np.zeros((2, D)), np.zeros((2, D))

    def fit(self, X, y_first):
        y = y_format(y_first)
        N, C = y.shape
        D = X.shape[1]
        mu, s = np.zeros((C, D)), np.zeros((C, D))
        for c in range(C):  # calculate mean and std
            inds = np.nonzero(y[:, c])[0]
            mu[c, :] = np.mean(X[inds, :], 0)
            s[c, :] = np.std(X[inds, :], 0)
        log_prior = np.log(np.mean(y, 0))
        self.log_pri = log_prior


    def predict(self, Xt):
        mu = self.mu
        log_likelihood = - np.sum(.5 * (((Xt[None, :, :] - mu[:, None, :])) ** 2), 2)
        arrayvalues = self.log_pri[:, None] + log_likelihood
        final_array = arrayvalues.T
        return_array = []
        for i in range(Xt.shape[0]):
            if (final_array[i, 0] < final_array[i, 1]):
                return_array.append(0)
            else:
                return_array.append(1)
        return np.array(return_array)


def y_format(y):
    y2 = np.copy(y)
    for i in range(len(y)):
        if y2[i] == 0:
            y2[i] = 1
        else:
            y2[i] = 0
    yfinal = np.array(np.column_stack([y, y2]).tolist())

    return yfinal
