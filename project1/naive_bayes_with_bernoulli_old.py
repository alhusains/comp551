import numpy as np


class NaiveBayes:

    def __init__(self, D):
        self.log_pri = None
        self.y1=None
        self.likelihood_b=None
        self.mu, self.s = np.zeros((2, D)), np.zeros((2, D))

    def fit(self, X, y_first):
        self.y1 = y_first
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
        self.likelihood_b=bernoulli_likelihood(X,self.y1)

    def predict(self, Xt):
        ber_lh = self.predict_bern(Xt)
        gaus_lh = self.predictGaussian(Xt)
        log_likelihood = gaus_lh + ber_lh
        log_likelihood = log_likelihood + self.log_pri
        return_array = []
        for i in range(Xt.shape[0]):
            if (log_likelihood[i, 0] < log_likelihood[i, 1]):
                return_array.append(0)
            else:
                return_array.append(1)
        return np.array(return_array)


    def predictGaussian(self, Xt):
        mu = self.mu
        log_likelihood = - np.sum(.5 * (((Xt[None, :, :] - mu[:, None, :])) ** 2), 2)
        arrayvalues = self.log_pri[:,None] + log_likelihood
        final_array = arrayvalues.T
        return final_array

    def predict_bern(self, Xt):
        N, D = Xt.shape
        num_classes = 2
        result = np.zeros((N, num_classes))
        for i in range(N):
            result[i, :] = np.sum(np.log(self.likelihood_b) * Xt[i, :, None] + np.log(1 - self.likelihood_b) * (1 - Xt[i, :, None]), 0)
        return result

def y_format(y):
    y2 = np.copy(y)
    for i in range(len(y)):
        if y2[i] == 0:
            y2[i] = 1
        else:
            y2[i] = 0
    yfinal = np.array(np.column_stack([y, y2]).tolist())

    return yfinal

def bernoulli_likelihood(X,y):
    p = np.zeros((X.shape[1], 2))
    '''np.sum(X,axis=0)/X.shape[0]'''
    numofones = np.count_nonzero(y)
    numofzeros = np.count_nonzero(np.logical_not(y))
    for i in range(X.shape[1]):
        ones = np.count_nonzero(np.logical_and(X[:,i], y))
        zeros = np.count_nonzero(np.logical_and(X[:, i], np.logical_not(y)))
        p[i, 0] = (zeros + 1) / (numofzeros + 2)
        p[i, 1] = (ones + 1) / (numofones + 2)
    return p
