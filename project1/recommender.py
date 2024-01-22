import numpy as np


def recommend_features(X, y, feature_types, prop_threshold=0.8, corr_threshold=0.2):
    """
    Recommends a subset of features suitable for training a binary classifier.

    Arguments:
        X: N x D NumPy array (N examples, D features)
        y: N x 1 NumPy array (N labels)
        feature_types: list of D (feature_name, feature_type) pairs
        prop_threshold: maximum proportion of a categorical feature
        corr_threshold: minimum correlation between a continuous feature and label

    Returns:
        list of D (feature_name, recommend) pairs
    """
    N = X.shape[0]
    D = X.shape[1]

    features = [True] * D

    for d in range(D):
        feature = X[:, d]
        feature_type = feature_types[d][1]
        if feature_type == 0:  # categorical
            distinct_values = np.unique(feature, return_counts=True)
            value_proportions = distinct_values[1] / N
            if np.any(value_proportions > prop_threshold):
                # many examples have the same value
                features[d] = False
        elif feature_type == 1:  # continuous
            feature = feature.astype(float)
            if np.std(feature) == 0:
                # all examples have the same value
                features[d] = False
            else:
                corr_coef = np.corrcoef(feature, y)[1, 0]
                if np.abs(corr_coef) < corr_threshold:
                    # weak correlation between feature and label
                    features[d] = False

    features = [(feature_type[0], features[i]) for i, feature_type in enumerate(feature_types)]

    return features
