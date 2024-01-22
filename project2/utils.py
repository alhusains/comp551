import re
import tarfile

import numpy as np
from sklearn.datasets import fetch_20newsgroups


def print_header(title):
    s = '=' * 32
    print('\n' + s, title, s)


def load_newsgroups(folder_name, shuffle=True, seed=42):
    remove = ('headers', 'footers', 'quotes')

    train_dataset = fetch_20newsgroups(data_home=folder_name, subset='train', remove=remove,
                                       shuffle=shuffle, random_state=seed)
    X_train = train_dataset.data
    y_train = list(train_dataset.target)

    test_dataset = fetch_20newsgroups(data_home=folder_name, subset='test', remove=remove,
                                      shuffle=shuffle, random_state=seed)
    X_test = test_dataset.data
    y_test = list(test_dataset.target)

    return X_train, y_train, X_test, y_test


def load_reviews(file_name, shuffle=True, seed=42):
    train_dataset = []
    test_dataset = []

    with tarfile.open(file_name, 'r:gz') as tar:
        for member in tar:
            match = review_regex.search(member.name)
            if match:
                review = tar.extractfile(member).read().decode()

                label = match.group('label')
                if label == 'pos':
                    label = 1
                else:  # label == 'neg'
                    label = 0

                dataset = match.group('dataset')
                if dataset == 'train':
                    train_dataset.append((review, label))
                else:  # dataset == 'test'
                    test_dataset.append((review, label))

    if shuffle:
        rs = np.random.RandomState(seed)
        rs.shuffle(train_dataset)
        rs.shuffle(test_dataset)

    X_train, y_train = zip(*train_dataset)
    X_test, y_test = zip(*test_dataset)

    return X_train, y_train, X_test, y_test


review_regex = re.compile(r'aclImdb/(?P<dataset>train|test)/(?P<label>pos|neg)/\d+_\d+\.txt')
