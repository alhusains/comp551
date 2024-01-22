from classifier_params import newsgroups_classifiers, newsgroups_feature_types, reviews_feature_types, reviews_classifiers
from evaluation import train_and_test
from utils import print_header, load_newsgroups, load_reviews

# newsgroups dataset

print_header('Newsgroups Dataset')
X_train, y_train, X_test, y_test = load_newsgroups('datasets/', shuffle=True, seed=42)
train_and_test(X_train, y_train, X_test, y_test, newsgroups_classifiers, newsgroups_feature_types)

# reviews dataset

print_header('Reviews Dataset')
X_train, y_train, X_test, y_test = load_reviews('datasets/aclImdb_v1.tar.gz', shuffle=True, seed=42)
train_and_test(X_train, y_train, X_test, y_test, reviews_classifiers, reviews_feature_types)
