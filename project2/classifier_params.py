from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from feature_extraction import FeatureType

# newsgroups dataset

newsgroups_classifiers = [
    {
        'clf': LogisticRegression(random_state=42),
        'params': {
            'solver': ['lbfgs', 'newton-cg'],
            'C': [1.0, 0.1],
            'tol': [0.01, 0.001]
        }
    },
    {
        'clf': DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'min_samples_leaf': [1, 5, 10]
        }
    },
    {
        'clf': LinearSVC(random_state=42),
        'params': {
            'C': [1.0, 0.1, 0.01, 0.001]
        }
    },
    {
        'clf': AdaBoostClassifier(random_state=42, n_estimators=100, algorithm='SAMME'),
        'params': {
            'base_estimator': [Perceptron(), SGDClassifier()]
        }
    },
    {
        'clf': RandomForestClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [1, 5, 10]
        }
    }
]

newsgroups_feature_types = [
    FeatureType.ALL_WORDS,
    # FeatureType.EMOTION_WORDS,
    # FeatureType.SENTENCE_EMOTION,
    # FeatureType.PART_OF_SPEECH,
    # FeatureType.TEXT_LENGTH,
    # FeatureType.SENTENCE_LENGTH,
]

# reviews dataset

reviews_classifiers = [
    {
        'clf': LogisticRegression(random_state=42),
        'params': {
            'solver': ['lbfgs', 'newton-cg'],
            'C': [1.0, 0.1],
            'tol': [0.01, 0.001]
        }
    },
    {
        'clf': DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'min_samples_leaf': [1, 5, 10]
        }
    },
    {
        'clf': LinearSVC(random_state=42),
        'params': {
            'C': [1.0, 0.1, 0.01, 0.001]
        }
    },
    {
        'clf': AdaBoostClassifier(random_state=42, n_estimators=100, algorithm='SAMME'),
        'params': {
            'base_estimator': [Perceptron(), SGDClassifier()]
        }
    },
    {
        'clf': RandomForestClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [1, 5, 10]
        }
    }
]

reviews_feature_types = [
    FeatureType.ALL_WORDS,
    FeatureType.EMOTION_WORDS,
    FeatureType.SENTENCE_EMOTION,
    # FeatureType.PART_OF_SPEECH,
    # FeatureType.TEXT_LENGTH,
    # FeatureType.SENTENCE_LENGTH,
]
