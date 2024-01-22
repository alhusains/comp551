from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from feature_extraction import get_features


def train_and_test(X_train, y_train, X_test, y_test, classifiers, feature_types):
    print('\nExtracting features', [feature_type.name for feature_type in feature_types], '...')
    train_features = {
        feature_type: get_features(X_train, feature_type)
        for feature_type in feature_types
    }

    for classifier in classifiers:
        clf = classifier['clf']
        params = classifier['params']

        # training
        best_acc = 0
        best_clf = None
        best_params = None
        best_feature_type = None
        best_vectorizer = None

        for feature_type in feature_types:
            X_train_features, vectorizer = train_features[feature_type]
            b_clf, b_params, mean_acc = train_model(X_train_features, y_train, clf, params, feature_type)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_clf = b_clf
                best_params = b_params
                best_feature_type = feature_type
                best_vectorizer = vectorizer

        # testing
        X_test_features, _ = get_features(X_test, best_feature_type, best_vectorizer)
        test_model(X_test_features, y_test, best_clf, best_params, best_feature_type)


def train_model(X_train, y_train, clf, params, feature_type):
    print('\n==== {}: hyperparameter tuning with 5-fold cross-validation ===='.format(type(clf).__name__))
    best_clf, best_params, mean_acc = k_fold_cross_validation_tune_hyperparams(5, X_train, y_train, clf, params)
    print('feature type:', feature_type.name)
    print('best hyperparameters:', best_params)
    print('mean accuracy: {:.1f}% ({})'.format(100 * mean_acc, mean_acc))
    return best_clf, best_params, mean_acc


def k_fold_cross_validation_tune_hyperparams(k, X_train, y_train, clf, params):
    gs = GridSearchCV(clf, params, scoring='accuracy', refit=True, cv=k, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def test_model(X_test, y_test, clf, params, feature_type):
    print('\n==== {}: accuracy on test dataset ===='.format(type(clf).__name__))
    yh = clf.predict(X_test)
    acc = accuracy_score(y_test, yh)
    print('feature type:', feature_type.name)
    print('hyperparameters:', params)
    print('accuracy: {:.1f}% ({})'.format(100 * acc, acc))
    return acc
