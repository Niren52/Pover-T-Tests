from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import pickle
import os


def select_features_variance(x_train, x_test, variance_threshold=0.05):
    print('Discarding features with less than {} variance...'.format(variance_threshold))
    print('Features before: {}'.format(x_train.shape[1]))
    selector = VarianceThreshold(threshold=variance_threshold)
    x_train = selector.fit_transform(x_train)
    x_test = selector.transform(x_test)
    print('Features after: {}'.format(x_train.shape[1]))
    return x_train, x_test


def select_features_L1(x_train, y_train, x_test):
    print('Fitting linear model with L1 norm to select features ...')
    print('Features before: {}'.format(x_train.shape[1]))
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    x_train = model.transform(x_train)
    x_test = model.transform(x_test)
    print('Features after: {}'.format(x_train.shape[1]))
    return x_train, x_test


def select_features_tree(x_train, y_train, x_test, n_trees=10):
    print('Fitting tree-based model to select features ...')
    print('Features before: {}'.format(x_train.shape[1]))
    clf = ExtraTreesClassifier(n_estimators=n_trees)
    clf.fit(x_train, y_train)
    model = SelectFromModel(clf, prefit=True)
    x_train = model.transform(x_train)
    x_test = model.transform(x_test)
    print('Features after: {}'.format(x_train.shape[1]))
    return x_train, x_test


def select_features_rfe(x_train, y_train, x_test, country, feature_selection_path='../data/selected_features_{}.pkl'):
    feature_selection_path = feature_selection_path.format(country)
    if os.path.exists(feature_selection_path):
        selected_features = pickle.load(open(feature_selection_path, 'rb'))
        x_train = x_train[:, selected_features]
        x_test = x_test[:, selected_features]
        return x_train, x_test
    print('Doing recursive feature elimination ...')
    clf = LogisticRegression(C=0.55, penalty='l2')
    selector = RFECV(clf, step=1, cv=6, verbose=True)
    selector = selector.fit(x_train, y_train)
    print('Feature support: ', selector.support_)
    print('Feature ranking: ', selector.ranking_)
    x_train = selector.transform(x_train)
    x_test = selector.transform(x_test)
    pickle.dump(selector.support_, open(feature_selection_path, 'wb'))
    return x_train, x_test
