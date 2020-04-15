from sklearn.svm import SVC
from data_pipeline import get_data
from sklearn.model_selection import cross_val_score
import numpy as np


def pr(x_train, y_i, y_train):
    """
    Computes the Logistic regression prior with Naive Bayes: P(feature|y=y_i)
    :param x_train: Training data matrix
    :param y_i: Binary label (integer: 0 or 1)
    :param y_train: Training labels vector
    :return: P(feature|y=y_i) for each feature
    """
    p = x_train[y_train == y_i].sum(0)
    return (p + 1) / ((y_train == y_i).sum() + 1)


def get_mdl(x_train, y_train):
    """
    Performs logistic regression with a Naive Bayes prior (multiplies each feature by the NB prior,
    to make the input features carry this prior knowledge)
    :param x_train: Training data matrix
    :param y_train: Training labels vector
    :return: Fitted model and prior NB vector (for each feature)
    """
    x_t, idxs = binarize_data(x_train)  # x_train[:, idxs]
    r = np.log(pr(x_t, 1, y_train) / pr(x_t, 0, y_train))  # NB prior knowledge
    x_nb = np.multiply(x_t, r)  # Input features after applying NB prior knowledge
    return x_nb, r, idxs


def binarize_data(x, idxs=None):
    """
    Scales features of x to be in [0, 1]. Deletes features with unique values
    """
    if idxs is None:  # Train data
        # Find features with a unique value, and discard them
        mask = np.any(x != x[0, :], axis=0)
        idxs = np.argwhere(mask).flatten()
        x = x[:, idxs]
        x_max = np.amax(x, axis=0)
        x_min = np.amin(x, axis=0)
        return (x - x_min) / (x_max - x_min), idxs

    # Validation/test data
    x = x[:, idxs]
    mask = np.any(x != x[0, :], axis=0)
    idxs_val = np.argwhere(mask).flatten()
    x_max = np.amax(x[:, idxs_val], axis=0)
    x_min = np.amin(x[:, idxs_val], axis=0)
    x[:, idxs_val] = (x[:, idxs_val] - x_min)/(x_max - x_min)
    return x, idxs

x_train, y_train, x_test, test_idxs = get_data('b')
x_train_nb, r, idxs = get_mdl(x_train, y_train)
model = SVC(C=2, probability=True)
cv_score = cross_val_score(model, x_train_nb, y_train, cv=5, scoring='neg_log_loss')
print('CV logloss. Mean: {}. Sd: {}'.format(-np.mean(cv_score), np.std(cv_score)))
