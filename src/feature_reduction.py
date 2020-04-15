from sklearn.decomposition import PCA
import numpy as np


def pca_reduction(x_train, x_test, n_components=200):
    print('Performing PCA feature reduction with {} components...'.format(n_components))
    print('Features before: {}'.format(x_train.shape[1]))
    pca = PCA(n_components=n_components, svd_solver='full')
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    print('Features after: {}'.format(x_train.shape[1]))
    print('Explained variance ratio: {}'.format(np.cumsum(pca.explained_variance_ratio_)[-1]))
    return x_train, x_test
