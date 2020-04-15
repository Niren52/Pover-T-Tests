import os
import pandas as pd
import numpy as np
import pickle
import warnings
from feature_selection import select_features_L1, select_features_tree, select_features_variance, select_features_rfe
from feature_reduction import pca_reduction

np.random.seed(0)

LABEL_COL = 'poor'
DATA_DIR = '../data'
NUMERICAL_TYPES = ['int64', 'float64']

data_paths_hhold = {'A': {'train': os.path.join(DATA_DIR, 'A_hhold_train.csv'),
                          'test': os.path.join(DATA_DIR, 'A_hhold_test.csv')},

                    'B': {'train': os.path.join(DATA_DIR, 'B_hhold_train.csv'),
                          'test': os.path.join(DATA_DIR, 'B_hhold_test.csv')},

                    'C': {'train': os.path.join(DATA_DIR, 'C_hhold_train.csv'),
                          'test': os.path.join(DATA_DIR, 'C_hhold_test.csv')}}

data_paths_indiv = {'A': {'train': os.path.join(DATA_DIR, 'A_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'A_indiv_test.csv')},

                    'B': {'train': os.path.join(DATA_DIR, 'B_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'B_indiv_test.csv')},

                    'C': {'train': os.path.join(DATA_DIR, 'C_indiv_train.csv'),
                          'test': os.path.join(DATA_DIR, 'C_indiv_test.csv')}}


def _load_data():
    data = {}
    data['a_hhold_train'] = pd.read_csv(data_paths_hhold['A']['train'], index_col='id')
    data['b_hhold_train'] = pd.read_csv(data_paths_hhold['B']['train'], index_col='id')
    data['c_hhold_train'] = pd.read_csv(data_paths_hhold['C']['train'], index_col='id')
    data['a_hhold_test'] = pd.read_csv(data_paths_hhold['A']['test'], index_col='id')
    data['b_hhold_test'] = pd.read_csv(data_paths_hhold['B']['test'], index_col='id')
    data['c_hhold_test'] = pd.read_csv(data_paths_hhold['C']['test'], index_col='id')
    data['a_indiv_train'] = pd.read_csv(data_paths_indiv['A']['train'], index_col=['id', 'iid'])
    data['b_indiv_train'] = pd.read_csv(data_paths_indiv['B']['train'], index_col=['id', 'iid'])
    data['c_indiv_train'] = pd.read_csv(data_paths_indiv['C']['train'], index_col=['id', 'iid'])
    data['a_indiv_test'] = pd.read_csv(data_paths_indiv['A']['test'], index_col=['id', 'iid'])
    data['b_indiv_test'] = pd.read_csv(data_paths_indiv['B']['test'], index_col=['id', 'iid'])
    data['c_indiv_test'] = pd.read_csv(data_paths_indiv['C']['test'], index_col=['id', 'iid'])
    return data


def _get_num_cols(df):
    return df.select_dtypes(include=NUMERICAL_TYPES).columns


def _get_cat_cols(df):
    return list(set(df.columns.values) - set(_get_num_cols(df)))


def _standardize_(series):
    return (series - series.mean()) / series.std()


def _standardize(df):
    num_cols = _get_num_cols(df)
    df[num_cols] = _standardize_(df[num_cols])  # (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    return df


def _contains_nan(x):
    return np.isnan(x).any()


def _get_data(country='a', fill_missing_num='median', fill_missing_cat='mode'):
    """
    Gets data for a specific country
    fill_missing_num in ['median', None]. None does not impute missing values
    fill_missing_cat in ['mode', None]. None does not impute missing values
    """
    preprocessed_file = '{}/preprocessed_{}.pkl'.format(DATA_DIR, country)
    if os.path.exists(preprocessed_file):
        ret = pickle.load(open(preprocessed_file, 'rb'))
        print('NaN in train: {}'.format(_contains_nan(ret['x_train'])))
        print('NaN in test: {}'.format(_contains_nan(ret['x_test'])))
        return ret['x_train'], ret['y_train'], ret['x_test'], ret['test_idxs']

    print('Preprocessing data ...')
    data = _load_data()
    train_hhold = data['{}_hhold_train'.format(country)]
    train_indiv = data['{}_indiv_train'.format(country)]
    test_hhold = data['{}_hhold_test'.format(country)]
    test_indiv = data['{}_indiv_test'.format(country)]

    # Remove country code
    train_hhold.drop('country', axis=1, inplace=True)
    test_hhold.drop('country', axis=1, inplace=True)
    train_indiv.drop('country', axis=1, inplace=True)
    test_indiv.drop('country', axis=1, inplace=True)
    assert 'country' not in train_hhold.columns

    # Standardize
    train_hhold = _standardize(train_hhold)
    test_hhold = _standardize(test_hhold)
    train_indiv = _standardize(train_indiv)
    test_indiv = _standardize(test_indiv)

    # Fill missing values hhold numerical data
    num_cols_hhold = _get_num_cols(train_hhold)
    if fill_missing_num == 'median':
        fill_with = train_hhold.loc[:, num_cols_hhold].median()  # Compute using test data as well?
        train_hhold.loc[:, num_cols_hhold].fillna(fill_with, inplace=True)
        test_hhold.loc[:, num_cols_hhold].fillna(fill_with, inplace=True)

    # Fill missing values hhold categorical data
    cat_cols_hhold = list(set(test_hhold.columns.values) - set(num_cols_hhold))
    if fill_missing_cat == 'mode':
        fill_with = train_hhold.loc[:, cat_cols_hhold].mode().iloc[0]
        train_hhold.loc[:, cat_cols_hhold].fillna(fill_with, inplace=True)
        test_hhold.loc[:, cat_cols_hhold].fillna(fill_with, inplace=True)

    # Transform categorical vars. into one-hot vectors
    train_hhold = pd.get_dummies(train_hhold, drop_first=True)  # drop first removes 1 explained column
    test_hhold = pd.get_dummies(test_hhold, drop_first=True)
    train_indiv = pd.get_dummies(train_indiv, drop_first=True)
    test_indiv = pd.get_dummies(test_indiv, drop_first=True)
    # print((set(train_indiv.columns.values)-{'poor'}).difference(set(test_indiv.columns.values)))

    # Remove multiindex (iid)
    train_indiv.index = train_indiv.index.droplevel(level=1)
    test_indiv.index = test_indiv.index.droplevel(level=1)

    # Aggregate indiv datasets + derived feature engineering
    train_indiv_groupby = train_indiv.groupby(train_indiv.index)
    test_indiv_groupby = test_indiv.groupby(test_indiv.index)
    agg_functions_num = [np.nanmean,
                         np.nanmedian]
    agg_functions_cat = [np.nanmean]  # mode. lambda x: stats.mode(x)[0][0]
    # The following aggregations might produce the following warnings:
    # * RuntimeWarning: Mean of empty slice (raised by np.nanmean: https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.nanmean.html)
    # * RuntimeWarning: All-NaN slice encountered (raised by np.nanmax: https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.nanmax.html)
    # * RuntimeWarning: Degrees of freedom <= 0 for slice (raised by np.nanstd: https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanstd.html)
    # * RuntimeWarning: All-NaN axis encountered (raised by some of these functions as well)
    # This might happen when, for a given column, the values of all individuals in a household
    # are NaN. In this case, all the aggregation operations produce NaNs, so it's safe to ignore
    # these warnings.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        num_cols_indiv = _get_num_cols(train_indiv)
        cat_cols_indiv = _get_cat_cols(train_indiv)
        agg_dict_train = {c: agg_functions_num for c in num_cols_indiv}
        agg_dict_train.update({c: agg_functions_cat for c in cat_cols_indiv})
        train_indiv = train_indiv_groupby.agg(agg_dict_train)
        train_indiv.columns = ['_'.join(x) for x in train_indiv.columns.ravel()]  # rename columns with a unique name
        train_indiv = train_indiv.join(train_indiv_groupby.size().to_frame(name='counts'))  # Individuals per household

        num_cols_indiv = _get_num_cols(test_indiv)
        cat_cols_indiv = _get_cat_cols(test_indiv)
        agg_dict_test = {c: agg_functions_num for c in num_cols_indiv}
        agg_dict_test.update({c: agg_functions_cat for c in cat_cols_indiv})
        test_indiv = test_indiv_groupby.agg(agg_dict_test)
        test_indiv.columns = ['_'.join(x) for x in test_indiv.columns.ravel()]
        test_indiv = test_indiv.join(test_indiv_groupby.size().to_frame(name='counts'))  # Individuals per household
    train_indiv.loc[:, 'counts'] = _standardize_(train_indiv.loc[:, 'counts'])
    test_indiv.loc[:, 'counts'] = _standardize_(test_indiv.loc[:, 'counts'])

    # Merge hhold and indiv
    train = pd.concat([train_hhold, train_indiv], axis=1)
    test = pd.concat([test_hhold, test_indiv], axis=1)
    print('NaN in train: {}'.format(train.isnull().values.any()))
    print('NaN in test: {}'.format(test.isnull().values.any()))

    # Match train/test columns
    to_drop = np.setdiff1d(test.columns, train.columns)
    to_add = np.setdiff1d(train.columns, test.columns)
    test.drop(to_drop, axis=1, inplace=True)
    train.drop(list(set(to_add) - {'poor'}), axis=1, inplace=True)
    print('Dropping columns {} from train set'.format(to_add))
    # Another option is to add them to the test set:
    # test = test.assign(**{c: 0 for c in to_add})

    # Align train/test columns
    train_cols = train.columns
    test = test.reindex_axis(train_cols, axis=1)

    # Return data
    y_train = train.poor.values
    x_train = train.drop(LABEL_COL, axis=1).values
    x_test = test.drop(LABEL_COL, axis=1).values
    test_idxs = test.index.values
    ret = {'x_train': x_train,
           'y_train': y_train,
           'x_test': x_test,
           'test_idxs': test_idxs,
           'columns': test.columns.values}
    pickle.dump(ret, open(preprocessed_file, 'wb'))
    return x_train, y_train, x_test, test_idxs


def _replace_nan_mean(x):
    idxs = np.where(np.isnan(x))
    x_mean = np.nanmean(x, axis=0)
    x[idxs] = np.take(x_mean, idxs[1])
    return x


def get_data(country, replace_nans='mean', variance_threshold=0, reduce_features='rfe', **kwargs):
    """
    Gets data for a specific country
    replace_nans in ['mean', None]. None does not impute missing values
    fill_missing_num in ['median', None]. None does not impute missing values
    fill_missing_cat in ['mode', None]. None does not impute missing values
    """
    x_train, y_train, x_test, test_idxs = _get_data(country, **kwargs)
    print('x_train shape: {}'.format(x_train.shape))
    print('x_test shape: {}'.format(x_test.shape))
    # If data with dummy variables still contains NaNs, replace them with 'replace_nans' applied along columns
    if replace_nans == 'mean':
        print('Replacing remaining NaNs with column mean ...')
        x_train = _replace_nan_mean(x_train)
        x_test = _replace_nan_mean(x_test)

    # Discard features with very low variance
    if variance_threshold > 0:  # Requires no NaNs in the dataset
        x_train, x_test = select_features_variance(x_train, x_test)

    # Feature selection/reduction
    if reduce_features == 'tree':
        x_train, x_test = select_features_tree(x_train, y_train, x_test)
    elif reduce_features == 'linear_L1':
        x_train, x_test = select_features_L1(x_train, y_train, x_test)
    elif reduce_features == 'pca':
        x_train, x_test = pca_reduction(x_train, x_test)
    elif reduce_features == 'rfe':
        x_train, x_test = select_features_rfe(x_train, y_train, x_test, country)
    print('Using {} features'.format(x_train.shape[1]))

    return x_train, y_train, x_test, test_idxs

# get_data('a')
# get_data('b')
# get_data('c')
