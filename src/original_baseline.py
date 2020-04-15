import os
import numpy as np
import pandas as pd


DATA_DIR = '../data'
NUMERICAL_TYPES = ['int64', 'float64']

data_paths_hhold = {'A': {'train': os.path.join(DATA_DIR, 'A_hhold_train.csv'),
                          'test': os.path.join(DATA_DIR, 'A_hhold_test.csv')},

                    'B': {'train': os.path.join(DATA_DIR, 'B_hhold_train.csv'),
                          'test': os.path.join(DATA_DIR, 'B_hhold_test.csv')},

                    'C': {'train': os.path.join(DATA_DIR, 'C_hhold_train.csv'),
                          'test': os.path.join(DATA_DIR, 'C_hhold_test.csv')}}


# load training data
a_train = pd.read_csv(data_paths_hhold['A']['train'], index_col='id')
b_train = pd.read_csv(data_paths_hhold['B']['train'], index_col='id')
c_train = pd.read_csv(data_paths_hhold['C']['train'], index_col='id')


# Standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df


def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))

    df = standardize(df)
    print("After standardization {}".format(df.shape))

    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))

    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})

    df.fillna(0, inplace=True)

    return df

print("Country A")
aX_train = pre_process_data(a_train.drop('poor', axis=1))
ay_train = np.ravel(a_train.poor)

print("\nCountry B")
bX_train = pre_process_data(b_train.drop('poor', axis=1))
by_train = np.ravel(b_train.poor)

print("\nCountry C")
cX_train = pre_process_data(c_train.drop('poor', axis=1))
cy_train = np.ravel(c_train.poor)

aX_train.head()

from sklearn.ensemble import RandomForestClassifier


def train_model(features, labels, **kwargs):
    # instantiate model
    model = RandomForestClassifier(n_estimators=50, random_state=0)

    # train model
    model.fit(features, labels)

    # get a (not-very-useful) sense of performance
    accuracy = model.score(features, labels)
    print("In-sample accuracy: {}".format(accuracy))

    return model

model_a = train_model(aX_train, ay_train)
model_b = train_model(bX_train, by_train)
model_c = train_model(cX_train, cy_train)


# load test data
a_test = pd.read_csv(data_paths_hhold['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths_hhold['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths_hhold['C']['test'], index_col='id')

# process the test data
a_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
b_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
c_test = pre_process_data(c_test, enforce_cols=cX_train.columns)

a_preds = model_a.predict_proba(a_test)
b_preds = model_b.predict_proba(b_test)
c_preds = model_c.predict_proba(c_test)


def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']

    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'],
                               index=test_feat.index)

    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]

# convert preds to data frames
a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')

submission = pd.concat([a_sub, b_sub, c_sub])
submission.head()
submission.tail()
submission.to_csv('../submissions/sample_submission.csv')
