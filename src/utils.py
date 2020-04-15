import pandas as pd
import numpy as np

def _make_country_sub(preds, test_idxs, country):
    # make sure we code the country correctly
    assert country in ['A', 'B', 'C']

    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds,
                               columns=['poor'],
                               index=test_idxs)

    # add the country code for joining later
    country_sub['country'] = country
    return country_sub[['country', 'poor']]


def create_submission(a_preds, b_preds, c_preds, a_idxs, b_idxs, c_idxs, name='submission'):
    """
    Create submission file from predictions
    """
    subm = pd.read_csv('../submissions/sample_submission.csv')
    submid = subm['id'].values
    a_sub = _make_country_sub(a_preds, a_idxs, 'A')
    b_sub = _make_country_sub(b_preds, b_idxs, 'B')
    c_sub = _make_country_sub(c_preds, c_idxs, 'C')
    submission = pd.concat([a_sub, b_sub, c_sub])
    submission = submission.reindex(submid)
    submission.to_csv('../submissions/{}.csv'.format(name))


def report(results, n_top=3):
    """
    Report CV scores
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")