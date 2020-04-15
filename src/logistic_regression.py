from sklearn.linear_model import LogisticRegression
from data_pipeline import get_data
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import os

# cv logloss scores [mean (sd)]
# A: 0.2542 (0.008) [without class balancing], C=0.55, RFE
# B: 0.2064 (0.009) [without class balancing], C=2.5, linear_L1
# C:

x_train, y_train, x_test, test_idxs = get_data('a', reduce_features='rfe')

model = LogisticRegression(C=0.55, penalty='l2')

cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_log_loss')
print('CV logloss. Mean: {}. Sd: {}'.format(-np.mean(cv_score), np.std(cv_score)))
