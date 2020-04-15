from data_pipeline import get_data
import numpy as np
import xgboost as xgb

# Consider handling missing values with XGBoost
# (see point 1.4 in https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
# Hyperparameter tuning: https://cambridgespark.com/content/tutorials/hyperparameter-tuning-in-xgboost/index.html

# cv logloss scores [mean (sd)]
# A: 0.279504 (0.004652) [without class balancing], eta: 0.1, rounds: 507
# B: 0.214285 (0.019963) [without class balancing], eta: 0.1, rounds: 132
# C: 0.018061 (0.008919) [without class balancing], eta: 0.1, rounds: 142

x_train, y_train, x_test, test_idxs = get_data('b')
x_train = xgb.DMatrix(x_train, label=y_train)
neg_pos_rate = np.sum(y_train == 0) / np.sum(y_train == 1)
print('Negative-positive rate: {}'.format(neg_pos_rate))

# XGBoost parameters
params = {'max_depth': 2,
          'eta': 0.1,  # learning rate
          # 'scale_pos_weight': neg_pos_rate,  # Balance classes ?
          'silent': 1,
          'objective': 'binary:logistic',
          'nthread': 4,
          'eval_metric': ['logloss'] # , 'error'
          }

cv = xgb.cv(params=params,
            dtrain=x_train,
            num_boost_round=5000,
            nfold=5,
            early_stopping_rounds=10,
            verbose_eval=1)
print(cv)
