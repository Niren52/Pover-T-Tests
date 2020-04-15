from data_pipeline import get_data
import numpy as np
import xgboost as xgb
from utils import create_submission

# A
ax_train, ay_train, ax_test, a_idxs = get_data('a')
x_train = xgb.DMatrix(ax_train, label=ay_train)
x_test = xgb.DMatrix(ax_test)
neg_pos_rate = np.sum(ay_train == 0) / np.sum(ay_train == 1)
params = {'max_depth': 2,
          'eta': 0.1,  # learning rate
           #'scale_pos_weight': neg_pos_rate,  # Balance classes ?
          'silent': 1,
          'objective': 'binary:logistic',
          'nthread': 4,
          'eval_metric': ['logloss']
          }
bst = xgb.train(params=params,
                dtrain=x_train,
                num_boost_round=507,
                #early_stopping_rounds=27,
                verbose_eval=1)
ay_test = bst.predict(x_test)

# B
bx_train, by_train, bx_test, b_idxs = get_data('b')
x_train = xgb.DMatrix(bx_train, label=by_train)
x_test = xgb.DMatrix(bx_test)
neg_pos_rate = np.sum(by_train == 0) / np.sum(by_train == 1)
params = {'max_depth': 2,
          'eta': 0.1,  # learning rate
          #'scale_pos_weight': neg_pos_rate,  # Balance classes ?
          'silent': 1,
          'objective': 'binary:logistic',
          'nthread': 4,
          'eval_metric': ['logloss']
          }
bst = xgb.train(params=params,
                dtrain=x_train,
                num_boost_round=132,
                #early_stopping_rounds=4,
                verbose_eval=1)
by_test = bst.predict(x_test)

# C
cx_train, cy_train, cx_test, c_idxs = get_data('c')
x_train = xgb.DMatrix(cx_train, label=cy_train)
x_test = xgb.DMatrix(cx_test)
neg_pos_rate = np.sum(cy_train == 0) / np.sum(cy_train == 1)
params = {'max_depth': 2,
          'eta': 0.1,  # learning rate
          # 'scale_pos_weight': neg_pos_rate,  # Balance classes ?
          'silent': 1,
          'objective': 'binary:logistic',
          'nthread': 4,
          'eval_metric': ['logloss']
          }
bst = xgb.train(params=params,
                dtrain=x_train,
                num_boost_round=142,
                #early_stopping_rounds=10,
                verbose_eval=1)
cy_test = bst.predict(x_test)

# Create submission
create_submission(ay_test, by_test, cy_test, a_idxs, b_idxs, c_idxs)