from sklearn.ensemble import AdaBoostClassifier
from data_pipeline import get_data
from sklearn.model_selection import cross_val_score
import numpy as np

# cv logloss scores [mean (sd)]
# A:
# B:
# C:
x_train, y_train, x_test, test_idxs = get_data('a', reduce_features=None)
model = AdaBoostClassifier(n_estimators=10, learning_rate=0.1)
#cv_score = cross_val_score(model, x_train, y_train, cv=5)
cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_log_loss')
print('CV logloss. Mean: {}. Sd: {}'.format(-np.mean(cv_score), np.std(cv_score)))
