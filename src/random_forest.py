from sklearn.ensemble import RandomForestClassifier
from data_pipeline import get_data
from sklearn.model_selection import cross_val_score
import numpy as np


x_train, y_train, x_test, test_idxs = get_data('b')
model = RandomForestClassifier(n_estimators=100, random_state=0)
cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_log_loss')
print('CV accuracy. Mean: {}. Sd: {}'.format(-np.mean(cv_score), np.std(cv_score)))
