from sklearn.svm import SVC
from data_pipeline import get_data
from sklearn.model_selection import cross_val_score
import numpy as np

# cv logloss scores [mean (sd)]
# A: 0.278571 (0.011237) [without class balancing], C=2
# B:
# C:
x_train, y_train, x_test, test_idxs = get_data('a')
model = SVC(C=2, probability=True)
cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_log_loss')
print('CV accuracy. Mean: {}. Sd: {}'.format(-np.mean(cv_score), np.std(cv_score)))
