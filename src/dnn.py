from sklearn.linear_model import LogisticRegression
from data_pipeline import get_data
from sklearn.model_selection import cross_val_score
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout


# Function to create model, required for KerasClassifier
def create_model(input_dim):
    # create model
    model = Sequential()
    model.add(Dense(50, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# cv logloss scores [mean (sd)]
# A:
# B:
# C:
x_train, y_train, x_test, test_idxs = get_data('a')

# create model
model = KerasClassifier(build_fn=lambda: create_model(x_train.shape[1]),
                        epochs=30,
                        batch_size=128)

# cv_score = cross_val_score(model, x_train, y_train, cv=5)
cv_score = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_log_loss')
print('CV logloss. Mean: {}. Sd: {}'.format(-np.mean(cv_score), np.std(cv_score)))
