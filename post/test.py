import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * np.random.randn(100, 1)

theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Binarizer, PolynomialFeatures

num_feature = np.random.randn(5,3)

minmax_scaler = MinMaxScaler(feature_range=(0,2))
standard_scaler = StandardScaler()

num_feature_scaler = minmax_scaler.fit_transform(num_feature)
num_feature_scaler= standard_scaler.fit_transform(num_feature)


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import torch
import torch.nn as nn




