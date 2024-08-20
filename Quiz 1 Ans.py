import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix

SL = 'sepal-length'
SW = 'sepal-width'
PL = 'Petal-length'
PW = 'petal-width'

header = [SL, SW, PL, PW, 'class']
data = pd.read_csv('./data/2.iris.csv', names=header)
# 시각화 먼저


array = data.values
X = array[:, 0:3]
Y = array[:, 4]
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)


print(rescaled_X)


(X_train, X_test, Y_train, Y_test) = train_test_split(rescaled_X, Y, test_size=0.3)


model = DecisionTreeClassifier(max_depth=)
fold = KFold(n_splits=10, shuffle = True)

