import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix


header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('./data/3.housing.csv', delim_whitespace=True, names=header)
df = pd.DataFrame(data)

# print(data)

array = data.values
X = array[:, 0:13]
Y = array[:, 13]
scaler = MinMaxScaler(feature_range=(0,1))
rescaled_X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size=0.2)
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# 모델 선택 및 분할
model = LinearRegression()
model.fit(X, Y)
y_pred = model.predict(X_test)

df_Y_test = pd.DataFrame(Y_test)
df_Y_pred = pd.DataFrame(y_pred)

plt.clf()
plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.scatter(range(len(X_test)), y_pred, color='red', label='Predicted Values', marker='x')

plt.title("Index")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')

# KFold Cross Validation
fold = KFold(n_splits=10, shuffle=True, random_state=42)
mse_scores = cross_val_score(model, X, Y, cv=fold, scoring='neg_mean_squared_error')
print(mse_scores)

# Output the mean MSE
mean_mse = -mse_scores.mean()
# print(f'Mean MSE: {mean_mse}')