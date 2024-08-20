import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/5.HeightWeight.csv', index_col=0)


array = data.values
array.shape
X = array[:, 0] # 독립변수 : 종속변수에 영향을 주는 변수
Y = array[:, 1] # 종속변수 : 독립 변수에 영향을 받는 변수

print(X)


# 근속연수 * 연봉
XR = X.reshape(-1,1)
# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(XR,Y , test_size=0.2)


# # 모델 선택 및 분할
model = LinearRegression()
model.fit(X_train, Y_train)

# X_test로 값을 예측해봐
y_pred = model.predict(X_test)
print(y_pred)


plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.plot(range(len(y_pred)), y_pred, color='red', label='Predicted Values', marker='x')
plt.show()


# 모델 정확도 계산
mean = mean_absolute_error(Y_test, y_pred)
print(mean)

# -------------------------------------------------------------------------------------

