import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/5.HeightWeight.csv', index_col=0)
df = pd.DataFrame(data)

array = data.values
# array.shape
X = array[:, 1]*0.453592 # 독립변수 : 종속변수에 영향을 주는 변수
Y = array[:, 0]*2.54# 종속변수 : 독립 변수에 영향을 받는 변수


# 데이터 분할
X = X.reshape(-1,1)
# testsize 0.2 = 20% 만 테스트에 쓰겠다.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size=0.2)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
df_Y_test = pd.DataFrame(Y_test)

plt.figure(figsize=(10,6))
plt.scatter(X_test, Y_test, color='blue', label='Actual Values')
plt.plot(X_test, y_pred, color='red', label='Predicted Values', marker='x')

plt.title("Index")
plt.xlabel("Weight")
plt.ylabel("Height")
plt.legend()
plt.show()
plt.savefig('./results/scatter.png')

mean = mean_absolute_error(Y_test, y_pred)
print(mean)