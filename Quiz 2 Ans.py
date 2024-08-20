import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('./data/3.housing.csv', delim_whitespace=True, names=header)

array = data.values
X = array[:, 0:13]
Y = array[:, 13]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
# print(model.coef_, model.intercept_)

plt.scatter(range(len(X_test[:15])), Y_test[:15], color='blue')
plt.scatter(range(len(X_test[:15])), y_pred[:15], color='red', marker='x')
plt.xlabel("Index")
plt.ylabel("MEDX ($1,000)")
plt.show()

mse = mean_squared_error(Y_test, y_pred)
print(mse)

fold = KFold(n_splits=5)
mse_scores = cross_val_score(model, X, Y, cv=fold, scoring='neg_mean_squared_error')
print(mse_scores)
print(mse.mean())