import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

boston = datasets.load_diabetes()

# print(boston.keys())
# print(boston.data)

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

diabetes_X = diabetes_X[:, np.newaxis, 2]

X_train = diabetes_X[:-100]
X_test = diabetes_X[-100:]

y_train = diabetes_y[:-100]
y_test = diabetes_y[-100:]

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color="green")
plt.plot(X_test, y_pred, color="red", linewidth=2)

plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Linear regression model')

plt.show()