import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets

boston = datasets.load_diabetes()

X = boston.data
y = boston.target

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(y, y_pred)

plt.plot(y, y, color='green', linestyle='dashed')

plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Linear regression model')

plt.show()