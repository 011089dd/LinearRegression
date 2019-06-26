# Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

from sklearn import model_selection
from sklearn.linear_model import LinearRegression

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.30, random_state=2)

model = LinearRegression()
model.fit(X_train,Y_train)
Y_prediction = model.predict(X_test)

Y_pred = model.predict([[21]])
print(Y_pred)

# Visualising the Training set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, model.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, model.predict(X_test), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
