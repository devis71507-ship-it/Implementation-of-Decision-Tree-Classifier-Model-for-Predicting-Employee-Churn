# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the employee dataset

2.Preprocess data (handle categorical values using encoding)

3.Split data into training and testing sets

4.Train Decision Tree Classifier using training data

5.Predict and evaluate model performance using accuracy
   

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: S.DEVI

RegisterNumber: 212225100008
*/
```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
dataset = pd.read_csv(r"C:\Users\acer\Downloads\Employee.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
salary_pred = regressor.predict([[6.5]])
print("Predicted Salary:", salary_pred)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

```

## Output:

<img width="765" height="622" alt="WhatsApp Image 2026-05-11 at 9 51 05 PM" src="https://github.com/user-attachments/assets/d699273a-e79f-45a9-88fd-b953cc0dd434" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
