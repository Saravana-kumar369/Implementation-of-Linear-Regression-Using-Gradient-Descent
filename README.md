# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1)Import necessary libraries: numpy, pandas, and StandardScaler from sklearn.preprocessing.
2)Read the dataset '50_Startups.csv' using pd.read_csv.
3)Extract input features X and target variable y from the dataset, excluding the first row.
4)Standardize the input features (X) and target variable (y) using StandardScaler.
5)Perform linear regression using gradient descent to find the weights (theta) that minimize the mean squared error between predictions and actual target values.
6)Print the predicted value for a new input data point after standardizing it and using the computed weights (theta) to predict the target value.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SARAVANA KUMAR
RegisterNumber: 212222230133
*/
```
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=100):
  x=np.c_[np.ones(len(x1)),x1]
  theta=np.zeros(x.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(x).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
  return theta
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_Scaled=scaler.fit_transform(x1)
y1_Scaled=scaler.fit_transform(y)
theta=linear_regression(x1_Scaled, y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/Saravana-kumar369/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/117925254/08ed9af1-a18f-4b24-8554-600d36fe2bd1)
![image](https://github.com/Saravana-kumar369/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/117925254/38e16b4a-248b-4c92-984b-4d7afd88cae8)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
