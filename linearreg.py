import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import pickle

df_05_charge = pd.read_csv('B0005_charge.csv')
df_05_discharge = pd.read_csv('B0005_discharge.csv')

df_06_charge = pd.read_csv("B0006_charge (1).csv")
df_06_discharge = pd.read_csv("B0006_discharge.csv")

df_07_charge = pd.read_csv("B0007_charge (1).csv")
df_07_discharge = pd.read_csv("B0007_discharge.csv")

df_18_discharge = pd.read_csv("B0018_discharge.csv")

d = pd.merge(df_05_discharge,df_06_discharge,how='outer')

d_final = pd.merge(d,df_07_discharge,how='outer')

#splitting dataset into X and Y
X = d_final.iloc[:,[0,1,3,4,5,6,7]]
Y = d_final.iloc[:,-1]

# splitting datasets into training and testing data with a 70 30 train test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

# Initializing and training the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train,Y_train)

# Evaluating error for Linear Regression
y_pred = linear_model.predict(X_test)
acc = mean_squared_error(Y_test,y_pred)
print(acc)

# Evaluating model based on r2 score
r2 = r2_score(Y_test,y_pred)
percentage_score = r2 * 100
print("the model is {0} percent accurage".format(percentage_score))

# Plotting Predicted vs actual values
plt.scatter(Y_test,y_pred)
plt.show()