import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Load the dataset
data = pd.read_csv("B0006_discharge.csv")

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data[['cycle']])
y_scaled = scaler.fit_transform(data[['capacity']])

# Reshape the input data for LSTM
X = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Create an LSTM model
model = Sequential()
model.add(LSTM(units=100, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=50,activation='relu'))
model.add(Dense(units=20,activation='relu'))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions to get actual capacity values
y_pred_actual = scaler.inverse_transform(y_pred).flatten()
y_test_actual = scaler.inverse_transform(y_test).flatten()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
print(f"Root Mean Squared Error (RMSE): {rmse}")

r2 = r2_score(y_test_actual, y_pred_actual)
print("accuracy is {}".format(r2))
