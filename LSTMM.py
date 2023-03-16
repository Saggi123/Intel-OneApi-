import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

import oneapi

oneapi.setup_environment()
from oneapi.dal import dataframe as df
from oneapi.dal.data_access import get_data_by_url

tickerSymbol = 'AAPL'
tickerData = get_data_by_url(f"https://query1.finance.yahoo.com/v7/finance/download/{tickerSymbol}?period1=0&period2=9999999999&interval=1mo&events=history&includeAdjustedClose=true", 'csv')
df = pd.DataFrame(tickerData.as_array())
df.columns = tickerData.column_names


df = df.set_index('Date')
df.index = pd.to_datetime(df.index)
df = df.sort_index(ascending=True)

train_size = int(len(df) * 0.7)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

scaler = MinMaxScaler(feature_range=(0,1))
train_data = scaler.fit_transform(train_df)
test_data = scaler.transform(test_df)

def create_sequences(data, seq_length):
    X = []
    y = []

    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])

    return np.array(X), np.array(y)

seq_length = 100

train_X, train_y = create_sequences(train_data, seq_length)
test_X, test_y = create_sequences(test_data, seq_length)

train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

model = keras.Sequential([
    layers.LSTM(units=50, activation='relu', return_sequences=True, input_shape=(train_X.shape[1], 1)),
    layers.Dropout(0.2),
    layers.LSTM(units=60, activation='relu', return_sequences=True),
    layers.Dropout(0.3),
    layers.LSTM(units=80, activation='relu', return_sequences=True),
    layers.Dropout(0.4),
    layers.LSTM(units=120, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(units=1)
])

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(train_X, train_y, epochs=50, batch_size=32, validation_data=(test_X, test_y))

model.save('keras_model.h5')

def predict(model, data, scaler):
    inputs = data[len(data) - len(test_X) - seq_length:]
    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(seq_length, inputs.shape[0]):
        X_test.append(inputs[i-seq_length:i,0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return scaler.inverse_transform(model.predict(X_test))

predicted_prices = predict(model, df['Close'].values.reshape(-1,1), scaler)

plt.figure(figsize=(16,8))
plt.plot(df.index, df['Close'], label='Actual Price')
plt.plot(df.index, predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
