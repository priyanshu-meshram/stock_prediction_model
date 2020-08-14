# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:43:14 2020

@author: Sharad
"""

#IMPORTING THE LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#GETTING THE DATA

dataset = pd.read_csv("training.csv")
training_set = dataset.iloc[::, 1:2].values

#FEATURE SCALING

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#GETTING DATA READY FOR THE LSTM MODEL

X_train = []
y_train = []
timeStep = 60
for i in range(timeStep, len(dataset)):
    X_train.append(training_set_scaled[i-timeStep:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#TRAINING THE MODEL

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(X_train, y_train, epochs = 150, batch_size = 32)

#PREDICTING THE STOCK PRICE FOR THE NEXT MONTH

predicted_stock_price = []
X_for_prediction = dataset.iloc[-60:, 1:2].values
X_for_prediction = sc.transform(X_for_prediction)
for x in range(20):
  data = X_for_prediction[x-60:]
  data = np.reshape(data,(1,len(data),1))
  prediction = model.predict(data)
  predicted_stock_price.append(prediction[0][0])
  X_for_prediction = np.append(X_for_prediction, prediction, axis = 0)

predicted_stock_price = sc.inverse_transform(X_for_prediction)
print(predicted_stock_price)

#PLOTTING THE RESULT

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[::, 1:2].values

plt.plot(real_stock_price, color="red", label="real stock price")
plt.plot(predicted_stock_price[0:20],color="blue", label="predicted stock price")
plt.title("Stock market prediction")
plt.xlabel("Time")
plt.ylabel("price")
plt.legend()
plt.show()