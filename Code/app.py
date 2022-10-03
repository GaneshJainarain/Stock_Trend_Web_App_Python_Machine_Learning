import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data 
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras import models    
import streamlit as st
from sklearn.metrics import r2_score



start = '2010-01-01'
end = '2022-03-03'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker')
df = data.DataReader(user_input, 'yahoo', start, end)

#Describing Data
st.subheader('Data from 2010-2022')
#n_years = st.slider('Years of prediction:', 1, 4)
#period = n_years * 365
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r', label = 'MA100')
plt.plot(ma200, 'g', label = 'MA200')
plt.plot(df.Close, 'b', label = 'Original Price')

plt.legend()
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []


for i in range(100, data_training_array.shape[0]):
    #appending data in our x_train, (i-100 because it should start from 0)
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])
#converting our arrays into numpy arrays so we can provide the data to our LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)


model = models.load_model('Code/keras_model.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scale_factor = 0.02099517
y_predicted = y_predicted / scale_factor
y_test = y_test / scale_factor
r_sqaured = r2_score(y_test, y_predicted)
print(r_sqaured)

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time(days)')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)