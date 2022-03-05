import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data 
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import streamlit as st



start = '2010-01-01'
end = '2022-02-28'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker')
df = data.DataReader(user_input, 'yahoo', start, end)

#Describing Data
st.subheader('Data from 2010-2022')
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs Time Chart )
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)



