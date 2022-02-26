import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data 
from sklearn.preprocessing import MinMaxScaler


start = '2010-01-01'
end = '2019-12-31'
#will make this more dynamic but for now we are testing getting data
df = data.DataReader('AAPL', 'yahoo', start, end)

#Resetting our index to be numbers instead of the date
df = df.reset_index()
df.head()
df = df.drop(['Date', 'Adj Close'], axis = 1)

#Printing our data to the terminal
print(df.head())
print(df.tail())

#Plotting
#plt.plot(df.Close)
#plt.show()

#Getting Our Moving Averages
#MA100
ma100 = df.Close.rolling(100).mean()
print(ma100)
#MA200
ma200 = df.Close.rolling(200).mean()
print(ma200)

#Plotting our moving average ontop of our stock data
plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')

#Showing our plot
#plt.show()

#print(df.shape)
  
#Splitting data into training and testing 70/30
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)
print(data_training.head())
print(data_testing.head())


#Scaling down our training data
scaler = MinMaxScaler(feature_range=(0,1))
#converting to array, scaler.fit_transform auto gives us an array
data_training_array = scaler.fit_transform(data_training)
print(data_training_array)

#Now we divide our data into an X AND Y train
x_train = []
y_train = []

'''Lets take an example
The logic we are going to follow when predicting our values is simple.
lets say we are given 10 days worth of stock data and we want to predict the 11th day
34, 36, 33, 40, 39, 38, 37, 42, 44, 38, --> Predict(11th day)

Note that the value on the 11th day is always dependent on the previous 10 days
and will be in the same range as the previous data give or take.

The values of the 10 days are the x_train and the 11th day will become the y_train.
Now what if we want to predict the 12th day? The process is still the same, we take the past 10 days
we dont look ath the first day any more, and now the 11th day really is our 10th day. 
for example our original input was 
34, 36, 33, 40, 39, 38, 37, 42, 44, 38, 
and now lets say we predicted that the 11th day would yield us 39 in order to predict the 12th day
our new input would be
36, 33, 40, 39, 38, 37, 42, 44, 38, 39*
since we are measuring the 10day moving average here we take the past 10 days this becomes our new x_train








'''

