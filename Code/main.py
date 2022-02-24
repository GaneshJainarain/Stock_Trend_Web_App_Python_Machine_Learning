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



scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)
#print(data_training_array)
