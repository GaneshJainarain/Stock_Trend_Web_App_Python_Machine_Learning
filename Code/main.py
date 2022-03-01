import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data 
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


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

#print(data_training.shape)
#print(data_testing.shape)
#print(data_training.head())
#print(data_testing.head())


#Scaling down our training data
scaler = MinMaxScaler(feature_range=(0,1))
#converting to array, scaler.fit_transform auto gives us an array
data_training_array = scaler.fit_transform(data_training)
#print(data_training_array)

#Now we divide our data into an X AND Y train
x_train = []
y_train = []

#Inserting values into our lists
#We start at 100 because we are taking a 100ma

#If we observe print(data_training_array.shape) we get (1761,1) meaning we have 1761 rows of
#data in the data_training_array var

#So instead of hard coding 1761 into the loop we can make it 
#dynamic and input data_training_array.shape[0] because 1761 is at our 0th index
for i in range(100, data_training_array.shape[0]):
    #appending data in our x_train, (i-100 because it should start from 0)
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])
#converting our arrays into numpy arrays so we can provide the data to our LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)

#Machine Learning Moddel
print(x_train.shape)
# (1661, 100, 1)

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, 
input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
#input shape, if we observe the x_train.shape we see we get 1661 rows and 100 columns
#100 columns because we defined the step to be 100,
#The first 100 vals have become our colms, they act as features for predicting our y train
model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4)) 

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))


#print(data_training_array.shape)

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

