import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data 
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import r2_score

start = '2007-03-03'
end = '2022-09-09'
#will make this more dynamic but for now we are testing getting data
df = data.DataReader('AAPL', 'yahoo', start, end)

#Resetting our index to be numbers instead of the date
df = df.reset_index()
df.head()
df = df.drop(['Date', 'Adj Close'], axis = 1)

#Printing our data to the terminal
#print(df.head())
#print(df.tail())

#Plotting
#plt.plot(df.Close)
#plt.show()

#Getting Our Moving Averages
#MA100
ma100 = df.Close.rolling(100).mean()
#print(ma100)
#MA200
ma200 = df.Close.rolling(200).mean()
#print(ma200)

#Plotting our moving average ontop of our stock data
#plt.figure(figsize = (12,6))
#plt.plot(df.Close)
#plt.plot(ma100, 'r')
#plt.plot(ma200, 'g')

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
#print(x_train.shape)
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

#Dense layer only has 1 unit because we are only predicting 1 value
model.add(Dense(units = 1))

#print(model.summary())

model.compile(optimizer='adam', loss= "mean_squared_error")
model.fit(x_train, y_train, epochs = 100)
model.save('keras_model.h5')


#Predicting values where we use the testing data, 30% was for testing

#print(data_testing.head())
#1761  28.955000
#1762  29.037500
#1763  29.004999
#1764  29.152500
#1765  29.477501
'''
observe how we get the 1761 value, in order to get this alue we need
the past 100 days, according to tie series analysis, 
the 100 values are located in our training data, so we must fetch them and append thosse 100 days/vals 
'''
#print(data_training.tail(100))
#1661  27.202499
#1662  27.000000
#1663  26.982500
#1664  27.045000
#1665  27.370001
#...         ...
#1756  29.072500
#1757  29.129999
#1758  29.315001
#1759  29.190001
#1760  29.182501
'''
The 100 values we need to append ^
'''
#We need previous 100 days, 
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

#Our testing data
#print(final_df.head())
#0  27.202499
#1  27.000000
#2  26.982500
#3  27.045000
#4  27.370001
'''
Notice how this data isnt scaled down, we can scale it down with a scaler transform
'''
input_data = scaler.fit_transform(final_df)
#print(input_data)
#print(input_data.shape)
#(855, 1)

# Working on our test now
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
#print(x_test.shape)
#print(y_test.shape)
#(755, 100, 1)
#(755,)

#Making our Predictions

y_predicted = model.predict(x_test)
#print(y_predicted.shape)
#(755, 1)
#print(y_test)
#print(y_predicted)
'''
All these values are scaled down, so we find the factror that they were scaled down
'''
#print(scaler.scale_)
#[0.02099517] -- Factor in which data was scaled down
#We need to divide y test and y predicted by this scale factor
scale_factor = 0.02099517
y_predicted = y_predicted / scale_factor
y_test = y_test / scale_factor
r_sqaured = r2_score(y_test, y_predicted)
print(r_sqaured)

plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()






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

