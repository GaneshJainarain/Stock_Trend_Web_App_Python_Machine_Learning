import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data 


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
plt.plot(df.Close)
plt.show()