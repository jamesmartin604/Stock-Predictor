import os
import json
import pandas as pd
import datetime as dt
import urllib.request
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data_source = 'kaggle'

if data_source == 'alphavantage':
    api_key = '<your API key>'  # Replace this with your API key
    ticker = "AAL"  # American Airlines stock ticker

    # Alpha Vantage API URL for daily time series data
    url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"

    # File to save the data
    file_to_save = f'stock_market_data-{ticker}.csv'

    if not os.path.exists(file_to_save):
        print('Fetching data from Alpha Vantage...')
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # Extract stock market data
            if "Time Series (Daily)" in data:
                data = data['Time Series (Daily)']
                df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
                for k, v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date(), float(v['3. Low']), float(v['2. High']),
                                float(v['4. close']), float(v['1. open'])]
                    df.loc[-1, :] = data_row
                    df.index = df.index + 1  # Shift index
                print(f'Data saved to: {file_to_save}')
                df.to_csv(file_to_save, index=False)
            else:
                print("Error: Failed to fetch data. Check your API key or API limit.")
    else:
        print('File already exists. Loading data from CSV...')
        df = pd.read_csv(file_to_save)
    print(df.head())

else:
    print('Loading data from Kaggle...')
    kaggle_file = os.path.join('Stocks', 'hpq.us.txt')  # Example file
    df = pd.read_csv(kaggle_file, delimiter=',', usecols=['Date','Open','High','Low','Close'])
    print('Data loaded successfully.')
    print(df.head())

#plt.figure(figsize= (18,9))
#plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
#plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
#plt.xlabel('Date',fontsize=18)
#plt.ylabel('Mid Price',fontsize=18)
#plt.show()

#calcuation of mid price from highest and lowest

high_prices = df.loc[:,'High'].to_numpy()
low_prices = df.loc[:,'Low'].to_numpy()
mid_prices = (high_prices+low_prices)/2.0

train_data = mid_prices[:11000]
test_data = mid_prices[11000:]

scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

#train the scaler 
smoothing_window_size = 2500
for di in range(0,10000,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

#normalise last bit of data
scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

#reshape both train and test data
train_data = train_data.reshape(-1)

#normalise test data
test_data = scaler.transform(test_data).reshape(-1)

#exponential moving average smoothing, so the data will have smoother curve than the original rugged data
EMA = 0.0
gamma = 0.1
for ti in range(11000):
    EMA = gamma*train_data[ti] + (1-gamma)*EMA
    train_data[ti] = EMA

#visualisation and testing purpose
all_mid_data = np.concatenate([train_data,test_data],axis=0)

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):
    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days = 1)
    else:
        date = df.loc[pred_idx,'Date']
    
    std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
    mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
    std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))



window_size = 100
N = train_data.size

run_avg_predictions = []
run_avg_x = []

mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):
    running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
    run_avg_predictions.append(running_mean)
    mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
    run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()
