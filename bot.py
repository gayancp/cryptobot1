import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import requests
from dateutil.relativedelta import relativedelta
import numpy as np
import datetime
import time
from tensorflow.keras.layers import concatenate
import talib
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

# Step 1: Fetch past price movement data and calculate technical indicators
# Fetch past price movement data from Binance API

# Define Binance API endpoints and parameters
base_url = 'https://api.binance.com'
klines_endpoint = '/api/v3/klines'
symbol = 'BTCUSDT'
intervals = ['3m','5m', '15m', '1h', '2h', '4h']#144 48 12 6 2
                            # Start time in milliseconds (e.g., Jan 1, 2021)
                            # End time in milliseconds (e.g., May 31, 2021)


# GLOBAL variables
price_movements = []
df = None
normalized_df = None
df8hour = pd.DataFrame()
combined_model = None

# Define technical indicators calculation functions
def calculate_sma(df, interval, timestamp, period):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < period:
        return None
    return interval_df['price'].tail(period).mean()

def calculate_rsi(df, interval, timestamp, period):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < period + 1:
        return None
    price_diff = interval_df['price'].diff().dropna()
    gain = price_diff.apply(lambda x: x if x > 0 else 0).tail(period)
    loss = -price_diff.apply(lambda x: x if x < 0 else 0).tail(period)
    avg_gain = gain.mean()
    avg_loss = loss.mean()
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, interval, timestamp):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < 26:
        return None
    close_prices = interval_df['price'].tail(26)
    ema_12 = close_prices.ewm(span=12, adjust=False).mean()
    ema_26 = close_prices.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-1] - signal_line.iloc[-1]

def calculate_bollinger_bands(df, interval, timestamp, period):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < period:
        return None
    prices = interval_df['price'].tail(period)
    middle_band = prices.mean()
    std_dev = prices.std()
    upper_band = middle_band + 2 * std_dev
    lower_band = middle_band - 2 * std_dev
    return upper_band, middle_band, lower_band

def calculate_atr(df, interval, timestamp, period):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < period + 1:
        return None
    high_prices = interval_df['high'].tail(period + 1)
    low_prices = interval_df['low'].tail(period + 1)
    close_prices = interval_df['close'].tail(period + 1)
    atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period).iloc[-1]
    return atr

def calculate_stochastic_oscillator(df, interval, timestamp, period):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < period + 3:
        return None
    high_prices = interval_df['high'].tail(period + 3)
    low_prices = interval_df['low'].tail(period + 3)
    close_prices = interval_df['close'].tail(period + 3)
    slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=period, slowk_period=3, slowd_period=3)
    return slowk.iloc[-1], slowd.iloc[-1]

def calculate_macd_histogram(df, interval, timestamp):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < 26:
        return None
    close_prices = interval_df['close'].tail(26)
    macd, macdsignal, macdhist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    return macdhist.iloc[-1]

def calculate_on_balance_volume(df, interval, timestamp):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < 1:
        return None
    close_prices = interval_df['close'].tail(1)
    volume = interval_df['volume'].tail(1)
    obv = talib.OBV(close_prices, volume)
    return obv.iloc[-1]

def calculate_ichimoku_cloud(df, interval, timestamp):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < 52:
        return None
    high_prices = interval_df['high'].tail(52)
    low_prices = interval_df['low'].tail(52)
    tenkan_sen, kijun_sen, _, _, senkou_span_a, senkou_span_b = talib.ICHIMOKU(high_prices, low_prices)
    return tenkan_sen.iloc[-1], kijun_sen.iloc[-1], senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]

def calculate_chaikin_money_flow(df, interval, timestamp, period):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < period:
        return None
    high_prices = interval_df['high'].tail(period)
    low_prices = interval_df['low'].tail(period)
    close_prices = interval_df['close'].tail(period)
    volume = interval_df['volume'].tail(period)
    cmf = talib.CMF(high_prices, low_prices, close_prices, volume, timeperiod=period)
    return cmf.iloc[-1]

def calculate_williams_percent_r(df, interval, timestamp, period):
    interval_df = df[(df['interval'] == interval) & (df['timestamp'] < timestamp)]
    if len(interval_df) < period + 1:
        return None
    high_prices = interval_df['high'].tail(period + 1)
    low_prices = interval_df['low'].tail(period + 1)
    close_prices = interval_df['close'].tail(period + 1)
    willr = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=period)
    return willr.iloc[-1]

#get input data for 1 week
def get_input(week_start, week_end):
    try:
        # Fetch price movement data for each interval
        for interval in intervals:
            params = {'symbol': symbol, 'interval': interval, 'startTime': week_start, 'endTime': week_end}
            response = requests.get(base_url + klines_endpoint, params=params)
            response.raise_for_status()

            # Parse response data
            data = response.json()

            # Process price movement data
            for item in data:
                timestamp = int(item[0])
                price = float(item[4])

                # Store price movements
                price_movements.append({'timestamp': timestamp, 'price': price, 'interval': interval})
            time.sleep(5)

        global df
        # Convert data to pandas DataFrame
        df = pd.DataFrame({'timestamp': [item['timestamp'] for item in price_movements],
                        'price': [item['price'] for item in price_movements],
                        'interval': [item['interval'] for item in price_movements]})

    except requests.exceptions.RequestException as e:
        print("Error fetching price movement data:", e)

    except Exception as e:
        print("An error occurred:", e)

def norm_and_preprocess():
    global df
    # Normalize and preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(df.values)

    # Create a new DataFrame with the normalized data
    global normalized_df
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

def calculate_technical_indicators():
    sma_values = []
    rsi_values = []
    macd_values = []
    upper_bands = []
    middle_bands = []
    lower_bands = []
    atr_values = []
    slowks = []
    slowds = []
    macd_hist_values = []
    obv_values = []
    tenkan_sens = []
    kijun_sens = []
    senkou_span_as = []
    senkou_span_bs = []
    cmf_values = []
    willr_values = []

    try:    
        for index, row in df8hour.iterrows():
            interval = row['interval']
            timestamp = row['timestamp']
            price = row['price']

            match(interval):
                case '3m':
                    period = 160
                case '5m':
                    period = 96
                case '15m':
                    period = 32
                case '1h':
                    period = 8
                case '2h':
                    period = 4
                case '4h':
                    period = 2

            # Calculate SMA
            sma_value = calculate_sma(df8hour, interval, timestamp, period)
            sma_values.append(sma_value)

            # Calculate RSI
            rsi_value = calculate_rsi(price, interval, timestamp, period)
            rsi_values.append(rsi_value)

            # Calculate MACD
            macd_value = calculate_macd(price, interval, timestamp)
            macd_values.append(macd_value)

            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = calculate_bollinger_bands(price, interval, timestamp, period)
            upper_bands.append(upper_band)
            middle_bands.append(middle_band)
            lower_bands.append(lower_band)

            # Calculate Average True Range
            atr_value = calculate_atr(price, interval, timestamp, period)
            atr_values.append(atr_value)

            # Calculate Stochastic Oscillator
            slowk, slowd = calculate_stochastic_oscillator(price, interval, timestamp, period)
            slowks.append(slowk)
            slowds.append(slowd)

            # Calculate Moving Average Convergence Divergence Histogram
            macd_hist_value = calculate_macd_histogram(price, interval, timestamp)
            macd_hist_values.append(macd_hist_value)

            # Calculate On-Balance Volume
            obv_value = calculate_on_balance_volume(price, interval, timestamp)
            obv_values.append(obv_value)

            # Calculate Ichimoku Cloud
            tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b = calculate_ichimoku_cloud(price, interval, timestamp)
            tenkan_sens.append(tenkan_sen)
            kijun_sens.append(kijun_sen)
            senkou_span_as.append(senkou_span_a)
            senkou_span_bs.append(senkou_span_b)

            # Calculate Chaikin Money Flow
            cmf_value = calculate_chaikin_money_flow(price, interval, timestamp, period)
            cmf_values.append(cmf_value)

            # Calculate Williams %R
            willr_value = calculate_williams_percent_r(price, interval, timestamp, period)
            willr_values.append(willr_value)

    except ValueError:
        print("Error! Interval out of range")
    
    bollinger_bands = [upper_bands, middle_bands, lower_bands]
    stochastic_oscillator = [slowks, slowds]
    Ichimoku_Cloud = [tenkan_sens, kijun_sens, senkou_span_as, senkou_span_bs]


    global df8hour
    df8hour['sma'] = sma_values
    df8hour["rsi"] = rsi_values
    df8hour["macd"] = macd_values
    df8hour["bollinger_bands"] = bollinger_bands
    df8hour["average_true_range"] = atr_values
    df8hour["stochastic_oscillator"] = stochastic_oscillator
    df8hour["Moving_Average_Convergence_Divergenc_Histogram"] = macd_hist_values
    df8hour["On_Balance_Volume"] = obv_values
    df8hour["Ichimoku_Cloud"] = Ichimoku_Cloud
    df8hour["Chaikin_Money_Flow"] = cmf_values
    df8hour["Williams_%R"] = willr_values

def define_models():
    n_features = 14
    n_timesteps = 302

    # Step 3: Combine the CNN and LSTM models
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=64, return_sequences=True, input_shape=(n_timesteps, n_features)))
    lstm_model.add(LSTM(units=64))

    global combined_model
    combined_model = Sequential()
    combined_model.add(concatenate([cnn_model, lstm_model]))
    combined_model.add(Dense(units=64, activation='relu'))
    combined_model.add(Dense(units=1, activation='sigmoid'))

    # Step 4: Compile the combined model
    combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#calculate RMSE - Prediction progress check
def calculate_rmse(predicted_values, actual_values):
    squared_diff = np.square(predicted_values - actual_values)
    mean_squared_diff = np.mean(squared_diff)
    rmse = np.sqrt(mean_squared_diff)
    return rmse


#running part------------------------------------------------------------------------------------------------------------

def training_process():
    #for 1 week data
    week_start = int(datetime.datetime(2021, 1, 1, 0, 0, 0).timestamp())
    week_end = int(datetime.datetime(2021, 1, 7, 23, 59, 59).timestamp())

    get_input(week_start, week_end)
    norm_and_preprocess()

    # Convert the 'timestamp' column to datetime type
    normalized_df['timestamp'] = pd.to_datetime(normalized_df['timestamp'], unit='ms')

    # Sort the DataFrame by timestamp in ascending order
    normalized_df.sort_values('timestamp', inplace=True)

    # Set the 'timestamp' column as the index
    normalized_df.set_index('timestamp', inplace=True)

    #for 8 hour data
    start_time = pd.to_datetime('2021-01-01 00:00:00')
    end_time = pd.to_datetime('2021-01-01 7:59:59')

    # Step 5: Define the training loop
    for i in range(20):
        while True:        
            #get 8 hour time
            global df8hour
            df8hour = normalized_df.loc[(normalized_df['timestamp'] >= start_time) & (normalized_df['timestamp'] <= end_time)].copy()

            #technical indicator calculation
            calculate_technical_indicators()

            input_data = df8hour.values

            #get price of within 2 hour time to the future
            pricelabel = np.array(normalized_df.loc[normalized_df['timestamp'] == (end_time + pd.Timedelta(hours=2)), 'price'].values[0])

            # Step 7: Train the model
            checkpoint = ModelCheckpoint(filepath='model_checkpoint.h5', monitor='val_loss', save_best_only=True)
            combined_model.fit(input_data, pricelabel, epochs=10, batch_size=32, callbacks=[checkpoint])

            # Step 8: Save the trained model
            combined_model.save('trained_model.h5')

            #time adding by 1 hour
            start_time += pd.Timedelta(minutes=1)
            end_time += pd.Timedelta(minutes=1)

            #check if the data present in the dataframe
            if not normalized_df['timestamp'].isin([end_time]).any():
                print("This datetime data currently not present in this DataFrame! Fetching Next DataFrame...")
                break
            elif not normalized_df['timestamp'].isin([start_time]).any():
                print("This datetime data currently not present in this DataFrame! Fetching Next DataFrame...")
                break

            # Calculate RMSE
            predictions = combined_model.predict(input_data)
            rmse = calculate_rmse(predictions, pricelabel)
            print("RMSE:(Accuracy Level): ", rmse)

            time.sleep(2)

        #next week data
        week_start += relativedelta(weeks=1)
        week_end += relativedelta(weeks=1)

        #get data for next week months
        get_input(week_start, week_end)

        #normalize and preprocess
        norm_and_preprocess()


define_models()
training_process()
