import pandas as pd
from django.conf import settings
import joblib
import numpy as np
import datetime
import xgboost as xgb
import cryptocompare

scaler = joblib.load("bitcoin_app\\scaler.save")

def load_data(coin, currency, limit):
    data = cryptocompare.get_historical_price_day(coin, currency, limit=limit)
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return pd.DataFrame({"date": df['time'], "close": df["close"]})

def predict(model, data, future_days, time_step):
    '''
        Input: data: numpy array
        Output: price_predictions: numpy array of shape (future_days, 1)
    '''
    price_predictions = []
    temp_prices = data[-time_step:]
    for i in range(future_days):
        if type(model) == xgb.sklearn.XGBRegressor:
            price_prediction = model.predict(temp_prices.reshape(1, -1))[0]
        else: 
            price_prediction = model.predict(temp_prices.reshape(1, -1, 1))[0][0]
        price_predictions.append(price_prediction)
        temp_prices = np.append(temp_prices, price_prediction)
        temp_prices = temp_prices[1:]
    price_predictions = np.array(price_predictions).reshape(-1, 1)
    return price_predictions

def preprocess(data):
    return scaler.transform(np.array(data).reshape(-1, 1)).reshape(1, -1)[0]

def postprocess(data):
    return scaler.inverse_transform(data).reshape(1, -1)[0]

def get_future_closes(model, data, future_days, time_step):
    today = data["date"].to_list()[-1]
    future_dates = [today + datetime.timedelta(days = i) for i in range(1, future_days + 1)]
    dates =  data["date"].to_list() + future_dates
    close_price = np.empty(len(dates))
    close_price[:] = np.nan
    close_price[:len(data["close"])] = data["close"]
    predicted_close = np.empty(len(dates))
    predicted_close[:] = np.nan
    price_predictions = predict(model, preprocess(data["close"]), future_days, time_step)
    price_predictions = postprocess(price_predictions)
    predicted_close[-future_days:] = price_predictions
    predicted_close[-future_days - 1] = close_price[-future_days - 1]
    return pd.DataFrame({"date": dates, "close": close_price, "predicted_close": predicted_close})