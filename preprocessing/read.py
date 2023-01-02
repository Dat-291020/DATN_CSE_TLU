import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def load_csv(data_dir, features = []):
    df = pd.read_csv(data_dir)
    if len(features) > 0:
        df = df[features]
    return df

def split(df, features, scaler, valid_index, test_index, end_index, time_lag=1,freq=None):
    X = df[features]
    X[features] = scaler.transform(X)
    train = X.copy()[X.index < valid_index]
    train = window_generate(train, features, time_lag=time_lag, freq=freq)
    valid = X.copy()[(X.index < test_index) & (X.index >= valid_index)]
    valid = window_generate(valid, features, time_lag=time_lag, freq=freq)
    test = X.copy()[(X.index < end_index) & (X.index >= test_index)]
    test = window_generate(test, features, time_lag=time_lag, freq=freq)
    return train, valid, test


def split_y(df, targets, scaler, valid_index, test_index, end_index, horizon=1, freq=None):
    X = df[targets]
    X[targets] = scaler.transform(X)
    m = targets
    for i in range(len(targets)):
        m[i]=targets[i]+"+"+str(horizon)
    X=X.set_axis(m, axis='columns')
    train = X.copy()[X.index < valid_index]
    train = train.shift(periods=-horizon, freq=freq)
    valid = X.copy()[(X.index < test_index) & (X.index >= valid_index)]
    valid = valid.shift(periods=-horizon, freq=freq)
    test = X.copy()[(X.index < end_index) & (X.index >= test_index)]
    test = test.shift(periods=-horizon, freq=freq)
    return train, valid, test


def window_generate(df, features, time_lag, freq=None):
    data = df.copy()[features]
    result = pd.DataFrame()
    for i in range(-time_lag+1, 1):
        columns = []
        data_i = data.shift(periods=-i, freq=freq)
        for j in range(len(features)):
            columnsj = data.columns[j] + '_%d'%(i)
            columns.append(columnsj)
        data_i = data_i.set_axis(columns, axis=1, inplace=False)
        result = pd.concat([result, data_i], axis = 1)
    return result

def power_set(seq):
    if len(seq)<=1:
        yield seq
        yield []
    else:
        for item in power_set(seq[1:]):
            yield [seq[0]]+item
            yield item
