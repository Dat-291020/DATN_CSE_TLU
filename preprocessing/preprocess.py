# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 20:24:16 2022

@author: ADMIN
"""
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import datetime as dt
from preprocessing.read import load_csv

datadir = 'data/TaTrach.csv'

features = []
df = load_csv(datadir, features = features)

d = pd.to_datetime(df['date'], format='%m/%d/%Y')
df = df.set_index(d)
df = df.drop('date', axis=1)

df1 = data.interpolate()
data['TaTrach'] = df1[0]

df.to_csv('data/tatrach_preprocessed.csv')

start_date = dt.date(2016, 1, 1)
end_date = dt.date(2022, 1, 1)

date = pd.date_range(start=start_date, end=end_date, freq='D')
# date = date.format(formatter=lambda x: x.strftime('%d/%m/%Y'))

m = np.empty(2193)
m[:] = np.NaN
data = pd.DataFrame(m)
data['date'] = date
data = data.set_index(data['date'])
data = data.drop('date', axis=1)

for i in range(len(df)):
    l = df.copy().loc[i]['date'].split(sep='/')
    index = pd.to_datetime(dt.date(int(l[2]), int(l[1]), int(l[0])))
    data.loc[index][0] = df.loc[i]['in lake']
         
    
data.to_csv('data/huana_preprocessed.csv')
