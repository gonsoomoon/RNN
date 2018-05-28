#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 09:09:16 2018

@author: gonsoomoon
"""


"""
Function Usage: 
   
   file_path = '../input/daily-minimum-temperatures-in-me.csv'
   series = ui.load_series_data(file_path)   
   
   ui.display_acf(series)
   ui.display_pacf(series, 50)

"""


def display_acf(series, lags = 50):
   from matplotlib import pyplot
   from statsmodels.graphics.tsaplots import plot_acf
   plot_acf(series)
   pyplot.show()

def display_pacf(series, lags = 50):
   from matplotlib import pyplot
   from statsmodels.graphics.tsaplots import plot_pacf
   plot_pacf(series, lags = lags)
   pyplot.show()





"""
Function Usage: 
   
import ts_input  as ui   
   from matplotlib import pyplot
   
   file_path = '../input/daily-minimum-temperatures-in-me.csv'
   
   series = ui.load_series_data(file_path)
   series.plot()
   pyplot.show()

"""


# Load data
import pandas as pd



def load_series_data(file_path):

   series = pd.read_csv(file_path, index_col=0,
                header=0, squeeze=True,parse_dates=True);
   
   return series



"""
Function Usage: 
   
import ts_input  as ui   
from pandas import DataFrame

raw = DataFrame()
raw['obs1'] = [x for x in range(10)]
raw['obs2'] = [x for x in range(50, 60)]
values = raw.values

data = ui.series_to_supervised(data = values, n_in = 1, n_out = 2)
print(data)
   
   
data = series_to_supervised(values, 1, 2)
print(data)
"""
from pandas import DataFrame
from pandas import concat


def series_to_supervised(data, n_in = 1, n_out = 1, dropnan=True):
    """
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... , t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ...t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis = 1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace = True)
    return agg