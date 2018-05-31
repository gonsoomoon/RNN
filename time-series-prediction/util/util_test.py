#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:04:36 2018

@author: gonsoomoon
"""
import ts_input  as ui



"""
Test: ts_input.series_to_supervised()
"""
def test_series_to_supervised():
   from pandas import DataFrame
   
   
   raw = DataFrame()
   raw['obs1'] = [x for x in range(10)]
   raw['obs2'] = [x for x in range(50, 60)]
   values = raw.values
   
   
   data = ui.series_to_supervised(data = values, n_in = 1, n_out = 2)
   print(data)

"""
Test: ts_input.load_series_data()
"""
def test_load_series_data():
   from matplotlib import pyplot
   
   file_path = '../input/daily-minimum-temperatures-in-me.csv'
   
   series = ui.load_series_data(file_path)
   series.plot()
   pyplot.show()
   
"""
Test: ts_input.test_display_acf()
"""
def test_display_acf():
   file_path = '../input/daily-minimum-temperatures-in-me.csv'
   series = ui.load_series_data(file_path)   
   
   ui.display_acf(series)
   ui.display_pacf(series, 50)

# test_load_series_data()   

test_display_acf()

