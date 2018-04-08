#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 20:54:07 2018

@author: gonsoomoon
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re

DATA_DIR = "."

# Open the file
fld = open(os.path.join(DATA_DIR, "LD2011_2014.txt"), encoding='utf-8')
elec_con = [] # to record elec. spending
date_cols = [] # to record date and time

line_num = 0
#cid = np.random.randint(0, 370, 1)
cid = 250 # to select a user named 250
for line in fld:
    # Skip the first line to refer to the users
    if line.startswith("\"\";"):
        continue
    # print a progress
    if line_num % 10000 == 0:
        print("{:d} lines read".format(line_num))
    # Except for the first column, date, parse with ";" and replect "," with "."
    cols = [float(re.sub(",", ".", x)) for x in 
            line.strip().split(";")[1:]]
    # add electricity spending
    elec_con.append(cols[cid])
    
    # extract the first column
    date = line.strip().split(";")[0]
    # Remove " on both sides
    date = date.strip('"')
    date_cols.append(date)
        
    line_num += 1
fld.close()


import pandas as pd
data_with_date = {'date':date_cols, 'elec_con':elec_con}
df_data_with_date = pd.DataFrame(data = data_with_date)
    
#print("df_data_with_date: \n", df_data_with_date)

# Save date and spending as a csv file
df_data_with_date.to_csv('data_with_date_250.csv')
# save spending as a np file
np.save(os.path.join(DATA_DIR, "LD_250.npy"), np.array(elec_con))

NUM_ENTRIES = 1000
plt.plot(range(NUM_ENTRIES), elec_con[0:NUM_ENTRIES])
plt.ylabel("electricity consumption")
plt.xlabel("time (1pt = 15 mins)")
plt.show()


