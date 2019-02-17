
import seaborn;seaborn.set()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import datetime as dt

def displayPowerUsage2(x, y, title):
    dates = x
    dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(dates, y[y.columns[0]], label='Actual Usage')
    ax.plot(dates, y[y.columns[1]], label='Forcast Usage')

    mondays = mdates.WeekdayLocator(mdates.SUNDAY)  # major ticks on the mondays
    alldays = mdates.DayLocator()  # minor ticks on the days
    weekFormatter = mdates.DateFormatter('%b %d')  # e.g., Jan 12
    dayFormatter = mdates.DateFormatter('%d')  # e.g., 12

#    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)

    # months = mdates.MonthLocator()  # every month
    # monthsFmt = mdates.DateFormatter('%m')

    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_ylabel('kwh (Energy)')
    ax.set_xlabel('Week')
    # format the ticks
    # ax.xaxis.set_major_locator(months)
    # ax.xaxis.set_major_formatter(monthsFmt)
    plt.show()


import pandas as pd

def read_input():
    data = {'device_id':['a1','a1','a1','a1'],
            'time':['2018-08-02', '2018-08-04', '2018-08-05','2018-08-20'],
            'power_2018':[49166.0,None , 49166.0, 49166.0],
            'cl_mean':[61309.086961,61274.944563,58413.492854,54850.790214  ]}

    df = pd.DataFrame(data=data)

    df['new_power_2018.08'] = round(df[~df.power_2018.isna()]['power_2018'][0] * 0.001, 2)
    df['new_cl_mean'] = round(df['cl_mean'] * 0.001, 2)

    return df

df_pre = read_input()

x = df['time']
y = pd.DataFrame([df_pre['new_power_2018.08'], df_pre['new_cl_mean']]).T

displayPowerUsage2(x,y,'test' )

