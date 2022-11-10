from datetime import datetime
import MetaTrader5 as mt5
from datetime import timedelta
import pandas_ta as ta
# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt5.__author__)
print("MetaTrader5 package version: ",mt5.__version__)
 
# import the 'pandas' module for displaying data obtained in the tabular form
import pandas as pd
pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display
# import pytz module for working with time zone
import pytz
 
# establish connection to MetaTrader 5 terminal

 
def scrap():
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    # set time zone to UTC
    timezone = pytz.timezone("Etc/UTC")
    # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
    time = datetime(2022, 7, 1, 0, 0, 0,tzinfo=timezone)# format (yyyy, mm, dd, hh, mm, ss)
    # time = datetime.now(tz=timezone)
    print(time)
    # get 10 EURUSD H4 bars starting from 01.10.2020 in UTC time zone
    rates = mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_M5, time, 1060)
    
    # shut down connection to the MetaTrader 5 terminal
    mt5.shutdown()
    # display each element of obtained data in a new line
    # print("Display obtained data 'as is'")
    # for rate in rates:
    #     print(rate)
    
    # create DataFrame out of the obtained data
    rates_frame = pd.DataFrame(rates)
    # convert time in seconds into the datetime format
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame["open"] = pd.to_numeric(rates_frame["open"], downcast="float")
    rates_frame["high"] = pd.to_numeric(rates_frame["high"], downcast="float")
    rates_frame["low"] = pd.to_numeric(rates_frame["low"], downcast="float")
    rates_frame["close"] = pd.to_numeric(rates_frame["close"], downcast="float")
    rates_frame["tick_volume"] = pd.to_numeric(rates_frame["tick_volume"], downcast="float")

    sma5 = ta.sma(rates_frame['close'], length=5)
    rsi = ta.rsi(rates_frame['close'].astype(float), length = 10)
    macd = ta.macd(rates_frame['close'].astype(float), length = 10)
    ichimoku = ta.ichimoku(rates_frame['high'],rates_frame['high'],rates_frame['high'])
    ichimoku = ichimoku[0]

    rates_frame['ma5']=sma5
    rates_frame['rsi']=rsi
    rates_frame['macd_l'] = macd['MACD_12_26_9']
    rates_frame['macd_s'] = macd['MACDs_12_26_9']
    rates_frame['tenkan_sen'] = ichimoku['ITS_9']
    rates_frame['kijun_sen'] = ichimoku['IKS_26']
    rates_frame['chikou_span'] = ichimoku['ICS_26']
    rates_frame['tenkan_sen'] = ichimoku['ITS_9']
    
    return rates_frame

def scrap_realtime():
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    # set time zone to UTC
    timezone = pytz.timezone("Etc/UTC")
    # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
    # time = datetime(2022, 7, 1, 0, 0, 0,tzinfo=timezone)# format (yyyy, mm, dd, hh, mm, ss)
    time = datetime.now(tz=timezone)
    print(time)
    # get 10 EURUSD H4 bars starting from 01.10.2020 in UTC time zone
    rates = mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_M5, time, 1060)
    
    # shut down connection to the MetaTrader 5 terminal
    mt5.shutdown()
    # display each element of obtained data in a new line
    # print("Display obtained data 'as is'")
    # for rate in rates:
    #     print(rate)
    
    # create DataFrame out of the obtained data
    rates_frame = pd.DataFrame(rates)
    # convert time in seconds into the datetime format
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame["open"] = pd.to_numeric(rates_frame["open"], downcast="float")
    rates_frame["high"] = pd.to_numeric(rates_frame["high"], downcast="float")
    rates_frame["low"] = pd.to_numeric(rates_frame["low"], downcast="float")
    rates_frame["close"] = pd.to_numeric(rates_frame["close"], downcast="float")
    rates_frame["tick_volume"] = pd.to_numeric(rates_frame["tick_volume"], downcast="float")

    sma5 = ta.sma(rates_frame['close'], length=5)
    rsi = ta.rsi(rates_frame['close'].astype(float), length = 10)
    macd = ta.macd(rates_frame['close'].astype(float), length = 10)
    ichimoku = ta.ichimoku(rates_frame['high'],rates_frame['high'],rates_frame['high'])
    ichimoku = ichimoku[0]

    rates_frame['ma5']=sma5
    rates_frame['rsi']=rsi
    rates_frame['macd_l'] = macd['MACD_12_26_9']
    rates_frame['macd_s'] = macd['MACDs_12_26_9']
    rates_frame['tenkan_sen'] = ichimoku['ITS_9']
    rates_frame['kijun_sen'] = ichimoku['IKS_26']
    rates_frame['chikou_span'] = ichimoku['ICS_26']
    rates_frame['tenkan_sen'] = ichimoku['ITS_9']
    
    return rates_frame