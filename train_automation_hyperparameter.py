from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
from numpy import concatenate

from modul.scrap import scrap
from modul.preprocessing import series_to_supervised
from modul.preprocessing import append_list_as_row
from modul.preprocessing import mape

from sklearn.metrics import mean_squared_error
from math import sqrt

import pandas as pd

currency = "EURUSD"
interval = "5m"
len_data = "1000"

df = scrap()
df = df[['time','close','open','high','low','ma5','rsi','macd_l','macd_s','tenkan_sen','kijun_sen','chikou_span' ]]

epoch = [25,50, 75,100,125]
data_train = [500,600,700,800,900]
batch_size = [8, 16, 32, 64,128]
neuron = [25,50, 75, 100]

for epoch_for in epoch:
    for batch_size_for in batch_size:
        for neuron_for in neuron:
            for data_train_for in data_train:
                print("epoch " + str(epoch_for))
                print("batch_size "+ str(batch_size_for))
                print("neuron "+ str(neuron_for))
                print("data_train " + str(data_train_for))
                dataset = df
                dataset = dataset.set_index('time')

                values = dataset.values
                values = values.astype('float32')
                # normalize features
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(values)

                # frame as supervised learning
                reframed = series_to_supervised(scaled, 1, 1)
                
                reframed = reframed.iloc[:, 0:(len(dataset.columns)+1)]

                # split into train and test sets
                values = reframed.values
                n_train_hours = data_train_for
                train = values[:n_train_hours, :]
                test = values[n_train_hours:, :]
                # split into input and outputs
                train_X, train_y = train[:, :-1], train[:, -1]
                test_X, test_y = test[:, :-1], test[:, -1]
                train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
                test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

                print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

                model = Sequential()
                model.add(LSTM(neuron_for, input_shape=(train_X.shape[1], train_X.shape[2])))
                model.add(Dense(1))
                model.compile(loss='mae', optimizer='adam')

                history = model.fit(train_X, train_y, epochs=epoch_for, batch_size= batch_size_for, validation_data=(test_X, test_y), verbose=1, shuffle=False)
                # pyplot.plot(history.history['loss'], label='train')
                # pyplot.plot(history.history['val_loss'], label='test')
                # pyplot.legend()
                # pyplot.show()

                # make a prediction
                yhat = model.predict(test_X)
                test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
                    # invert scaling for forecast
                inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
                inv_yhat = scaler.inverse_transform(inv_yhat)
                inv_yhat = inv_yhat[:,0]
                    # invert scaling for actual
                test_y = test_y.reshape((len(test_y), 1))
                inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
                inv_y = scaler.inverse_transform(inv_y)
                inv_y = inv_y[:,0]

                rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
                mape_result = mape(inv_y, inv_yhat) 
                print('Test RMSE: %.3f' % rmse)
                
                row_contents = [epoch_for,batch_size_for,neuron_for,data_train_for,rmse,mape_result]
                # Append a list as new line to an old csv file
                append_list_as_row('result/hyperparameter.csv', row_contents)


