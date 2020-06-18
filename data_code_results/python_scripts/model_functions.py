# arima fit and forecast
def arima_fit_and_forecast(test, n_lag, n_seq):
    forecasts = []
    for sample in tqdm(test):
        # fit
        arima_model = auto_arima(sample[:n_lag], seasonal=True, trace=False, stepwise=True, suppress_warnings=True,
                                    start_p=1, start_q=1, max_p=n_lag, max_q=n_lag, 
                                    start_P=0, start_Q=0, max_P=n_lag, max_Q=n_lag,
                                    d=0, D=0, error_action="ignore")
        # predictions
        forecasts.append(arima_model.predict(n_periods=n_seq))
    return forecasts

# arima fit and forecast
def arima_boost_data(data, n_lag, n_seq):
    fitted_and_forecasts = []
    for sample in tqdm(data):
        # fit
        arima_model = auto_arima(sample[:n_lag], seasonal=True, trace=False, stepwise=True, suppress_warnings=True,
                                    start_p=1, start_q=1, max_p=n_lag, max_q=n_lag, 
                                    start_P=0, start_Q=0, max_P=n_lag, max_Q=n_lag,
                                    d=0, D=0, error_action="ignore")
        # data
        fitted_and_forecasts.append(np.concatenate((arima_model.predict(n_periods=n_seq), arima_model.predict(n_periods=n_seq)), axis=None))
    return fitted_and_forecasts

# create a differenced series
def difference(dataset, interval=1):
    diff = []
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

# convert time series into supervised learning problem
def series_to_supervised(data, n_in, n_out, dropnan=True):
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
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# transform series into train and test sets for supervised learning
def prepare_data(series, prop_test, n_lag, n_seq):
    # transform data to be stationary
    diff_values = difference(series.values, 1).values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    n_test = int(prop_test * len(supervised_values))
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test, n_test

def prepare_data_mv(feature_df_adj): 
    Xy_train = []
    Xy_test = []
    n_features = feature_df_adj.shape[1]
    col_idx = 0
    for column in tqdm(feature_df_adj):
        scaler, train, test, n_test = prepare_data(feature_df_adj[column], prop_test, n_lag, n_seq)
        if len(Xy_train)==0:
            Xy_train = np.zeros((train.shape[0], n_features, train.shape[1]))
        if len(Xy_test)==0:
            Xy_test = np.zeros((test.shape[0], n_features, test.shape[1]))
        for s in range(len(train)):
            Xy_train[s, col_idx] = train[s]
        for s in range(len(test)):
            Xy_test[s, col_idx] = test[s]
        col_idx += 1
    return scaler, Xy_train, Xy_test

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    # design network
    backend.clear_session()
    model = Sequential()
    model.add(CuDNNLSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(n_seq))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    losses = []
    for i in tqdm(range(n_epochs)):
        history = model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        losses.append(history.history['loss'][0])
        model.reset_states()
    return model, losses

def fit_lstm_mv(train, n_lag, n_seq, n_batch, n_epochs, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, :, 0:n_lag], train[:, -1, n_lag:] 
    X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
    # design network
    backend.clear_session()
    model = Sequential()
    model.add(CuDNNLSTM(n_neurons, batch_input_shape = (n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(n_seq))
    model.compile(optimizer='adam', loss='mse')
    # fit network
    losses = []
    for i in tqdm(range(n_epochs)):
        history = model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        losses.append(history.history['loss'][0])
        model.reset_states()
    return model, losses

# forecast LSTM values
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, len(X), 1)
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]

def forecast_lstm_mv(model, X, n_batch, n_lag, n_seq):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, X.shape[1], X.shape[0])
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = []
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

def make_forecasts_mv(model, n_batch, train, test, n_lag, n_seq):
    forecasts = []
    for i in range(len(test)):
        X, y = test[i, :, 0:n_lag], test[i, -1, n_lag:] 
        # make forecast
        forecast = forecast_lstm_mv(model, X, n_batch, n_lag, n_seq)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = []
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted
 
# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = []
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = np.array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted
 
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    rmses = []
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        rmses.append(rmse)
        print('t+%d RMSE: %f' % ((i+1), rmse))
    return rmses

# plot train loss across stocks
def plot_train_loss(train_loss, sym, n_neurons, n_seq, fs):    
    # train loss
    plt.figure()
    plt.plot(train_loss, 'o-', alpha=.5, label=sym)
    plt.title('{} Training Loss, m={}, u={}'.format(model_name, n_seq, n_neurons), fontsize=fs*1.3)
    plt.xlabel('Training Sample', fontsize=fs)
    plt.ylabel('MSE', fontsize=fs)
    plt.grid()
    plt.legend(fontsize=fs)
    plt.show()
    
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test, sym, n_neurons, n_seq, fs):
    # only plot test data
    series = series[-n_test:]
    # plot the entire dataset in blue
    plt.figure()
    plt.plot(series.values, label='price')
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = i
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        plt.plot(xaxis, yaxis, color='red', alpha=.2)
        if i == 0:
            plt.plot(xaxis, yaxis, color='red', alpha=.2, label='forecasts')
    plt.title('{} Prices with Forecasts, m={}, u={}'.format(sym, n_seq, n_neurons), fontsize=fs*1.3)
    plt.xlabel('Day', fontsize=fs)
    plt.ylabel('Price', fontsize=fs)
    plt.grid()
    plt.legend(fontsize=fs)  
    # show the plot
    plt.show()

# plot the forecasts in the context of the original dataset
def plot_forecasts_full(series, forecasts, n_test):
    # plot the entire dataset in blue
    plt.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        plt.plot(xaxis, yaxis, color='red', alpha=.2)
    # show the plot
    plt.show()

# plot the arima forecasts in the context of the original dataset in grid
def plot_arima_forecasts_grid(series, forecasts, n_seq, n_test, axarr, j, fs):
    # only plot test data
    series = series[-n_test:]
    # plot the entire dataset in blue
    axarr[j].plot(series.values, label='price')
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = i
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        axarr[j].plot(xaxis, yaxis, color='red', alpha=.2)
        if i == 0:
            axarr[j].plot(xaxis, yaxis, color='red', alpha=.2, label='forecasts')
    axarr[j].set_title('{} Prices with Forecasts, m={}'.format(sym, n_seq), fontsize=fs*1.3)
    axarr[j].set_xlabel('Day', fontsize=fs)
    axarr[j].set_ylabel('Price', fontsize=fs)
    axarr[j].grid()
    axarr[j].legend(fontsize=fs)

# plot the forecasts in the context of the original dataset in grid
def plot_forecasts_grid(series, forecasts, n_seq, n_test, axarr, j, k, fs):
    # only plot test data
    series = series[-n_test:]
    # plot the entire dataset in blue
    axarr[j,k].plot(series.values, label='price')
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = i
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        axarr[j,k].plot(xaxis, yaxis, color='red', alpha=.2)
        if i == 0:
            axarr[j,k].plot(xaxis, yaxis, color='red', alpha=.2, label='forecasts')
    axarr[j,k].set_title('{} Prices with Forecasts, m={}, u={}'.format(sym, n_seq, n_neurons), fontsize=fs*1.3)
    axarr[j,k].set_xlabel('Day', fontsize=fs)
    axarr[j,k].set_ylabel('Price', fontsize=fs)
    axarr[j,k].grid()
    axarr[j,k].legend(fontsize=fs)

# generate features
def generate_inputs(sym, name):
    # get stock data
    if sym == 'TSLA':
        hdata = pd.read_csv('stocks_data/' + name + '.csv')
    else:
        hdata = pd.read_sql("SELECT * FROM fact_table WHERE symbol='{}' AND date BETWEEN '{}' AND '{}'".format(sym, startdate, enddate), con=conn)
    
    # get time series
    ys = hdata['adj_close']
    hdata.drop(labels=['symbol'], axis=1, inplace=True)

    # prepare target data
    series = ys
    scaler, train, test, n_test = prepare_data(series, prop_test, n_lag, n_seq)

    return series, scaler, train, test, n_test

# generate features
def generate_inputs_mv(sym, name, gramtype):
    # get stock data
    if sym == 'TSLA':
        hdata = pd.read_csv('stocks_data/' + name + '.csv')
    else:
        hdata = pd.read_sql("SELECT * FROM fact_table WHERE symbol='{}' AND date BETWEEN '{}' AND '{}'".format(sym, startdate, enddate), con=conn)
    # get time series
    ys = hdata['adj_close']
    hdata.drop(labels=['symbol'], axis=1, inplace=True)

    # prepare target data
    series = ys
    scaler, train, test, n_test = prepare_data(series, prop_test, n_lag, n_seq)

    # modify for dictionary use
    hdata.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'ADJ_CLOSE', 'VOLUME']
    hdata.drop(columns=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'], inplace=True)
    
    # tokenize and store, else load from memory
    try: 
        feature_df = load_multivariate(name, gramtype)
    except:
        feature_df_onegram, feature_df_bigram, feature_df_trigram = generate_df(name, hdata)
        store_gram(feature_df_onegram, name, 'onegram')
        store_gram(feature_df_bigram, name, 'bigram')
        store_gram(feature_df_trigram, name, 'trigram')

    # prepare multivariate target data
    feature_df = load_multivariate(name, gramtype)
    feature_df_adj = feature_df.drop(columns=['DATE'])
    ys = feature_df_adj.pop('ADJ_CLOSE')
    feature_df_adj['ADJ_CLOSE'] = ys
    feature_df_adj.astype('float')
    scaler_mult, train_mult, test_mult = prepare_data_mv(feature_df_adj)

    return series, n_test, scaler_mult, train_mult, test_mult