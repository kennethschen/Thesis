# store model
def store_keras_model(model, model_name, name, n_epochs, n_neurons, n_lag, n_seq):
    full_name = '{}_{}_{}_{}_{}_{}'.format(model_name, name, n_epochs, n_neurons, n_lag, n_seq)
    # serialize model to JSON
    model_json = model.to_json() 
    with open('models/' + full_name, 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('weights/' + full_name + '.h5') 

# load model 
def load_keras_model(model_name, name, n_epochs, n_neurons, n_lag, n_seq):
    full_name = '{}_{}_{}_{}_{}_{}'.format(model_name, name, n_epochs, n_neurons, n_lag, n_seq)
    # load json and create model
    json_file = open('models/' + full_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('weights/' + full_name + '.h5')
    return model

# store ARIMA forecasts
def store_arima_forecasts(forecasts, model_name, name, n_epochs, n_neurons, n_lag, n_seq):
    f = open('arima_forecasts/{}_{}_{}_{}_{}_{}'.format(model_name, name, n_epochs, n_neurons, n_lag, n_seq), 'wb')
    pickle.dump(forecasts, f)
    f.close()

# store train rmses
def store_train_rmses(rmses, model_name, name, n_epochs, n_neurons, n_lag, n_seq):
    f = open('train_rmses/{}_{}_{}_{}_{}_{}'.format(model_name, name, n_epochs, n_neurons, n_lag, n_seq), 'wb')
    pickle.dump(rmses, f)
    f.close()

# store test rmses
def store_test_rmses(rmses, model_name, name, n_epochs, n_neurons, n_lag, n_seq):
    f = open('test_rmses/{}_{}_{}_{}_{}_{}'.format(model_name, name, n_epochs, n_neurons, n_lag, n_seq), 'wb')
    pickle.dump(rmses, f)
    f.close()

# load ARIMA forecasts
def load_arima_forecasts(model_name, name, n_epochs, n_neurons, n_lag, n_seq):
    f = open('arima_forecasts/{}_{}_{}_{}_{}_{}'.format(model_name, name, n_epochs, n_neurons, n_lag, n_seq), 'rb')
    data = pickle.load(f)
    f.close()   
    return data

# load train rmses
def load_train_rmses(model_name, name, n_epochs, n_neurons, n_lag, n_seq):
    f = open('train_rmses/{}_{}_{}_{}_{}_{}'.format(model_name, name, n_epochs, n_neurons, n_lag, n_seq), 'rb')
    data = pickle.load(f)
    f.close()   
    return data

# load test rmses
def load_test_rmses(model_name, name, n_epochs, n_neurons, n_lag, n_seq):
    f = open('test_rmses/{}_{}_{}_{}_{}_{}'.format(model_name, name, n_epochs, n_neurons, n_lag, n_seq), 'rb')
    data = pickle.load(f)
    f.close()   
    return data

# save multivariate dataframe
def store_gram(feature_df, name, gramtype):
    f = open('mv_dataframe/{}_{}'.format(name, gramtype), 'wb')
    pickle.dump((feature_df), f)
    f.close()

# load multivariate dataframe
def load_multivariate(name, gramtype):
    f = open('mv_dataframe/{}_{}'.format(name, gramtype), 'rb')
    data = pickle.load(f)
    f.close()   
    return data

# store ARIMA train forecasts
def store_arima_boost_data(forecasts, model_name, name, n_epochs, n_neurons, n_lag, n_seq):
    f = open('arima_boost_data/{}_{}_{}_{}_{}_{}'.format(model_name, name, n_epochs, n_neurons, n_lag, n_seq), 'wb')
    pickle.dump(forecasts, f)
    f.close()

# load ARIMA train forecasts
def load_arima_boost_data(model_name, name, n_epochs, n_neurons, n_lag, n_seq):
    f = open('arima_boost_data/{}_{}_{}_{}_{}_{}'.format(model_name, name, n_epochs, n_neurons, n_lag, n_seq), 'rb')
    data = pickle.load(f)
    f.close()   
    return data