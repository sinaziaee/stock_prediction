import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from models import LSTMModel, GRUModel, RNNModel
import time
import math, time
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch

def plot_stock_price(df):
    sns.set_style("darkgrid")
    plt.figure(figsize = (15,9))
    plt.plot(df[['Adj Close']])
    plt.title("Bitcoin Stock Price",fontsize=18, fontweight='bold')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price (USD)',fontsize=18)
    plt.show()
    
def normalize_data(df):
    scaler = MinMaxScaler()
    normalized_data_close_price = scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))
    return normalized_data_close_price, scaler

def split_data(stock, lookback):
    if type(stock) is np.ndarray:
        data_raw = stock
    else:
        data_raw = stock.to_numpy() # convert to numpy array
    data = []
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback+1])
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    return [x_train, y_train, x_test, y_test]

def plot_training_results(original, predict, loss_list):
    plt.figure(figsize=(15, 5))
    sns.set_style("darkgrid")    
    sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
    plt.title('Stock price', size = 14, fontweight='bold')
    plt.xlabel("Days", size = 14)
    plt.ylabel("Cost (USD)", size = 14)
    plt.show()

    plt.figure(figsize=(6, 3))
    sns.lineplot(data=loss_list, color='royalblue')
    plt.xlabel("Epoch", size = 14)
    plt.ylabel("Loss", size = 14)
    plt.title("Training Loss", size = 14, fontweight='bold')
    plt.show()


def plot_all_results(train_y, train_predict, test_y, test_predict):
    plt.figure(figsize=(15, 5))
    sns.set_style("darkgrid")    
    train_y_length = len(train_y)
    test_x_values = range(train_y_length, train_y_length + len(test_y))
    
    sns.lineplot(x=train_y.index, y=train_y[0], label="Train Data", color='royalblue')
    sns.lineplot(x=train_predict.index, y=train_predict[0], label="Train Prediction (LSTM)", color='tomato')
    sns.lineplot(x=test_x_values, y=test_y[0], label="Test Data", color='blue')
    sns.lineplot(x=test_x_values, y=test_predict[0], label="Test Prediction (LSTM)", color='gold')
    
    plt.title('Stock price', size=14, fontweight='bold')
    plt.xlabel("Days", size=14)
    plt.ylabel("Cost (USD)", size=14)
    plt.show()


def plot_future_predictions(df, predictions):
    plt.figure(figsize=(15, 5))
    sns.set_style("darkgrid")
    plt.plot(range(len(df)), df['Adj Close'], label='Previous Data', color='blue')
    plt.plot(range(len(df)-1, len(df)+len(predictions)-1), predictions, label='Predictions in the next 10 days', color='red')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Predictions')
    plt.legend()
    plt.show()
    
    
def define_model(input_dim, hidden_dim, output_dim, num_layers, model_type='LSTM'):
    if model_type == 'LSTM':
        model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    elif model_type == 'GRU':
        model = GRUModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    elif model_type == 'RNN':
        model = RNNModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    return model
    
def train_model(model, x_train, y_train, criterion, optimiser, num_epochs=100):
    loss_list = []
    start_time = time.time()
    model.train()
    for i in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        loss_list.append(loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if i % 10 == 0:
            print("Epoch: {}, Loss: {}".format(i, loss.item()))
    training_time = time.time()-start_time
    print("Training time: {}".format(training_time))
    return loss_list, model, y_train_pred

def test_model(model, x_test, y_test_lstm, scaler):
    model.eval()
    y_test_pred = model(x_test)
    predict_test_y = pd.DataFrame(scaler.inverse_transform(y_test_pred.detach().numpy()))
    original_test_y = pd.DataFrame(scaler.inverse_transform(y_test_lstm.detach().numpy()))
    testScore = math.sqrt(mean_squared_error(original_test_y.to_numpy()[:,0], predict_test_y.to_numpy()[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    return predict_test_y, original_test_y

def predict_next_day(model, x_test, scaler):
    # Get the last sequence of data from the test set
    last_sequence = x_test[-1:]
    # Predict the next day's value
    next_day_prediction = model(last_sequence)
    # Invert the prediction
    next_day_prediction = scaler.inverse_transform(next_day_prediction.detach().numpy())
    return next_day_prediction.item()

# Initialize an empty list to store the predictions
def predict_next_n_days(model, x_test, scaler, num_days=10):
    predictions = []
    # Get the last sequence of data from the test set
    last_sequence = x_test[-1:]
    # Iterate over the next n days
    for _ in range(num_days):
        # Predict the next day's value
        next_day_prediction = model(last_sequence)
        # Invert the prediction
        next_day_prediction = scaler.inverse_transform(next_day_prediction.detach().numpy())
        # Append the prediction to the list
        predictions.append(next_day_prediction)
        # Update the input sequence by removing the first element and adding the prediction
        last_sequence = torch.cat((last_sequence[:, 1:, :], torch.tensor(next_day_prediction)[:, np.newaxis, :]), axis=1)
    # Convert the predictions list to a numpy array
    predictions = np.concatenate(predictions)
    return predictions