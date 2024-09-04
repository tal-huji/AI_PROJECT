# utils.py
import os
import numpy as np
import torch
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.pyplot import title
from config import hyperparams

from config import hyperparams
from config import dynamic_features_arr

# Set all seeds for reproducibility
def set_all_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fetch data using yfinance
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df['close'] = df['Adj Close']
    df['open'] = df['Open']
    df['high'] = df['High']
    df['low'] = df['Low']
    df['volume'] = df['Volume']
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df.dropna()

# Plot performance
def plot_performance(train_performance, test_performance, train_market_values, test_market_values,
                     df_train, df_test, algorithm_name, n_dynamic_features, ticker,
                     start_train, end_train, start_test, end_test):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Training performance
    train_dates = df_train.index[-len(train_performance):]
    ax1.plot(train_dates, train_performance, label=f'{algorithm_name} Portfolio Value')
    ax1.plot(train_dates, train_market_values[-len(train_performance):], label='Market Value', color='green')

    title = f'Training Performance on {ticker}\n'

    n_episodes = hyperparams['n_episodes']
    learning_rate = hyperparams['learning_rate']
    positions = hyperparams['positions']


    title += f'Episodes: {n_episodes}, Dynamic Features: {n_dynamic_features}, Learning Rate: {learning_rate} | Positions\n'
    title+= f'Start Train: {start_train}, End Train: {end_train}, Start Test: {start_test}\n'

    if algorithm_name == 'dqn':
        title += f'Hidden Layer Size: {hyperparams["hidden_layer_size"]}, Memory Size: {hyperparams["memory_size"]}, Batch Size: {hyperparams["batch_size"]}\n'

    ax1.set_title(title)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()

    # Testing performance
    test_dates = df_test.index[-len(test_performance):]
    ax2.plot(test_dates, test_performance, label=f'{algorithm_name} Portfolio Value')
    ax2.plot(test_dates, test_market_values[-len(test_performance):], label='Market Value', color='green')
    ax2.set_title(f'Testing Performance on {ticker}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.legend()

    plt.tight_layout()
    plt.show()


