# utils.py
import os
import numpy as np
import torch
import yfinance as yf
import matplotlib.pyplot as plt
from gym_trading_env.utils.history import History
from matplotlib.pyplot import title
from config import hyperparams
import inspect
import pandas as pd
import numpy as np
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

    # Dynamic features (printing their names)
    dynamic_features_names = []
    for dynamic_feature in dynamic_features_arr:
        # Get the source code (name) of the lambda function
        feature_name = inspect.getsource(dynamic_feature).strip()

        # Remove the unwanted part before "dynamic_feature_"
        feature_name = feature_name.split("dynamic_feature_", 1)[-1]

        dynamic_features_names.append(feature_name)

    # Create a string representation of dynamic features
    dynamic_features_str = "\n".join(dynamic_features_names)

    title += f'Episodes: {n_episodes}, Dynamic Features: {n_dynamic_features}, Learning Rate: {learning_rate} | Positions: {positions}\n'
    title += f'Start Train: {start_train}, End Train: {end_train}, Start Test: {start_test}\n'
    title += f'Dynamic Features:\n{dynamic_features_str}\n'  # Add the names of dynamic features to the title

    if algorithm_name == 'dqn':
        title += f'Hidden Layer Size: {hyperparams["hidden_layer_size"]}, Memory Size: {hyperparams["memory_size"]}, Batch Size: {hyperparams["batch_size"]}\n'

    if algorithm_name == 'q-learning':
        title += f'Num Bins: {hyperparams["num_bins"]}\n'

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




def compute_min_max_for_features(df_train, dynamic_features_arr):
    """
    Compute the min and max values for each active dynamic feature in the training data.

    Parameters:
    - df_train: A DataFrame of the historical training data.
    - dynamic_features_arr: List of active dynamic features (lambdas).

    Returns:
    - min_max_array: List of tuples containing min and max values for each feature, in the correct order.
    """
    # Make a copy of the DataFrame
    df_train_copy = df_train.copy()

    # Initialize the History object
    history = History(max_size=100000)

    # Initialize history storage using the set function (set the first row)
    first_row = df_train_copy.iloc[0]
    history.set(
        idx=0,
        date=first_row.name,
        position=0,  # Example: Initial position as 0
        real_position=0,
        data=first_row.to_dict()
    )

    min_max_array = []  # Renamed to min_max_array

    # Iterate through each dynamic feature in dynamic_features_arr
    for dynamic_feature in dynamic_features_arr:
        feature_min = np.inf
        feature_max = -np.inf

        # Loop through each row of the DataFrame for the given feature
        for i in range(1, len(df_train_copy)):  # Start from 1 to allow windows to be applied
            row = df_train_copy.iloc[i]

            # Add current row data to history using the add function
            history.add(
                idx=i,
                date=row.name,
                position=0,  # Example: Position is static for simplicity
                real_position=0,  # Example: Real position is static
                data=row.to_dict()
            )

            # Compute the feature value using the dynamic feature
            feature_value = dynamic_feature(history)

            # Update the min and max for this feature
            feature_min = min(feature_min, feature_value)
            feature_max = max(feature_max, feature_value)

        # Store the min and max for this feature in the correct order
        min_max_array.append((feature_min, feature_max))

    return min_max_array

