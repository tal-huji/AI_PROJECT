import os
import torch
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from config import hyperparams

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


def plot_performance(train_performance=None, test_performance=None, train_market_values=None, test_market_values=None,
                     df_train=None, df_test=None, algorithm_name="", n_dynamic_features=0, ticker="",
                     start_train="", end_train="", start_test="", end_test="",
                     train_positions=None, test_positions=None):
    # Set common parameters for the title
    n_episodes = hyperparams['n_episodes']
    learning_rate = hyperparams['learning_rate']
    positions = hyperparams['positions']  # The list of possible actions (positions)

    # Function to plot positions (actions) with different colors based on action indices
    def plot_positions(ax, dates, positions_data, portfolio_values):
        # Dictionary to track if a label has been added to the legend
        label_added = {'sell_all': False, 'long': False, 'short': False}

        # Loop through each position in the data
        for i, pos in enumerate(positions_data):
            if pos == 0:
                ax.scatter(dates[i], portfolio_values[i], color='black', marker='o',zorder=3)
                if not label_added['sell_all']:
                    ax.scatter([], [], color='black', marker='o', s=50, label='Sell All')
                    label_added['sell_all'] = True
            elif pos > 0:
                # size = 100 if pos == 1 else (25 + 25 * np.abs(pos))
                ax.scatter(dates[i], portfolio_values[i], color='blue', marker='^',  zorder=3)
                if not label_added['long']:
                    ax.scatter([], [], color='blue', marker='^', s=50, label='Buy All')
                    label_added['long'] = True
            elif pos < 0:
                # size = 25 if pos == -1 else (25 + 25 * np.abs(pos))
                ax.scatter(dates[i], portfolio_values[i], color='red', marker='v',zorder=3)
                if not label_added['short']:
                    ax.scatter([], [], color='red', marker='v', s=50, label='Short')
                    label_added['short'] = True

    # Plot Training performance if training data is provided
    if train_performance is not None and df_train is not None:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        train_dates = df_train.index[-len(train_performance):]
        ax1.plot(train_dates, train_performance, label=f'{algorithm_name} Portfolio Value')
        ax1.plot(train_dates, train_market_values[-len(train_performance):], label='Market Value', color='green')

        # Plot positions (actions taken by the agent) above the lines
        if train_positions is not None:
            plot_positions(ax1, train_dates, np.array(train_positions), np.array(train_performance))


        ax1.set_title(get_title('Training', algorithm_name, ticker))
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()

        plt.tight_layout()
        plt.show()

    # Plot Testing performance if testing data is provided
    if test_performance is not None and df_test is not None:
        # Check that the test data lengths are aligned
        test_performance_len = len(test_performance)
        test_dates_len = len(df_test.index)
        test_market_values_len = len(test_market_values)

        # Take the minimum of the lengths to avoid mismatch
        min_test_len = min(test_performance_len, test_dates_len, test_market_values_len)

        # Only use the matching length for dates, performance, and market values
        test_dates = df_test.index[-min_test_len:]
        test_performance = test_performance[-min_test_len:]
        test_market_values = test_market_values[-min_test_len:]

        fig, ax2 = plt.subplots(figsize=(12, 6))

        ax2.plot(test_dates, test_performance, label=f'{algorithm_name} Portfolio Value')
        ax2.plot(test_dates, test_market_values, label='Market Value', color='green')

        # Plot positions (actions taken by the agent) above the lines
        if test_positions is not None:
            test_positions = np.array(test_positions[-min_test_len:])  # Ensure positions match the test length
            plot_positions(ax2, test_dates, test_positions, np.array(test_performance))



        ax2.set_title(get_title('Testing', algorithm_name, ticker))
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend()

        plt.tight_layout()
        plt.show()


def get_title(mode, algorithm_name, ticker):

    n_episodes = hyperparams['n_episodes']
    learning_rate = hyperparams['learning_rate']
    positions = hyperparams['positions']  # The list of possible actions (positions)
    start_date = hyperparams['start_date']
    end_date = hyperparams['end_date']

    title = f'{mode} Performance on {ticker} from {start_date} to {end_date}'
    title += f'\n{algorithm_name} with {n_episodes} episodes, learning rate {learning_rate}'
    title += f'\nPositions: {positions}'

    if 'dqn' in algorithm_name.lower():
        title += f' | Hidden layers: {hyperparams["hidden_layer_size"]}'

    if algorithm_name.lower() == 'dqn_lstm' or algorithm_name.lower() == 'dqn_gru':
        title += f'\nLSTM with {hyperparams["lstm_num_layers"]} layers, hidden size {hyperparams["lstm_hidden_size"]}'


    return title
