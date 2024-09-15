import os
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import title

from config import hyperparams, dynamic_features_arr

# Ensure the directory exists for saving the plots
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Convert algorithm names to the desired format
def convert_algorithm_name(algorithm):
    algorithm_mapping = {
        'q-learning': 'qlearning',
        'dqn': 'dqn_regular',
        'dqn_gru': 'dqn_gru',
        'dqn_cnn': 'dqn_cnn',
        'dqn_gru_cnn': 'dqn_gru_cnn',
        'policy_gradient': 'policy_gradient_regular',
        'policy_gradient_gru': 'policy_gradient_gru',
        'policy_gradient_cnn': 'policy_gradient_cnn',
        'policy_gradient_gru_cnn': 'policy_gradient_gru_cnn'
    }
    return algorithm_mapping.get(algorithm, algorithm)

# Convert stock ticker to lowercase format (if necessary)
def convert_stock_ticker(ticker):
    return ticker.lower()

# Function to create a hashable state
def get_hashasble_state(state):
    if isinstance(state, np.ndarray):
        state = state.flatten()
    elif isinstance(state, list):
        state = np.array(state).flatten()
    return tuple(state)

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

# Plot performance through time
def plot_performance_through_time(
    ticker,
    df,
    portfolio_dict,
    buy_hold_dict,
    test_actions_dict,
    baseline_portfolio_dict,
    baseline_buy_hold_dict,
    baseline_actions_dict,
    interval
):
    """
    Plot the yearly performance of the algorithm against the buy-and-hold strategy and a baseline model.
    """
    # Convert the dictionary values into lists aligned to the full year dates
    trading_dates = df.index

    # Extract portfolio values, buy-hold values, and actions
    portfolio_values = [portfolio_dict.get(date, None) for date in trading_dates]
    buy_hold_values = [buy_hold_dict.get(date, None) for date in trading_dates]
    test_actions = [test_actions_dict.get(date, None) for date in trading_dates]

    baseline_portfolio_values = [baseline_portfolio_dict.get(date, None) for date in trading_dates]
    baseline_buy_hold_values = [baseline_buy_hold_dict.get(date, None) for date in trading_dates]
    baseline_actions = [baseline_actions_dict.get(date, None) for date in trading_dates]

    # Convert lists to numpy arrays for interpolation
    portfolio_values = np.array([np.nan if v is None else v for v in portfolio_values])
    buy_hold_values = np.array([np.nan if v is None else v for v in buy_hold_values])
    baseline_portfolio_values = np.array([np.nan if v is None else v for v in baseline_portfolio_values])

    # Interpolate missing values for portfolio and buy-hold values
    interpolated_portfolio_values = pd.Series(portfolio_values).interpolate(method='linear').to_numpy()
    interpolated_buy_hold_values = pd.Series(buy_hold_values).interpolate(method='linear').to_numpy()
    interpolated_baseline_portfolio_values = pd.Series(baseline_portfolio_values).interpolate(method='linear').to_numpy()

    # Generate buy/sell signals from the test actions dictionary
    buy_signals = [np.nan] * len(test_actions)
    sell_signals = [np.nan] * len(test_actions)
    buy_signals_baseline = [np.nan] * len(baseline_actions)
    sell_signals_baseline = [np.nan] * len(baseline_actions)

    for i, action in enumerate(test_actions):
        if action == 1:
            buy_signals[i] = interpolated_portfolio_values[i]
        elif action == 0:
            sell_signals[i] = interpolated_portfolio_values[i]

    for i, action in enumerate(baseline_actions):
        if action == 1:
            buy_signals_baseline[i] = interpolated_baseline_portfolio_values[i]
        elif action == 0:
            sell_signals_baseline[i] = interpolated_baseline_portfolio_values[i]

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 6))

    algorithm_name = convert_algorithm_name(hyperparams['algorithm'])
    baseline_name = convert_algorithm_name(hyperparams['baseline_algorithm'])
    ticker_name = convert_stock_ticker(ticker)

    # Plot portfolio vs buy-hold values
    ax.plot(trading_dates, interpolated_portfolio_values, label=f'{algorithm_name} Portfolio Value', color='orange')
    ax.plot(trading_dates, interpolated_buy_hold_values, label='Buy-and-Hold Value', color='blue')
    ax.plot(trading_dates, interpolated_baseline_portfolio_values, label=f'{baseline_name} Portfolio Value', color='purple')

    # Add buy/sell markers
    if hyperparams['show_buy_sell_signals']:
        ax.plot(trading_dates, buy_signals, '^', markersize=2, color='green', label='Buy Signal', linestyle='None')
        ax.plot(trading_dates, sell_signals, 'o', markersize=2, color='black', label='Sell Signal', linestyle='None')

    # Set title and labels
    title = f'Interval: {interval}\n'
    title  += f'{ticker} - Yearly Performance - {algorithm_name.upper()} vs {baseline_name}'

    title = add_params_to_title(title)

    ax.set_title(title)
    ax.set_ylabel('Value (USD)')
    ax.set_xlabel('Date')

    # Add a legend
    ax.legend()

    # Format the x-axis for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Ensure the directory for saving exists
    output_directory = f'plots/{algorithm_name}'
    ensure_directory_exists(output_directory)

    # Save the plot as a PNG file (override if exists)
    plt.savefig(f'{output_directory}/{algorithm_name}_{ticker_name}.png', dpi=300)

    # Show the plot
    plt.show()

# Plot final results comparing different algorithms
def plot_final_results(results, baseline_results):
    """
    Plot final results comparing algorithm performance vs buy-and-hold and a baseline.
    """
    # Close any previous figures to avoid creating a blank plot
    plt.close()

    # Bar plot for each stock showing percentage return of final buy-and-hold vs final portfolio value
    tickers = [result['ticker'] for result in results]
    final_portfolio_values = [result['final_portfolio_value'] for result in results]
    final_buy_hold_values = [result['final_buy_hold'] for result in results]
    final_portfolio_values_baseline = [result['final_portfolio_value'] for result in baseline_results]

    # Calculate percentage returns for each stock
    initial_investment = 10000  # Assuming 10000 initial investment for each stock
    portfolio_returns = [(val - initial_investment) / initial_investment * 100 if val is not None else 0 for val in final_portfolio_values]
    buy_hold_returns = [(val - initial_investment) / initial_investment * 100 if val is not None else 0 for val in final_buy_hold_values]
    portfolio_returns_baseline = [(val - initial_investment) / initial_investment * 100 if val is not None else 0 for val in final_portfolio_values_baseline]

    # Calculate total returns for the entire portfolio
    total_portfolio_value = sum([val for val in final_portfolio_values if val is not None])
    total_buy_hold_value = sum([val for val in final_buy_hold_values if val is not None])
    total_portfolio_value_baseline = sum([val for val in final_portfolio_values_baseline if val is not None])

    total_portfolio_return = ((total_portfolio_value - initial_investment * len(tickers)) / (initial_investment * len(tickers))) * 100
    total_buy_hold_return = ((total_buy_hold_value - initial_investment * len(tickers)) / (initial_investment * len(tickers))) * 100
    total_portfolio_return_baseline = ((total_portfolio_value_baseline - initial_investment * len(tickers)) / (initial_investment * len(tickers))) * 100

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(tickers))
    width = 0.25  # Reduced the width to fit three bars

    algorithm_name = convert_algorithm_name(hyperparams['algorithm'])
    baseline_name = convert_algorithm_name(hyperparams['baseline_algorithm'])

    # Offset the bars for each group
    ax.bar(x - width, portfolio_returns, width, label=f'{algorithm_name.upper()} Portfolio Return (%) {total_portfolio_return:.2f}%', color='orange')
    ax.bar(x, buy_hold_returns, width, label=f'Buy-and-Hold Return (%) {total_buy_hold_return:.2f}%', color='blue')
    ax.bar(x + width, portfolio_returns_baseline, width, label=f'{baseline_name.upper()} Portfolio Return (%) {total_portfolio_return_baseline:.2f}%', color='purple')

    # Set title and labels
    ax.set_title(add_params_to_title(f'{algorithm_name.upper()} vs Buy-and-Hold vs {baseline_name.upper()} - Portfolio Return (%)'))
    ax.set_ylabel('Return (%)')
    ax.set_xlabel('Stock')

    # Add a legend with the percentage returns in the labels
    ax.legend()

    # Set the x-axis labels to be the tickers
    ax.set_xticks(x)
    ax.set_xticklabels([convert_stock_ticker(ticker) for ticker in tickers])

    # Ensure the directory for saving exists
    output_directory = f'plots/{algorithm_name}'
    ensure_directory_exists(output_directory)

    # Save the plot as a PNG file (override if exists)
    plt.savefig(f'{output_directory}/{algorithm_name}_comparison_bar.png', dpi=300)

    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Helper function to add parameters to the title
def add_params_to_title(initial_title):
    title = initial_title
    title += f' (Interval: {hyperparams["interval_days"]} days) | Episodes: {hyperparams["n_episodes"]} | Retrain: {hyperparams["retrain"]}\n'

    if 'policy' or 'dqn' in hyperparams['algorithm']:
        title += f'Learning Rate: {hyperparams["learning_rate"]} | Hidden Layer Size: {hyperparams["hidden_layer_size"]}'

    if 'lstm' in hyperparams['algorithm'] or 'gru' in hyperparams['algorithm']:
        title += f' | Memory Num Layers: {hyperparams["lstm_num_layers"]}'

    title += f' | dynamic_features len: {len(dynamic_features_arr)} | Exploration Rate: {hyperparams["exploration_rate"]}'

    return title
