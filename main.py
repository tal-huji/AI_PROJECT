import math
from textwrap import shorten

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import torch

from device import device
from utils import fetch_data, plot_performance, set_all_seeds
from models.dqn_gru import DQNAgent_GRU
from models.dqn_lstm import DQNAgent_LSTM
from models.ql import QLearningAgent, get_hashasble_state
from models.dqn import DQNAgent
from env_utils import create_trading_env
from config import hyperparams, dynamic_features_arr

def get_last_valid_value(values_list):
    """
    Helper function to get the last valid (non-None) value from a list.
    """
    for value in reversed(values_list):
        if value is not None:
            return value
    return 0  # Return 0 if no valid value is found


def main(agent_type, interval_days):
    print(f"Running {agent_type.upper()} agent for training and testing each stock...")

    set_all_seeds(hyperparams['seed'])

    tickers = [
        'TSLA',
        'GOOGL',
        'NFLX',
        'AAPL',
        'AMZN',
        'MSFT',
    ]

    final_results = []
    initial_position = hyperparams['initial_position']

    for ticker in tickers:
        print(f"\nTraining and testing for {ticker}...")

        start_year = hyperparams['start_year']
        end_year = hyperparams['end_year']

        # Fetch stock data
        df_full_year = fetch_data(ticker, start_year, end_year).sort_index()
        n_days = len(df_full_year)


        # Calculate the highest multiple of interval_days that fits into the total number of trading days
        max_n_intervals = math.floor(n_days / interval_days)
        adjusted_n_days = max_n_intervals * interval_days - 1  # This is the adjusted number of days

        # Track portfolio, market values, and buy-and-hold (market strategy)

        # Only take the adjusted number of trading days
        trading_dates =   df_full_year.index[:adjusted_n_days]
        portfolio_dict = {date: None for date in trading_dates}
        buy_hold_market_dict = {date: None for date in trading_dates}
        test_actions_dict = {date: None for date in trading_dates}

        # Loop over train/test intervals, using the adjusted number of trading days
        i = 0

        # Initialize variable to store the last test day for the next train interval
        initial_close_price = None
        initial_portfolio_value = None

        while i + interval_days <= adjusted_n_days:
            # Define the training set with overlap: add the last day from the previous test
            df_train = df_full_year.iloc[i:i + interval_days]

            # Define the test set: start from the last day of the train and extend for the test interval
            end_test = i + interval_days + interval_days
            df_test = df_full_year.iloc[i + interval_days - 1: end_test]

            if initial_portfolio_value is None:
                initial_portfolio_value = hyperparams['portfolio_initial_value']

            if initial_close_price is None:
                initial_close_price = df_test['close'].iloc[0]

            # Pass the propagated initial_portfolio_value when creating env_train and env_test
            env_train = create_trading_env(df_train, dynamic_features_arr, initial_portfolio_value)
            env_test = create_trading_env(df_test, dynamic_features_arr, initial_portfolio_value)

            state_shape = env_train.observation_space.shape
            action_size = env_train.action_space.n

            # Initialize agent
            agent = initialize_agent(agent_type, env_train, state_shape, action_size)

            print(f"Training agent for {ticker} from {df_train.index[0]} to {df_train.index[-1]}...")
            train_portfolio_values, train_actions = agent.train_agent()

            print(f"Testing agent for {ticker} from {df_test.index[0]} to {df_test.index[-1]}...")
            test_portfolio_values, test_actions, final_portfolio_return, final_market_return = test_agent(agent,
                                                                                                          env_test,
                                                                                                          agent_type)
            # After testing, update the initial_portfolio_value with the final portfolio value
            initial_portfolio_value = test_portfolio_values[-1] if test_portfolio_values else initial_portfolio_value
            initial_position = test_actions[-1] if test_actions else initial_position

            # Map test results to dates, passing the initial_close_price set once
            map_test_results_to_dates(
                df_test,
                test_portfolio_values,
                test_actions,
                portfolio_dict,
                buy_hold_market_dict,
                test_actions_dict,
                initial_close_price  # No longer updating this value, remains fixed from start
            )

            i += interval_days

        if i ==0:
            # Raise an error if the interval_days is too large
            raise ValueError(f"Interval days {interval_days} is too large for the total number of trading days {n_days}.")

        # Plot yearly performance
        plot_yearly_performance(ticker, df_full_year, portfolio_dict, buy_hold_market_dict, test_actions_dict)

        final_portfolio_result = portfolio_dict.get(trading_dates[-1], None)
        final_buy_hold_result = buy_hold_market_dict.get(trading_dates[-1], None)

        final_results.append({
            'ticker': ticker,
            'final_portfolio_value': final_portfolio_result,
            'final_buy_hold': final_buy_hold_result
        })

    # Plot final results across all tickers
    plot_final_results(final_results)



def initialize_agent(agent_type, env_train, state_shape, action_size):
    if agent_type == 'q-learning':
        return QLearningAgent(env_train, None, action_size)
    elif agent_type == 'dqn':
        return DQNAgent(env_train, state_shape, action_size)
    elif agent_type == 'dqn_lstm':
        return DQNAgent_LSTM(env_train, state_shape, action_size)
    elif agent_type == 'dqn_gru':
        return DQNAgent_GRU(env_train, state_shape, action_size)
    else:
        raise ValueError(f'Invalid agent type: {agent_type}')


def map_test_results_to_dates(df_test, test_portfolio_values, test_actions, portfolio_dict, buy_hold_market_dict,
                              actions_dict, initial_test_close_price):
    for j, date in enumerate(df_test.index):
        if j < len(test_portfolio_values):
            portfolio_dict[date] = test_portfolio_values[j]
            actions_dict[date] = hyperparams['positions'][test_actions[j]]

            # Calculate Buy-and-Hold starting from the first test price (initial_test_close_price)
            buy_hold_market_dict[date] = df_test['close'].iloc[j] * (hyperparams['portfolio_initial_value'] / initial_test_close_price)
        else:
            buy_hold_market_dict[date] = None

def test_agent(agent, env_test, agent_type):
    """
    Test a single agent.
    """
    portfolio_values, actions = [], []
    state, _ = env_test.reset(seed=42)
    total_reward, done = 0, False

    initial_portfolio_value = hyperparams['portfolio_initial_value']
    initial_market_value, final_market_value = None, None

    if agent_type in ['dqn', 'dqn_lstm', 'dqn_gru']:
        agent.model.eval()
        agent.target_model.eval()

    while not done:
        q_values = select_action(agent, agent_type, state)
        action = np.argmax(q_values)

        next_state, reward, terminated, truncated, info = env_test.step(action)
        done = terminated or truncated

        current_open_price = info.get('data_open', 0)
        current_close_price = info.get('data_close', 0)

        # Check if the necessary info is available
        if 'portfolio_valuation' in info and current_open_price and current_close_price:
            portfolio_value = info['portfolio_valuation'] * current_open_price / current_close_price
        else:
            portfolio_value = 0
            print(f"Warning: Missing 'portfolio_valuation' or price data for test step. Info: {info}")

        initial_market_value = initial_market_value or current_close_price
        final_market_value = current_close_price

        state = next_state
        total_reward += reward
        portfolio_values.append(portfolio_value)
        actions.append(hyperparams['positions'][action])


    final_portfolio_value = portfolio_values[-1] if portfolio_values else 0
    final_portfolio_return = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100 if initial_portfolio_value else 0
    final_market_return = ((final_market_value - initial_market_value) / initial_market_value) * 100 if initial_market_value else 0

    # Debug prints to check final portfolio and market returns
    print(f"Final Portfolio Value: {final_portfolio_value}, Final Market Value: {final_market_value}")
    print(f"Final Portfolio Return: {final_portfolio_return}%, Final Market Return: {final_market_return}%")

    return portfolio_values, actions, final_portfolio_return, final_market_return


def select_action(agent, agent_type, state):
    if agent_type in ['dqn', 'dqn_lstm', 'dqn_gru']:
        agent.normalizer.update(state)
        normalized_state = agent.normalizer.normalize(state)
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
        with torch.no_grad():
            if agent_type == 'dqn':
                q_values = agent.model(state_tensor).detach().cpu().numpy()[0]
            else:
                q_values, _ = agent.model(state_tensor)
                q_values = q_values.detach().cpu().numpy()[0]
    elif agent_type == 'q-learning':
        hashable = get_hashasble_state(state)
        q_values = agent.q_table[hashable]
    return q_values


def calculate_final_returns(df_full_year, aligned_portfolio_values):
    """
    Calculate the final portfolio and market returns.
    """
    valid_portfolio_values = [val for val in aligned_portfolio_values if val is not None]

    if len(valid_portfolio_values) == 0:
        return 0, 0

    initial_portfolio_value = valid_portfolio_values[0]
    final_portfolio_value = valid_portfolio_values[-1]

    initial_market_value = df_full_year['close'].iloc[0]
    final_market_value = df_full_year['close'].iloc[-1]

    total_portfolio_return = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100
    total_market_return = ((final_market_value - initial_market_value) / initial_market_value) * 100

    return total_portfolio_return, total_market_return


def calculate_final_buy_hold_returns(df_full_year, aligned_buy_hold_market_values):
    """
    Calculate the final buy-and-hold market returns.
    """
    valid_market_values = [val for val in aligned_buy_hold_market_values if val is not None]

    if len(valid_market_values) == 0:
        return 0

    initial_market_value = valid_market_values[0]
    final_market_value = valid_market_values[-1]

    total_market_return = ((final_market_value - initial_market_value) / initial_market_value) * 100

    return total_market_return


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_yearly_performance(ticker, df, portfolio_dict, buy_hold_dict, test_actions_dict):
    # Convert the dictionary values into lists aligned to the full year dates
    trading_dates = df.index

    # Extract portfolio values, buy-hold values, and actions
    portfolio_values = [portfolio_dict.get(date, None) for date in trading_dates]
    buy_hold_values = [buy_hold_dict.get(date, None) for date in trading_dates]
    test_actions = [test_actions_dict.get(date, None) for date in trading_dates]

    # Convert lists to numpy arrays for interpolation
    portfolio_values = np.array([np.nan if v is None else v for v in portfolio_values])
    buy_hold_values = np.array([np.nan if v is None else v for v in buy_hold_values])

    # Interpolate missing values for portfolio and buy-hold values
    interpolated_portfolio_values = pd.Series(portfolio_values).interpolate(method='linear').to_numpy()
    interpolated_buy_hold_values = pd.Series(buy_hold_values).interpolate(method='linear').to_numpy()

    # Generate buy/sell signals from the test actions dictionary
    buy_signals = [np.nan] * len(test_actions)
    sell_signals = [np.nan] * len(test_actions)

    for i, action in enumerate(test_actions):
        if action == 1:  # Assuming 1 is the "buy" action
            buy_signals[i] = interpolated_portfolio_values[i]
        elif action == 0:  # Assuming 0 is the "sell" action
            sell_signals[i] = interpolated_portfolio_values[i]

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot portfolio vs buy-hold values
    ax.plot(trading_dates, interpolated_portfolio_values, label='Portfolio Value', color='orange')
    ax.plot(trading_dates, interpolated_buy_hold_values, label='Buy-and-Hold Value', color='blue')

    # Add buy/sell markers
    ax.plot(trading_dates, buy_signals, '^', markersize=2, color='green', label='Buy Signal', linestyle='None')
    ax.plot(trading_dates, sell_signals, 'o', markersize=2, color='black', label='Sell Signal', linestyle='None')

    # Set title and labels
    title = f'{ticker} - Yearly Performance - {hyperparams["algorithm"].upper()}'

    if hyperparams['interval_days'] > 1:
        title += f' (Interval: {hyperparams["interval_days"]} days)\n '

    if 'dqn' in hyperparams['algorithm']:
        title += 'Hidden Layer Size: ' + str(hyperparams['hidden_layer_size'])

    if 'lstm' in hyperparams['algorithm'] or 'gru' in hyperparams['algorithm']:
        title += ' | Memory Num Layers: ' + str(hyperparams['lstm_num_layers'])

    ax.set_title(title)
    ax.set_ylabel('Value (USD)')
    ax.set_xlabel('Date')

    # Add a legend
    ax.legend()

    # Format the x-axis for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_final_results(results):
    # bar plot for each stock showing percentage return of final buy-and-hold vs final portfolio value

    tickers = [result['ticker'] for result in results]
    final_portfolio_values = [result['final_portfolio_value'] for result in results]
    final_buy_hold_values = [result['final_buy_hold'] for result in results]

    # Calculate percentage returns for each stock
    initial_investment = 10000  # Assuming 10000 initial investment for each stock
    portfolio_returns = [(val - initial_investment) / initial_investment * 100 if val is not None else 0 for val in final_portfolio_values]
    buy_hold_returns = [(val - initial_investment) / initial_investment * 100 if val is not None else 0 for val in final_buy_hold_values]

    # Calculate total returns for the entire portfolio
    total_portfolio_value = sum([val for val in final_portfolio_values if val is not None])
    total_buy_hold_value = sum([val for val in final_buy_hold_values if val is not None])

    total_portfolio_return = ((total_portfolio_value - initial_investment * len(tickers)) / (initial_investment * len(tickers))) * 100
    total_buy_hold_return = ((total_buy_hold_value - initial_investment * len(tickers)) / (initial_investment * len(tickers))) * 100

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(tickers))
    width = 0.35

    ax.bar(x - width/2, portfolio_returns, width, label='Portfolio Return (%)', color='orange')
    ax.bar(x + width/2, buy_hold_returns, width, label='Buy-and-Hold Return (%)', color='blue')

    # Set title and labels
    ax.set_title('Portfolio Return (%) vs Buy-and-Hold Return (%)')
    ax.set_ylabel('Return (%)')
    ax.set_xlabel('Stock')

    # Annotate the total returns on the plot
    total_return_text = f'Total Portfolio Return: {total_portfolio_return:.2f}%\nTotal Buy-and-Hold Return: {total_buy_hold_return:.2f}%'
    ax.text(0.95, 0.95, total_return_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.6))

    # Add a legend
    ax.legend()

    # Set the x-axis labels to be the tickers
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)

    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




# Example usage
main(agent_type=hyperparams['algorithm'], interval_days=hyperparams['interval_days'])

