import torch
from torch import nn
from device import device
from env_utils import create_trading_env
from models.dqn_gru import DQN_GRU, DQNAgent_GRU  # GRU-based model
from models.dqn_lstm import DQNAgent_LSTM
from models.ql import QLearningAgent, get_hashasble_state
from utils import fetch_data, plot_performance, set_all_seeds
from models.dqn import DQNAgent
from config import hyperparams
from config import dynamic_features_arr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import mplfinance as mpf

def main(agent_type, train_interval_days, test_interval_days):
    print(f"Running {agent_type.upper()} agent for training and testing each stock...")

    set_all_seeds(hyperparams['seed'])

    tickers = [
        'INTU',
        'TXN',
        'NOW',
        'AMD',
        'MU',
        'ADSK',
        'GOOGL',
        'INTC',
        'TSLA',
        'AAPL',
        'MSFT',
        'IBM',
        'AMZN',
        'ORCL',
        'QCOM',
        'CSCO',
        'CRM',
        'PYPL',
        #_____________________________________________________________________________________
        'NKE',
        'DIS',
        'MCD',
        'SBUX',
        'KO',
    ]

    final_results = []

    for ticker in tickers:
        print(f"\nTraining and testing for {ticker}...")

        start_year = hyperparams['start_year']
        end_year = hyperparams['end_year']

        # Fetch stock data for the given ticker and year
        df_full_year = fetch_data(ticker, start_year, end_year)

        # Ensure data is sorted by date (index)
        df_full_year = df_full_year.sort_index()

        # Extract trading dates from the DataFrame (these are actual trading days)
        trading_dates = df_full_year.index

        # Initialize storage for portfolio values and market values
        portfolio_dict = {date: None for date in trading_dates}
        market_dict = {date: None for date in trading_dates}

        i = 0
        while i + train_interval_days + test_interval_days <= len(df_full_year):
            df_train = df_full_year.iloc[i:i + train_interval_days]
            df_test = df_full_year.iloc[i + train_interval_days:i + train_interval_days + test_interval_days]

            # Create trading environments for training and testing
            env_train = create_trading_env(df_train, dynamic_features_arr)
            env_test = create_trading_env(df_test, dynamic_features_arr)

            state_shape = env_train.observation_space.shape
            action_size = env_train.action_space.n

            # Initialize the agent based on the chosen type
            if agent_type == 'q-learning':
                agent = QLearningAgent(env_train, None, action_size)
            elif agent_type == 'dqn':
                agent = DQNAgent(env_train, state_shape, action_size)
            elif agent_type == 'dqn_lstm':
                agent = DQNAgent_LSTM(env_train, state_shape, action_size)
            elif agent_type == 'dqn_gru':
                agent = DQNAgent_GRU(env_train, state_shape, action_size)
            else:
                raise ValueError(f'Invalid agent type: {agent_type}')

            # Train the agent
            print(f"Training agent for {ticker} from {df_train.index[0]} to {df_train.index[-1]}...")
            train_portfolio_values, train_actions = agent.train_agent()

            # Test the agent
            print(f"Testing agent for {ticker} from {df_test.index[0]} to {df_test.index[-1]}...")
            test_portfolio_values, test_actions, final_portfolio_return, final_market_return = test_agent(agent, env_test, agent_type)

            # Map test portfolio values and market values to the actual test dates
            for j, date in enumerate(df_test.index):
                if j < len(test_portfolio_values):
                    portfolio_dict[date] = test_portfolio_values[j]
                    market_dict[date] = df_test['close'].iloc[j] * (10000 / df_test['close'].iloc[0])
                else:
                    print(f"Warning: Not enough portfolio values for date {date}, skipping...")

            i += train_interval_days + test_interval_days

        # Create lists of portfolio and market values aligned to the actual trading days
        aligned_portfolio_values = [portfolio_dict[date] for date in trading_dates]
        aligned_market_values = [market_dict[date] for date in trading_dates]

        # Plot yearly performance using trading dates (no continuous dates needed)
        plot_yearly_performance(ticker, df_full_year, aligned_portfolio_values)

        # Calculate the overall portfolio and market returns
        final_portfolio_value = [val for val in aligned_portfolio_values if val is not None][-1]
        initial_portfolio_value = [val for val in aligned_portfolio_values if val is not None][0]

        final_market_value = df_full_year['close'].iloc[-1]
        total_market_return = (final_market_value - df_full_year['close'].iloc[0]) / df_full_year['close'].iloc[0] * 100
        total_portfolio_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100

        final_results.append({
            'ticker': ticker,
            'final_portfolio_return': total_portfolio_return,
            'final_market_return': total_market_return
        })

    # Plot the final results across all tickers
    plot_final_results(final_results)




def test_agent(agent, env_test, agent_type):
    """
    Test a single agent.
    """
    portfolio_values = []
    actions = []

    # Initialize the environment and get the initial state
    state, _ = env_test.reset(seed=42)

    total_reward = 0
    done = False
    initial_portfolio_value = hyperparams['portfolio_initial_value']
    initial_market_value = None
    final_market_value = None

    if agent_type == 'dqn' or agent_type == 'dqn_lstm' or agent_type == 'dqn_gru':  # Added GRU
        agent.model.eval()
        agent.target_model.eval()

    while not done:
        if agent_type == 'dqn' or agent_type == 'dqn_lstm' or agent_type == 'dqn_gru':  # Added GRU
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

        action = np.argmax(q_values)

        # Take a step in the environment
        next_state, reward, terminated, truncated, info = env_test.step(action)
        done = terminated or truncated

        current_open_price = info['data_open']
        current_close_price = info['data_close']

        portfolio_value = info['portfolio_valuation'] * current_open_price / current_close_price

        if initial_market_value is None:
            initial_market_value = current_close_price

        final_market_value = current_close_price

        state = next_state
        total_reward += reward
        portfolio_values.append(portfolio_value)
        actions.append(hyperparams['positions'][action])

    final_portfolio_value = portfolio_values[-1]
    final_portfolio_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100

    final_market_return = (final_market_value - initial_market_value) / initial_market_value * 100

    return portfolio_values, actions, final_portfolio_return, final_market_return


def plot_yearly_performance(ticker, df, portfolio_values):
    """
    Plots the yearly performance of the portfolio vs the market using mplfinance for professional stock graphing.
    """
    # Convert the index to a datetime format for mplfinance
    df.index = pd.to_datetime(df.index)

    # Create a copy of df to avoid modifying the original DataFrame
    df_plot = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Clean up the portfolio values: convert None to NaN for gaps
    portfolio_values = np.array([np.nan if v is None else v for v in portfolio_values])

    # Interpolate to fill gaps (NaN values) and make the line continuous
    interpolated_portfolio_values = pd.Series(portfolio_values).interpolate(method='linear')

    # Add the interpolated portfolio values as an additional plot
    additional_plot = mpf.make_addplot(interpolated_portfolio_values, color='Orange', secondary_y=True)

    # Create the plot
    mpf.plot(
        df_plot,
        type='candle',  # Candlestick chart for stock prices
        volume=False,  # Show the volume bars
        title=f'{ticker} Yearly Performance',
        ylabel='Stock Price',
        ylabel_lower='Volume',
        addplot=additional_plot,  # Add the interpolated portfolio performance line
        style='charles',  # A professional style preset
        figratio=(12, 8),  # Figure ratio for a better visual fit
        figscale=1.2,  # Scale the figure size for clarity
        tight_layout=True  # Adjusts layout to avoid overlap
    )


def plot_final_results(results):
    """
    Plots a bar chart comparing the final portfolio return and market return for each stock,
    and displays the combined portfolio return and market return.
    """
    tickers = [result['ticker'] for result in results]
    portfolio_returns = [result['final_portfolio_return'] for result in results]
    market_returns = [result['final_market_return'] for result in results]

    # Calculate combined portfolio return and market return
    combined_portfolio_return = np.mean(portfolio_returns)
    combined_market_return = np.mean(market_returns)

    x = np.arange(len(tickers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size to give more room for text
    rects1 = ax.bar(x - width / 2, portfolio_returns, width, label='Portfolio Return')
    rects2 = ax.bar(x + width / 2, market_returns, width, label='Market Return')

    ax.set_xlabel('Tickers')
    ax.set_ylabel('Returns (%)')
    ax.set_title('Total Portfolio vs Market Returns for Each Stock')

    # Set the x-ticks and reduce the fontsize of the tick labels
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha='right',
                       fontsize=8)  # Adjust fontsize and rotate for better readability

    # Add a legend
    ax.legend()

    # Add combined results as text annotations
    combined_text = f"Combined Portfolio Return: {combined_portfolio_return:.2f}%\nCombined Market Return: {combined_market_return:.2f}%"
    ax.text(0.95, 0.05, combined_text, transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()  # Adjust layout to avoid clipping of the labels
    plt.show()


# Example usage
main(
    agent_type=hyperparams['algorithm'],
    train_interval_days=hyperparams['train_interval_days'],
    test_interval_days=hyperparams['test_interval_days']
)  # Adjusted for dynamic interval
