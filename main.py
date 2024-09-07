

from env_utils import create_trading_env
from models.ql import QLearningAgent
from utils import fetch_data, plot_performance, set_all_seeds
from models.dqn import DQNAgent
from config import hyperparams
from config import dynamic_features_arr
import numpy as np




import numpy as np
import matplotlib.pyplot as plt

def split_data_by_month(df):
    """
    Splits the dataframe into separate dataframes by year and month.
    Returns a dictionary where the key is a tuple (year, month) and the value is the corresponding dataframe.
    """
    monthly_data = {}
    grouped = df.groupby([df.index.year, df.index.month])
    for (year, month), group in grouped:
        monthly_data[(year, month)] = group
    return monthly_data


def weighted_ensemble_q_values(agents, state):
    """
    Calculate a weighted average of Q-values based on each agent's training_reward_last_episode.
    Agents with better rewards in their last training episode have more influence.
    """
    q_values_list = [agent.get_q_values(state) for agent in agents]
    rewards_last_episode = [agent.training_reward_last_episode for agent in agents]

    # Normalize rewards to create performance weights
    performance_weights = np.array(rewards_last_episode) / np.sum(rewards_last_episode)

    # Weighted average of Q-values
    weighted_q_values = np.average(q_values_list, axis=0, weights=performance_weights)
    return weighted_q_values


def main(agent_type='q-learning'):
    print(f"Running {agent_type.upper()} agent with multiple stocks and ensemble testing...")
    set_all_seeds(hyperparams['seed'])

    # Define the list of tickers (stocks) to use
    tickers = [
        'NVDA',
        'GOOGL',
        'AMD',
        'TSLA',
        'AAPL',
        'MSFT',
        'AMZN',
        'INTC',
    ]

    # Store all agents across all stocks
    all_agents = []

    # Train agents for each stock
    for ticker in tickers:
        print(f"\nTraining agents for {ticker}...")

        # Fetch data for each stock
        start_train = hyperparams['start_train']
        end_train = hyperparams['end_train']
        start_test = hyperparams['start_test']
        end_test = hyperparams['end_test']

        df_train = fetch_data(ticker, start_train, end_train)
        df_test = fetch_data(ticker, start_test, end_test)

        # Split training data by month for original data
        monthly_train_data = split_data_by_month(df_train)

        # Store agents trained for this stock
        stock_agents = []

        # Function to train agent on a given set of monthly data
        def train_agents_on_data(monthly_data):
            for month, df_train_month in monthly_data.items():
                if len(df_train_month) == 0:
                    continue  # Skip months with no data

                print(f"Training agent for {ticker}, month {month}...")

                # Create environment for this month
                env_train = create_trading_env(df_train_month, dynamic_features_arr)

                state_shape = env_train.observation_space.shape
                action_size = env_train.action_space.n

                # Initialize agent
                if agent_type == 'q-learning':
                    agent = QLearningAgent(env_train, None, action_size)
                elif agent_type == 'dqn':
                    agent = DQNAgent(state_shape, action_size)
                else:
                    raise ValueError(f'Invalid agent type: {agent_type}')

                # Train agent
                train_portfolio_values, train_actions = agent.train_agent()

                # Save the last training reward for this agent
                agent.training_reward_last_episode = np.sum(train_portfolio_values)  # Example metric

                train_market_value = df_train_month['close'] * (10000 / df_train_month['close'].iloc[0])

                # Plot results
                plot_performance(train_portfolio_values, None, train_market_value, None,
                                 df_train_month, None, algorithm_name=agent_type.upper(),
                                 n_dynamic_features=len(dynamic_features_arr), ticker=ticker,
                                 start_train=start_train, end_train=end_train, start_test=start_test, end_test=end_test,
                                 train_positions=train_actions)

                # Store the trained agent
                stock_agents.append(agent)

        # Train on original data
        train_agents_on_data(monthly_train_data)

        # Add all agents trained for this stock to the global agent pool
        all_agents.extend(stock_agents)

    # Now test the ensemble of agents from all stocks on each stock
    final_portfolio_returns = []
    final_market_returns = []

    for ticker in tickers:
        print(f"\nTesting ensemble of agents on {ticker}...")

        # Fetch test data for this stock
        df_test = fetch_data(ticker, start_test, end_test)

        # Create test environment
        env_test = create_trading_env(df_test, dynamic_features_arr)

        # Test the ensemble of agents from all stocks on this stock's test data
        test_portfolio_values, ensemble_rewards, test_actions, final_portfolio_return, final_market_return = (
            ensemble_test(all_agents, env_test, agent_type))

        final_portfolio_returns.append(final_portfolio_return)
        final_market_returns.append(final_market_return)

        # Market values for comparison during testing
        test_market_values = df_test['close'] * (10000 / df_test['close'].iloc[0])

        # Plot ensemble testing performance compared to market value
        plot_performance(
            None,  # No individual training values here
            test_portfolio_values,  # Ensemble portfolio values during testing
            None,  # No individual training market values here
            test_market_values,  # Market portfolio values during testing
            None,  # No individual training data
            df_test,  # Test data
            algorithm_name=f'{agent_type.upper()} ENSEMBLE on {ticker}',
            n_dynamic_features=len(dynamic_features_arr),
            ticker=ticker,
            start_train=start_train,
            end_train=end_train,
            start_test=start_test,
            end_test=end_test,
            test_positions=test_actions
        )

    # Plot bar chart of final portfolio returns compared to market returns
    plot_final_returns(final_portfolio_returns, final_market_returns, tickers, agent_type)


def ensemble_test(agents, env_test, agent_type):
    """
    Test the ensemble of agents by interacting with the test environment.
    Applies weighted voting using agents' training_reward_last_episode to avoid bias towards the "buy" action.
    """
    portfolio_values = []
    actions = []

    # Initialize the environment and get the initial state
    state, _ = env_test.reset(seed=42)

    total_reward = 0
    done = False
    initial_portfolio_value = None
    initial_market_value = None  # Store the initial market value (closing price)
    final_market_value = None    # Store the final market value (closing price)

    while not done:
        # Calculate weighted Q-values based on training reward performance
        weighted_q_values = weighted_ensemble_q_values(agents, state)

        # Choose action based on the highest weighted Q-value
        action = np.argmax(weighted_q_values)

        # Take step in environment
        next_state, reward, terminated, truncated, info = env_test.step(action)
        done = terminated or truncated

        current_open_price = info['data_open']  # Get the current day's open price
        current_close_price = info['data_close']  # Get the current day's close price

        # Update portfolio value based on the current day's open price
        portfolio_value = info['portfolio_valuation'] * current_open_price / current_close_price

        # Store the initial portfolio value and market value
        if initial_portfolio_value is None:
            initial_portfolio_value = portfolio_value
            initial_market_value = current_close_price

        # Track the final market value (from the last step)
        final_market_value = current_close_price

        # Update state and tracking variables
        state = next_state
        total_reward += reward
        portfolio_values.append(portfolio_value)
        actions.append(hyperparams['positions'][action])

    # Calculate total portfolio return
    final_portfolio_value = portfolio_values[-1]
    final_portfolio_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100

    # Calculate market return based on the price change
    final_market_return = (final_market_value - initial_market_value) / initial_market_value * 100

    print(f"Ensemble Test Total Reward: {total_reward}")
    print(f"Portfolio Return: {final_portfolio_return:.2f}%")
    print(f"Market Return: {final_market_return:.2f}%")

    return portfolio_values, total_reward, actions, final_portfolio_return, final_market_return


def plot_final_returns(final_portfolio_returns, final_market_returns, tickers, agent_type):
    """
    Plot a bar chart of final portfolio returns compared to market returns for each stock.
    Additionally, display the total portfolio % and total market %.
    """
    import matplotlib.pyplot as plt

    x = np.arange(len(tickers))
    width = 0.35

    # Calculate the total average returns
    total_portfolio_return = np.mean(final_portfolio_returns)
    total_market_return = np.mean(final_market_returns)

    # Print total returns
    print(f"\nTotal Portfolio Return: {total_portfolio_return:.2f}%")
    print(f"Total Market Return: {total_market_return:.2f}%")

    # Create the bar plot
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width / 2, final_portfolio_returns, width, label='Portfolio Return')
    bars2 = ax.bar(x + width / 2, final_market_returns, width, label='Market Return')

    # Add text annotations for total returns on the plot
    ax.text(0.5, 0.95, f'Total Portfolio Return: {total_portfolio_return:.2f}%', transform=ax.transAxes,
            ha='center', va='center', fontsize=10, color='green', weight='bold')
    ax.text(0.5, 0.90, f'Total Market Return: {total_market_return:.2f}%', transform=ax.transAxes,
            ha='center', va='center', fontsize=10, color='blue', weight='bold')

    # Set labels and titles
    ax.set_ylabel('Return (%)')
    ax.set_title(f'Final Portfolio Returns vs. Market Returns ({agent_type.upper()} Ensemble)')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()


# Example usage
main(agent_type='q-learning')





