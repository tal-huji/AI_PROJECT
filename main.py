# main.py
from models.qn import QLearningAgent
from utils import fetch_data, plot_performance, set_all_seeds, compute_min_max_for_features
from env_utils import create_trading_env
from models.dqn import DQNAgent
from config import hyperparams
from config import dynamic_features_arr
import numpy as np


def main(agent_type='q-learning'):
    print(f"Running {agent_type.upper()} agent...")
    set_all_seeds(hyperparams['seed'])

    # Fetch data
    ticker = 'INTC'
    start_train = '2023-01-01'
    end_train = '2024-01-01'
    start_test = '2024-01-01'
    end_test = None

    df_train = fetch_data(ticker, start_train, end_train)
    df_test = fetch_data(ticker, start_test, end_test)

    # Create environments
    env_train = create_trading_env(df_train, dynamic_features_arr)
    env_test = create_trading_env(df_test, dynamic_features_arr)

    # Initialize agent
    state_shape = env_train.observation_space.shape
    action_size = env_train.action_space.n

    # Initialize Q-Learning agent
    if agent_type == 'q-learning':
        min_max_dict = compute_min_max_for_features(df_train, dynamic_features_arr)
        agent = QLearningAgent(env_train, env_test, action_size, min_max_dict)
    elif agent_type == 'dqn':
        agent = DQNAgent(state_shape, action_size)
    else:
        raise ValueError(f'Invalid agent type: {agent_type}')

    # Train agent
    train_portfolio_values = agent.train_agent()

    # Test agent
    test_reward, test_portfolio_values = agent.test_agent()

    # Market values for comparison
    train_market_values = df_train['close'] * (10000 / df_train['close'].iloc[0])
    test_market_values = df_test['close'] * (10000 / df_test['close'].iloc[0])

    # Plot results
    plot_performance(train_portfolio_values, test_portfolio_values, train_market_values, test_market_values,
                     df_train, df_test, algorithm_name=agent_type.upper(),
                     n_dynamic_features=len(dynamic_features_arr), ticker=ticker,
                     start_train=start_train, end_train=end_train, start_test=start_test, end_test=end_test,)


# Example usage
main(agent_type='q-learning')
