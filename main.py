import math
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from device import device
from models.policy_gradient import PolicyGradientAgent
from models.policy_gradient_gru import PolicyGradientAgent_GRU
from utils import fetch_data, plot_yearly_performance, plot_final_results
from models.dqn_gru import DQNAgent_GRU
from models.ql import QLearningAgent, get_hashasble_state
from models.dqn import DQNAgent
from env_utils import create_trading_env
from config import hyperparams, dynamic_features_arr
from stable_baselines3 import PPO
import models.simple_agnet as simple_agent


def get_last_valid_value(values_list):
    """
    Helper function to get the last valid (non-None) value from a list.
    """
    for value in reversed(values_list):
        if value is not None:
            return value
    return 0  # Return 0 if no valid value is found


def main(agent_type, interval_days, retrain, baseline):
    print(f"Start training with hyperparameters: {hyperparams}")
    print(f"Number of dynamic features: {len(dynamic_features_arr)}")

    #set_all_seeds(hyperparams['seed'])

    tickers = [
        'NVDA',
        'QCOM',
        'ADBE',
        'CRM',
        'INTC',

        'TSLA',
        'GOOGL',
        'NFLX',
        'AAPL',
        'MSFT',


        # 'ORCL',
        # 'AMZN',
        # 'META',
        # 'AMD',
        # 'CSCO',
        # 'IBM',
    ]

    final_results = []
    initial_position = hyperparams['initial_position']
    final_results_baseline = []

    # Initialize agent only once if retraining is True
    agent = None

    for ticker in tickers:
        print(f"\nTraining and testing for {ticker}...")

        start_year = hyperparams['start_year']
        end_year = hyperparams['end_year']

        # Fetch stock data
        df_full_year = fetch_data(ticker, start_year, end_year).sort_index()

        # Interval-based approach for other agents
        n_days = len(df_full_year)

        # Calculate the highest multiple of interval_days that fits into the total number of trading days
        max_n_intervals = math.floor(n_days / interval_days)
        adjusted_n_days = max_n_intervals * interval_days - 1  # This is the adjusted number of days
        trading_dates = df_full_year.index[:adjusted_n_days]

        df_dates = df_full_year.loc[df_full_year.index.isin(trading_dates)]

        print("Testing baseline...")
        portfolio_dict_baseline, \
         buy_hold_market_dict_baseline, \
         test_actions_dict_baseline,\
         baseline_results \
            = train_and_test_baseline(df_dates, ticker, interval_days, final_results_baseline, trading_dates)


        portfolio_dict = {date: None for date in trading_dates}
        buy_hold_market_dict = {date: None for date in trading_dates}
        test_actions_dict = {date: None for date in trading_dates}

        # Loop over train/test intervals, using the adjusted number of trading days
        i = 0
        initial_close_price = None
        initial_portfolio_value = None

        while i + interval_days <= adjusted_n_days:
            # Define the training set with overlap: add the last day from the previous test
            df_train = df_dates.iloc[i:i + interval_days]

            # Define the test set: start from the last day of the train and extend for the test interval
            end_test = i + interval_days + interval_days
            df_test = df_dates.iloc[i + interval_days - 1: end_test]

            if initial_portfolio_value is None:
                initial_portfolio_value = hyperparams['portfolio_initial_value']

            if initial_close_price is None:
                initial_close_price = df_test['close'].iloc[0]

            # Pass the propagated initial_portfolio_value when creating env_train and env_test
            env_train = create_trading_env(df_train, dynamic_features_arr, initial_portfolio_value)
            env_test = create_trading_env(df_test, dynamic_features_arr, initial_portfolio_value)

            state_shape = env_train.observation_space.shape
            action_size = env_train.action_space.n

            # If retraining is False, initialize a new agent for each ticker.
            if not retrain or agent is None:
                agent = initialize_agent(agent_type, env_train, state_shape, action_size)

            agent.train_agent()
            test_portfolio_values, test_actions, final_portfolio_return, final_market_return = test_agent(agent, env_test, agent_type)

            # After testing, update the initial_portfolio_value with the final portfolio value
            initial_portfolio_value = test_portfolio_values[-1] if test_portfolio_values else initial_portfolio_value
            initial_position = test_actions[-1] if test_actions else initial_position

            # Map test results to dates
            map_test_results_to_dates(
                df_test,
                test_portfolio_values,
                test_actions,
                portfolio_dict,
                buy_hold_market_dict,
                test_actions_dict,
                initial_close_price
            )

            i += interval_days

        # Plot yearly performance
        plot_yearly_performance(
            ticker,
            df_full_year,
            portfolio_dict,
            buy_hold_market_dict,
            test_actions_dict,
            portfolio_dict_baseline,
            buy_hold_market_dict_baseline,
            test_actions_dict_baseline,
        )
        #
        # final_portfolio_result = portfolio_dict.get(trading_dates[-1], None)
        # final_buy_hold_result = buy_hold_market_dict.get(trading_dates[-1], None)

        # Find last none None value
        final_portfolio_result = get_last_valid_value(portfolio_dict.values())
        final_buy_hold_result = get_last_valid_value(buy_hold_market_dict.values())

        final_results.append({
            'ticker': ticker,
            'final_portfolio_value': final_portfolio_result,
            'final_buy_hold': final_buy_hold_result
        })



    # Plot final results across all tickers
    plot_final_results(final_results, final_results_baseline)


def test_ppo(agent, env_test):
    """
    Test a single agent across the entire dataset without breaking it into intervals.
    """
    obs = env_test.reset()
    done = False
    total_rewards = 0
    portfolio_values = []
    actions = []

    while not done:
        action, _ = agent.predict(obs)  # PPO uses predict method to get action
        obs, rewards, done, info = env_test.step(action)
        total_rewards += rewards

        # Store portfolio values and actions for plotting
        portfolio_values.append(info[0]['portfolio_valuation'])
        actions.append(action[0])  # Actions will be in an array, so we take action[0]

    # Return the portfolio values and actions to be used for plotting
    return portfolio_values, actions




def initialize_agent(agent_type, env_train, state_shape, action_size):
    if agent_type == 'q-learning':
        return QLearningAgent(env_train, None, action_size)
    elif agent_type == 'dqn':
        return DQNAgent(env_train, state_shape, action_size)
    elif agent_type == 'dqn_gru':
        return DQNAgent_GRU(env_train, state_shape, action_size)
    elif agent_type =='policy_gradient':
        return PolicyGradientAgent(env_train, state_shape, action_size)
    elif agent_type == 'policy_gradient_gru':
        return PolicyGradientAgent_GRU(env_train, state_shape, action_size)
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

    if agent != 'simple':
     agent.exploration_rate = 0  # Set exploration rate to 0 for testing

    initial_portfolio_value = hyperparams['portfolio_initial_value']
    initial_market_value, final_market_value = None, None

    if agent_type in ['dqn', 'dqn_lstm', 'dqn_gru']:
        agent.model.eval()
        agent.target_model.eval()

    if agent_type in ['policy_gradient', 'policy_gradient_gru']:
        agent.model.eval()

    while not done:
        if agent_type != 'simple':
            q_values = select_action(agent, agent_type, state)
            action = np.argmax(q_values)
        else:
            action = agent.choose_action(state)

        next_state, reward, terminated, truncated, info = env_test.step(action)
        done = terminated or truncated

        portfolio_value = info['portfolio_valuation']

        current_close_price = info.get('data_close', 0)


        initial_market_value = initial_market_value or current_close_price
        final_market_value = current_close_price

        state = next_state
        total_reward += reward
        portfolio_values.append(portfolio_value)
        actions.append(hyperparams['positions'][action])

    final_portfolio_value = portfolio_values[-1] if portfolio_values else 0
    final_portfolio_return = ((final_portfolio_value - initial_portfolio_value) / initial_portfolio_value) * 100 if initial_portfolio_value else 0
    final_market_return = ((final_market_value - initial_market_value) / initial_market_value) * 100 if initial_market_value else 0

    return portfolio_values, actions, final_portfolio_return, final_market_return


def select_action(agent, agent_type, state):
    if agent_type in ['dqn', 'dqn_lstm', 'dqn_gru']:
        agent.normalizer.update(state)
        normalized_state = agent.normalizer.normalize(state)
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
        with torch.no_grad():
            if agent_type == 'dqn':
                values_vec = agent.model(state_tensor).detach().cpu().numpy()[0]
            else:
                values_vec, _ = agent.model(state_tensor)
                values_vec = values_vec.detach().cpu().numpy()[0]

    elif 'policy_gradient' in agent_type:
        if agent_type == 'policy_gradient':
            values_vec = agent.get_action_probabilities(state)
        else:
            values_vec, _ = agent.get_action_probabilities(state, hidden_state=None)

    elif agent_type == 'q-learning':
        hashable = get_hashasble_state(state)
        values_vec = agent.q_table[hashable]

    else:
        raise ValueError(f'Invalid agent type: {agent_type}')

    return values_vec


def train_and_test_baseline(df_dates, ticker, interval_days, final_results, trading_dates):
    if hyperparams['baseline_algorithm'] == 'ppo':
        return train_and_test_ppo(df_dates, ticker, interval_days, final_results, trading_dates)

    elif hyperparams['baseline_algorithm'] == 'simple':
        return train_and_test_simple(df_dates, ticker, interval_days, final_results, trading_dates)

def initialize_baseline_agent(baseline, env_train, state_shape, action_size):
    if baseline == 'ppo':
        return PPO(
        'MlpPolicy',
        env_train,
        n_steps=100,
        learning_rate=hyperparams['learning_rate'],
        batch_size=4,
        verbose=1,

    )
    elif baseline == 'simple':
        return simple_agent.SimpleAgent(env_train, state_shape, action_size)
    else:
        raise ValueError(f'Invalid baseline algorithm: {baseline}')

def train_and_test_simple(df_dates, ticker, interval_days, final_results, trading_dates):
    df_simple_test = df_dates.iloc[interval_days:]
    df_simple_train = fetch_data(ticker, hyperparams['ppo_start_train'], df_simple_test.index[0]).sort_index()

    env_train_simple = create_trading_env(df_simple_train, dynamic_features_arr, hyperparams['portfolio_initial_value'])
    env_train_simple = DummyVecEnv([lambda: env_train_simple])
    env_train_simple.reset()

    # Initialize Simple agent
    agent = simple_agent.SimpleAgent(env_train_simple, env_train_simple.observation_space.shape, env_train_simple.action_space.n)

    # Train Simple on the Simple training dataset
    agent.train_agent()

    # Create the test environment (same period as interval method)
    env_test_simple = create_trading_env(df_simple_test, dynamic_features_arr, hyperparams['portfolio_initial_value'])

    # Test Simple on the test period
    test_portfolio_values, test_actions, final_portfolio_return, final_market_return = test_agent(
        agent, env_test_simple, hyperparams['baseline_algorithm'])

    portfolio_dict = {date: None for date in df_dates.index}
    buy_hold_market_dict = {date: None for date in df_dates.index}
    test_actions_dict = {date: None for date in df_dates.index}

    initial_test_close_price = df_simple_test['close'].iloc[0]

    map_test_results_to_dates(
        df_simple_test,
        test_portfolio_values,
        test_actions,
        portfolio_dict,
        buy_hold_market_dict,
        test_actions_dict,
        initial_test_close_price
    )

    final_portfolio_result = get_last_valid_value(portfolio_dict.values())
    final_buy_hold_result = get_last_valid_value(buy_hold_market_dict.values())

    final_results.append({
        'ticker': ticker,
        'final_portfolio_value': final_portfolio_result,
        'final_buy_hold': final_buy_hold_result
    })

    return portfolio_dict, buy_hold_market_dict, test_actions_dict, final_results

def train_and_test_ppo(df_dates, ticker, interval_days, final_results, trading_dates):
    df_ppo_test = df_dates.iloc[interval_days:]
    df_ppo_train = fetch_data(ticker, hyperparams['ppo_start_train'], df_ppo_test.index[0]).sort_index()

    env_train_ppo = create_trading_env(df_ppo_train, dynamic_features_arr, hyperparams['portfolio_initial_value'])
    env_train_ppo = DummyVecEnv([lambda: env_train_ppo])
    env_train_ppo.reset()

    # Initialize PPO agent
    agent = PPO(
        'MlpPolicy',
        env_train_ppo,
        n_steps=100,
        learning_rate=hyperparams['learning_rate'],
        batch_size=4,
        verbose=1,

    )

    # Train PPO on the PPO training dataset
    agent.learn(total_timesteps=hyperparams['ppo_timestamps'])

    # Create the test environment (same period as interval method)
    env_test_ppo = create_trading_env(df_ppo_test, dynamic_features_arr, hyperparams['portfolio_initial_value'])
    env_test_ppo = DummyVecEnv([lambda: env_test_ppo])
    env_test_ppo.reset()

    # Test PPO on the test period
    test_portfolio_values, test_actions = test_ppo(agent, env_test_ppo)

    portfolio_dict = {date: None for date in df_dates.index}
    buy_hold_market_dict = {date: None for date in df_dates.index}
    test_actions_dict = {date: None for date in df_dates.index}

    initial_test_close_price = df_ppo_test['close'].iloc[0]

    map_test_results_to_dates(
        df_ppo_test,
        test_portfolio_values,
        test_actions,
        portfolio_dict,
        buy_hold_market_dict,
        test_actions_dict,
        initial_test_close_price
    )

    final_portfolio_result = get_last_valid_value(portfolio_dict.values())
    final_buy_hold_result = get_last_valid_value(buy_hold_market_dict.values())

    final_results.append({
        'ticker': ticker,
        'final_portfolio_value': final_portfolio_result,
        'final_buy_hold': final_buy_hold_result
    })

    return portfolio_dict, buy_hold_market_dict, test_actions_dict, final_results


# Example usage
main(
    agent_type=hyperparams['algorithm'],
    interval_days=hyperparams['interval_days'],
    retrain=hyperparams['retrain'],
    baseline = hyperparams['baseline_algorithm']
)
