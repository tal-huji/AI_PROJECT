# env_utils.py
from gym_trading_env.environments import TradingEnv
from config import hyperparams

# Custom reward function
# def custom_reward_function(history):
#     if len(history['data_close']) < 2:
#         return 0.0
#
#     current_return = (history['data_close', -1] / history['data_close', -2]) - 1
#     if len(history['data_close']) >= 30:
#         market_return = (history['data_close', -1] / history['data_close', -30]) - 1
#         outperformance = current_return - market_return
#     else:
#         outperformance = 0.0
#
#     reward = current_return + 0.5 * outperformance
#     return reward

# Create trading environment
def create_trading_env(df, dynamic_features, initial_portfolio_value=None, initial_position=None):
    """
    Creates the trading environment, propagating the initial portfolio value if provided.
    """
    trading_fees = hyperparams['trading_fees']
    windows = hyperparams['windows']
    verbose = hyperparams['verbose']

    # Use the provided initial_portfolio_value, otherwise default to hyperparams' value
    if initial_portfolio_value is None:
        initial_portfolio_value = hyperparams['portfolio_initial_value']

    if initial_position is None:
        initial_position = hyperparams['initial_position']

    env = TradingEnv(
        df=df,
        positions=hyperparams['positions'],
        trading_fees=trading_fees,
        portfolio_initial_value=initial_portfolio_value,  # Pass the propagated value here
        windows=windows,
        verbose=verbose,
        dynamic_feature_functions=dynamic_features,
        initial_position=initial_position,
        # reward_function=custom_reward_function
    )

    return env

