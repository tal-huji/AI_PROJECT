# config.py
import dynamic_features
import numpy as np

hyperparams = {
    # General parameters
    'start_train': '2023-01-01',
    'end_train': '2024-01-01',
    'start_test': '2024-01-01',
    'end_test':  '2024-02-01',

    'initial_position': 1,  # Initial position (1=long, -1=short, 0=cash)  ,

    # Common parameters for both QL and DQN
    'seed': 42,  # Random seed for reproducibility (used in both QN and DQN)
    'n_episodes': 100,  # Number of episodes for training (both QN and DQN)
    'positions': [-1,0,1],  # Available actions (hold, sell, buy) - used in both QN and DQN
    'discount_factor': 0,  # Discount factor (gamma) for future rewards (both QN and DQN)
    'exploration_rate': 0,  # Starting exploration rate (epsilon) for both QN and DQN
    'exploration_decay': 0.995,  # Decay rate of exploration (both QN and DQN)
    'exploration_min': 0.01,  # Minimum exploration rate (both QN and DQN)
    'trading_fees': 0.01,  # Trading fees applied on each trade (both QN and DQN)
    'portfolio_initial_value': 10000,  # Initial portfolio value for both QN and DQN
    'windows': 1,
    'verbose': 1,  # Verbosity level (both QN and DQN)
    'learning_rate': 0.001,  # Learning rate for updating Q-values (used in both QN and DQN)

    # QN-specific parameters
    'num_bins':2,

    # DQN-specific parameters
    'hidden_layer_size': 64,  # DQN-specific: Size of hidden layers in the neural network
    'memory_size': 10000,  # DQN-specific: Size of the experience replay memory
    'batch_size': 16,  # DQN-specific: Batch size for training the neural network
}


def get_discrete_value(value, bins):
    return np.digitize(value, bins)

dynamic_features_arr = [
    lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=5),
                                       bins=[-0.1, -0.05, 0, 0.05, 0.1]),

    lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=10),
                                       bins=[-0.1, -0.05, 0, 0.05, 0.1]),

    lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=14),
                                        bins=[-0.1, -0.05, 0, 0.05, 0.1]),

    # lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=14),
    #                                     bins=[-0.06, 0.06]),


    # lambda history: get_discrete_value(dynamic_features.dynamic_feature_bollinger_bands(history, window=7)[0],
    #                                    bins=[-1, 0, 1]),
    #
    # lambda history: get_discrete_value(dynamic_features.dynamic_feature_bollinger_bands(history, window=7)[1],
    #                                     bins=[-1, 0, 1]),
    #
    # lambda history: get_discrete_value(dynamic_features.dynamic_feature_bollinger_bands(history, window=7)[2],
    #                                     bins=[-1, 0, 1]),

]

