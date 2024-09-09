# config.py
import dynamic_features
import numpy as np

hyperparams = {
    # General parameters
    # 'start_train': '2023-07-01',
    # 'end_train': '2023-07-14',
    #
    # 'start_test': '2023-07-14',
    # 'end_test':   '2023-08-01',
    'start_year': '2024-01-01',
    'end_year':None,

    'initial_position': 1,  # Initial position (1=long, -1=short, 0=cash)  ,
    # Common parameters for both QL and DQN
    'seed': 42,  # Random seed for reproducibility (used in both QN and DQN)
    'n_episodes':1,
    'positions': [0,1], # Possible positions (both QN and DQN)
    'discount_factor': 0.995,  # Discount factor (gamma) for future rewards (both QN and DQN)
    'exploration_rate': 1,  # Starting exploration rate (epsilon) for both QN and DQN
    'exploration_decay': 0.995,  # Decay rate of exploration (both QN and DQN)
    'exploration_min': 0.01,  # Minimum exploration rate (both QN and DQN)
    'trading_fees': 0.01,  # Trading fees applied on each trade (both QN and DQN)
    'portfolio_initial_value': 10000,  # Initial portfolio value for both QN and DQN
    'windows': 1,
    'verbose': 1,  # Verbosity level (both QN and DQN)
    'learning_rate': 0.01,  # Learning rate for updating Q-values (used in both QN and DQN)

    # QN-specific parameters
    'num_bins':2,

    # DQN-specific parameters
    'hidden_layer_size': 3,  # DQN-specific: Size of hidden layers in the neural network
    'lstm_hidden_size': 3,  # DQN-specific: Size of hidden layers in the LSTM network
    'lstm_num_layers': 1,  # DQN-specific: Size of hidden layers in the LSTM network
    'memory_size': 10000,  # DQN-specific: Size of the experience replay memory
    'batch_size': 16,  # DQN-specific: Batch size for training the neural network
}


def get_discrete_value(value, bins):
    return np.digitize(value, bins)

# Discrete features for QL
# dynamic_features_arr = [
#     lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=3),
#                                        bins=[-0.1,-0.05, 0, 0.05, 0.1]),
#
#     lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=7),
#                                        bins=[-0.1,-0.05, 0, 0.05, 0.1])
# ]

# Non-discrete features for DQN - Year Split
dynamic_features_arr = [
    # Price Change (covering more granular short, medium, and long-term windows)
    lambda history: dynamic_features.dynamic_feature_price_change(history, window=4),
    lambda history: dynamic_features.dynamic_feature_price_change(history, window=7),
    lambda history: dynamic_features.dynamic_feature_price_change(history, window=10),


    # Momentum (capturing trends over more granular timeframes)
    lambda history: dynamic_features.dynamic_feature_momentum(history, window=4),
    lambda history: dynamic_features.dynamic_feature_momentum(history, window=7),
    lambda history: dynamic_features.dynamic_feature_momentum(history, window=10),


    # ATR (Average True Range over more windows)
    lambda history: dynamic_features.dynamic_feature_atr(history, window=4),
    lambda history: dynamic_features.dynamic_feature_atr(history, window=7),
    lambda history: dynamic_features.dynamic_feature_atr(history, window=10),


    # Bollinger Band Width (adding more intermediate windows)
    lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=4),
    lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=7),
    lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=10),


    # Stochastic Oscillator (denser windows)
    lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=4),
    lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=7),
    lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=10),


    # RSI (more windows for trend strength across different timeframes)
    lambda history: dynamic_features.dynamic_feature_rsi(history, window=4),
    lambda history: dynamic_features.dynamic_feature_rsi(history, window=7),
    lambda history: dynamic_features.dynamic_feature_rsi(history, window=10),


    # Price Difference (more granular windows)
    lambda history: dynamic_features.dynamic_feature_price_diff(history, window=4),
    lambda history: dynamic_features.dynamic_feature_price_diff(history, window=7),
    lambda history: dynamic_features.dynamic_feature_price_diff(history, window=10),


    # Volume Moving Average (denser timeline for volume trends)
    lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=4),
    lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=7),
    lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=10),


    # Exponential Moving Average (EMA with more windows)
    lambda history: dynamic_features.dynamic_feature_ema(history, window=4),
    lambda history: dynamic_features.dynamic_feature_ema(history, window=7),
    lambda history: dynamic_features.dynamic_feature_ema(history, window=10),
]



