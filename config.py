# config.py
from sympy.physics.units import years

import dynamic_features
import numpy as np

INTERVAL_SIZE = 18
hyperparams = {
    'algorithm': 'dqn_gru',  # q-learning, dqn, dqn_gru, dqn_lstm
    'interval_days': INTERVAL_SIZE,
    'ppo_timestamps':1,
    'show_buy_sell_signals': False,

    'start_year': '2022-01-01',
    'end_year': None,

    'retrain': False, # For interval training

    'initial_position': 0,  # Initial position (1=long, -1=short, 0=cash)  ,

    'seed': 42,
    'n_episodes':1,
    'positions': [0,1],
    'discount_factor': 0.995,
    'exploration_rate': 1,
    'exploration_decay': 0.995,
    'exploration_min': 0.01,
    'trading_fees': 0.01,
    'portfolio_initial_value': 10000,
    'windows': 1,
    'verbose': 1,
    'learning_rate': 0.00001,

    # QN-specific parameters
    'num_bins':2,

    # DQN-specific parameters
    'hidden_layer_size': 128,
    'lstm_hidden_size': 128,
    'lstm_num_layers': 80,
    'memory_size': 1000,
    'batch_size': 16,
}



def get_discrete_value(value, bins):
    return np.digitize(value, bins)

if hyperparams['algorithm'] == 'q-learning':
    dynamic_features_arr = [
        lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=3),
                                           bins=[-0.1, -0.05, 0, 0.05, 0.1]),

        lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=7),
                                           bins=[-0.1, -0.05, 0, 0.05, 0.1]),

        lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=15),
                                             bins=[-0.1, -0.05, 0, 0.05, 0.1]),

        lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=30),
                                                bins=[-0.1, -0.05, 0, 0.05, 0.1]),
    ]

else:


    # Non-discrete features for DQN
    dynamic_features_arr = [
        #High-Low Range (covering more granular timeframes)

        # Price Change (covering more granular short, medium, and long-term windows)
        lambda history: dynamic_features.dynamic_feature_price_change(history, window=INTERVAL_SIZE//3),
        lambda history: dynamic_features.dynamic_feature_price_change(history, window=INTERVAL_SIZE//2),
        lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE),

        # Momentum (capturing trends over more granular timeframes)
        lambda history: dynamic_features.dynamic_feature_momentum(history, window=INTERVAL_SIZE//3),
        lambda history: dynamic_features.dynamic_feature_momentum(history, window=INTERVAL_SIZE//2),
        lambda history: dynamic_features.dynamic_feature_momentum(history, window=INTERVAL_SIZE),


        # ATR (Average True Range over more windows)
        lambda history: dynamic_features.dynamic_feature_atr(history, window=INTERVAL_SIZE//3),
        lambda history: dynamic_features.dynamic_feature_atr(history, window=INTERVAL_SIZE//2),
        lambda history: dynamic_features.dynamic_feature_atr(history, window=INTERVAL_SIZE),

        # Bollinger Band Width (adding more intermediate windows)
        lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=INTERVAL_SIZE//3),
        lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=INTERVAL_SIZE//2),
        lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=INTERVAL_SIZE),

        # Stochastic Oscillator (denser windows)
        lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=INTERVAL_SIZE//3),
        lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=INTERVAL_SIZE//2),
        lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=INTERVAL_SIZE),

        # RSI (more windows for trend strength across different timeframes)
        lambda history: dynamic_features.dynamic_feature_rsi(history, window=INTERVAL_SIZE//3),
        lambda history: dynamic_features.dynamic_feature_rsi(history, window=INTERVAL_SIZE//2),
        lambda history: dynamic_features.dynamic_feature_rsi(history, window=INTERVAL_SIZE),


        # Price Difference (more granular windows)
        lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE//3),
        lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE//2),
        lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE),

        # Volume Moving Average (denser timeline for volume trends)
        lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=INTERVAL_SIZE//3),
        lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=INTERVAL_SIZE//2),
        lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=INTERVAL_SIZE),

        # Exponential Moving Average (EMA with more windows)
        lambda history: dynamic_features.dynamic_feature_ema(history, window=INTERVAL_SIZE//3),
        lambda history: dynamic_features.dynamic_feature_ema(history, window=INTERVAL_SIZE//2),
        lambda history: dynamic_features.dynamic_feature_ema(history, window=INTERVAL_SIZE),
    ]



