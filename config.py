# config.py
from sympy.physics.units import years

import dynamic_features
import numpy as np

hyperparams = {
    'algorithm': 'dqn_gru',  # q-learning, dqn, dqn_gru, dqn_lstm
    'interval_days': 60,

    'start_year': '2022-01-01',
    'end_year': None,

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
    'learning_rate': 0.0001,

    # QN-specific parameters
    'num_bins':2,

    # DQN-specific parameters
    'hidden_layer_size': 128,
    'lstm_hidden_size': 128,
    'lstm_num_layers': 40,
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
        # lambda history: dynamic_features.dynamic_feature_price_change(history, window=4),
        lambda history: dynamic_features.dynamic_feature_price_change(history, window=7),
        lambda history: dynamic_features.dynamic_feature_price_change(history, window=10),
        lambda history: dynamic_features.dynamic_feature_price_change(history, window=15),
        lambda history: dynamic_features.dynamic_feature_price_change(history, window=30),
        lambda history: dynamic_features.dynamic_feature_price_change(history, window=60),


        # Momentum (capturing trends over more granular timeframes)
        # lambda history: dynamic_features.dynamic_feature_momentum(history, window=4),
        lambda history: dynamic_features.dynamic_feature_momentum(history, window=7),
        lambda history: dynamic_features.dynamic_feature_momentum(history, window=10),
        lambda history: dynamic_features.dynamic_feature_momentum(history, window=15),
        lambda history: dynamic_features.dynamic_feature_momentum(history, window=30),
        lambda history: dynamic_features.dynamic_feature_momentum(history, window=60),


        # ATR (Average True Range over more windows)
        # lambda history: dynamic_features.dynamic_feature_atr(history, window=4),
        lambda history: dynamic_features.dynamic_feature_atr(history, window=7),
        lambda history: dynamic_features.dynamic_feature_atr(history, window=10),
        lambda history: dynamic_features.dynamic_feature_atr(history, window=15),
        lambda history: dynamic_features.dynamic_feature_atr(history, window=30),
        lambda history: dynamic_features.dynamic_feature_atr(history, window=60),


        # Bollinger Band Width (adding more intermediate windows)
        # lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=4),
        lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=7),
        lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=10),
        lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=15),
        lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=30),
        lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=60),


        # Stochastic Oscillator (denser windows)
        # lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=4),
        lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=7),
        lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=10),
        lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=15),
        lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=30),
        lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=60),


        # RSI (more windows for trend strength across different timeframes)
        # lambda history: dynamic_features.dynamic_feature_rsi(history, window=4),
        lambda history: dynamic_features.dynamic_feature_rsi(history, window=7),
        lambda history: dynamic_features.dynamic_feature_rsi(history, window=10),
        lambda history: dynamic_features.dynamic_feature_rsi(history, window=15),
        lambda history: dynamic_features.dynamic_feature_rsi(history, window=30),
        lambda history: dynamic_features.dynamic_feature_rsi(history, window=60),



        # Price Difference (more granular windows)
        # lambda history: dynamic_features.dynamic_feature_price_diff(history, window=4),
        lambda history: dynamic_features.dynamic_feature_price_diff(history, window=7),
        lambda history: dynamic_features.dynamic_feature_price_diff(history, window=10),
        lambda history: dynamic_features.dynamic_feature_price_diff(history, window=15),
        lambda history: dynamic_features.dynamic_feature_price_diff(history, window=30),
        lambda history: dynamic_features.dynamic_feature_price_diff(history, window=60),


        # Volume Moving Average (denser timeline for volume trends)
        # lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=4),
        lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=7),
        lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=10),
        lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=15),
        lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=30),
        lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=60),


        # Exponential Moving Average (EMA with more windows)
        # lambda history: dynamic_features.dynamic_feature_ema(history, window=4),
        lambda history: dynamic_features.dynamic_feature_ema(history, window=7),
        lambda history: dynamic_features.dynamic_feature_ema(history, window=10),
        lambda history: dynamic_features.dynamic_feature_ema(history, window=15),
        lambda history: dynamic_features.dynamic_feature_ema(history, window=30),
        lambda history: dynamic_features.dynamic_feature_ema(history, window=60),
    ]



