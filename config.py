# config.py
import dynamic_features
import numpy as np

hyperparams = {
    'algorithm': 'q-learning',  # q-learning, dqn_gru, dqn_lstm
    'train_interval_days': 10,
    'test_interval_days': 10,


    'start_year': '2024-01-01',
    'end_year':None,

    'initial_position': 1,  # Initial position (1=long, -1=short, 0=cash)  ,

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
    'learning_rate': 0.01,

    # QN-specific parameters
    'num_bins':2,

    # DQN-specific parameters
    'hidden_layer_size': 3,
    'lstm_hidden_size': 3,
    'lstm_num_layers': 1,
    'memory_size': 10000,
    'batch_size': 16,
}


def get_discrete_value(value, bins):
    return np.digitize(value, bins)

if hyperparams['algorithm'] == 'q-learning':
    dynamic_features_arr = [
        lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=3),
                                           bins=[-0.1, -0.05, 0, 0.05, 0.1]),

        lambda history: get_discrete_value(dynamic_features.dynamic_feature_price_change(history, window=7),
                                           bins=[-0.1, -0.05, 0, 0.05, 0.1])
    ]

else:
    #Non-discrete features for DQN
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



