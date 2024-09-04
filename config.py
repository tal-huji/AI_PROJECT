# config.py
import dynamic_features

hyperparams = {
    # Common parameters for both QN and DQN
    'seed': 42,  # Random seed for reproducibility (used in both QN and DQN)
    'n_episodes': 1000000,  # Number of episodes for training (both QN and DQN)
    'positions': [-1, 0, 1],  # Available actions (hold, sell, buy) - used in both QN and DQN
    'discount_factor': 0.99,  # Discount factor (gamma) for future rewards (both QN and DQN)
    'exploration_rate': 1.0,  # Starting exploration rate (epsilon) for both QN and DQN
    'exploration_decay': 0.995,  # Decay rate of exploration (both QN and DQN)
    'exploration_min': 0.01,  # Minimum exploration rate (both QN and DQN)
    'trading_fees': 0.01,  # Trading fees applied on each trade (both QN and DQN)
    'portfolio_initial_value': 10000,  # Initial portfolio value for both QN and DQN
    'windows': 1,  # Window size for calculating features (both QN and DQN)
    'verbose': 1,  # Verbosity level (both QN and DQN)
    'learning_rate': 0.01,  # Learning rate for updating Q-values (used in both QN and DQN)

    # QN-specific parameters
    'num_bins': 20,  # Number of bins for discretization (QN-specific)

    # DQN-specific parameters
    'hidden_layer_size': 64,  # DQN-specific: Size of hidden layers in the neural network
    'memory_size': 10000,  # DQN-specific: Size of the experience replay memory
    'batch_size': 16,  # DQN-specific: Batch size for training the neural network
}

dynamic_features_arr = [
    # dynamic_features.dynamic_feature_last_position_taken,
    # dynamic_features.dynamic_feature_real_position,
    # Adjusted Close SMA Ratios
    lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=20),
    lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=50),
    # lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=200),

    # RSI with different windows
    # lambda history: dynamic_features.dynamic_feature_rsi(history, window=14),
    # lambda history: dynamic_features.dynamic_feature_rsi(history, window=7),
    # lambda history: dynamic_features.dynamic_feature_rsi(history, window=21),
    #
    # # MACD
    # lambda history: dynamic_features.dynamic_feature_macd(history),
    #
    # # Bollinger Bands (middle, upper, lower)
    # lambda history: dynamic_features.dynamic_feature_bollinger_bands(history, window=20, num_std=2)[0],
    # lambda history: dynamic_features.dynamic_feature_bollinger_bands(history, window=20, num_std=2)[1],
    # lambda history: dynamic_features.dynamic_feature_bollinger_bands(history, window=20, num_std=2)[2],
    #
    # # On-Balance Volume (OBV)
    # lambda history: dynamic_features.dynamic_feature_obv(history),
    #
    # # Average Directional Index (ADX)
    # lambda history: dynamic_features.dynamic_feature_adx(history, window=14),
    #
    # # New Features
    # lambda history: dynamic_features.dynamic_feature_ema(history, window=20),
    # lambda history: dynamic_features.dynamic_feature_ema(history, window=50),
    # lambda history: dynamic_features.dynamic_feature_ema(history, window=100),
    # lambda history: dynamic_features.dynamic_feature_ema(history, window=200),
    #
    # lambda history: dynamic_features.dynamic_feature_roc(history, window=10),
    # lambda history: dynamic_features.dynamic_feature_roc(history, window=20),
    # lambda history: dynamic_features.dynamic_feature_roc(history, window=50),
    #
    # lambda history: dynamic_features.dynamic_feature_price_diff(history, window=1),
    # lambda history: dynamic_features.dynamic_feature_price_diff(history, window=5),
    # lambda history: dynamic_features.dynamic_feature_price_diff(history, window=10),
    #
    # lambda history: dynamic_features.dynamic_feature_daily_return(history),
    #
    # lambda history: dynamic_features.dynamic_feature_high_low_range(history),
    #
    # lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=20),
    # lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=50),
    #
    # lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=20, num_std=2),
    #
    # lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=14),
    #
    # lambda history: dynamic_features.dynamic_feature_atr(history, window=14),
    #
    # lambda history: dynamic_features.dynamic_feature_momentum(history, window=10),
    # lambda history: dynamic_features.dynamic_feature_momentum(history, window=20),
    #
    # lambda history: dynamic_features.dynamic_feature_skewness(history, window=30),
    # lambda history: dynamic_features.dynamic_feature_kurtosis(history, window=30),
    #
    # lambda history: dynamic_features.dynamic_feature_zscore(history, window=20),
    #
    # lambda history: dynamic_features.dynamic_feature_mfi(history, window=14),
]