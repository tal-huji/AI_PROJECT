
import dynamic_features
import numpy as np

"""
  Possible Algorithms:
  q-learning,
  dqn,
  dqn_gru,
  dqn_cnn,
  dqn_gru_cnn,
  policy_gradient,
  policy_gradient_gru,
  policy_gradient_gru_cnn,
  policy_gradient_cnn **/
"""
# ----------------------------------------------

INTERVAL_SIZE = 30 # 60 Also works well
hyperparams = {
    # ----------------------------------------------

    'algorithm': 'dqn_gru_cnn',

    'baseline_algorithm': 'price_comparison',  # price_comparison, more could be added
    'interval_days': INTERVAL_SIZE,
    'baseline_interval_days': INTERVAL_SIZE,

    'show_buy_sell_signals': True,

    'start_year': '2022-01-01',
    'end_year': '2024-01-01',

    'retrain': False, # For interval training

    'initial_position': 0,  # Initial position (1=long, -1=short, 0=cash)  ,

    'n_episodes':10,
    'positions': [0,1],
    'discount_factor': 0.995,
    'exploration_rate':0.4,
    'exploration_decay': 0.995,
    'exploration_min': 0.01,
    'trading_fees': 0.01,
    'portfolio_initial_value': 10000,
    'windows': 1,
    'verbose': 1,
    'learning_rate': 0.000001,

    # DQN-specific parameters
    'hidden_layer_size':128,
    'lstm_hidden_size': 128,
    'lstm_num_layers': 3,
    'sequence_length': 3, # For DQN-GRU-CNN, needs to relative to the interval size
    'memory_size': 1000,
}


def get_discrete_value(value, bins):
    return np.digitize(value, bins)

# Discrete features for Q-learning, simple etc...
if 'dqn' not in hyperparams['algorithm'] and 'policy' not in hyperparams['algorithm']:
    dynamic_features_arr = [

        lambda history: get_discrete_value(dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=INTERVAL_SIZE),
                                                bins=[0.9, 1.0, 1.1]),

        lambda history: get_discrete_value(dynamic_features.dynamic_feature_sma(history, window=INTERVAL_SIZE//3),
                                                bins=[0.9, 1.0, 1.1]),

        lambda history: get_discrete_value(dynamic_features.dynamic_feature_sma(history, window=INTERVAL_SIZE//2),

                                                bins=[0.9, 1.0, 1.1]),
    ]

else:


    # Non-discrete features for gradient-based algorithms
    # dynamic_features_arr = [
    #     # *[lambda history, t=w: dynamic_features.dynamic_stock_price_at_time(history, time=t) for w in
    #     #   range(1, INTERVAL_SIZE + 1)], # Too much noise?
    #
    #     lambda history: dynamic_features.dynamic_feature_last_position_taken(history),
    #     lambda history: dynamic_features.dynamic_feature_real_position(history),
    #
    #     lambda history: dynamic_features.dynamic_stock_price_at_time(history, time=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_stock_price_at_time(history, time=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_stock_price_at_time(history, time=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_price_change(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_price_change(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_price_change(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_sma(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_sma(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_sma(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_adjusted_close_ema_ratio(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_adjusted_close_ema_ratio(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_adjusted_close_ema_ratio(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_ema(history, window=INTERVAL_SIZE // 3),
    #     lambda history: dynamic_features.dynamic_feature_ema(history, window=INTERVAL_SIZE // 2),
    #     lambda history: dynamic_features.dynamic_feature_ema(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_momentum(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_momentum(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_momentum(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_atr(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_atr(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_atr(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_rsi(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_rsi(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_rsi(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE),
    #
    #     lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=INTERVAL_SIZE//3),
    #     lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=INTERVAL_SIZE//2),
    #     lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=INTERVAL_SIZE),
    # ]

    dynamic_features_arr = [
        *[lambda history, t=w: dynamic_features.dynamic_stock_price_at_time(history, time=t) for w in
          range(1, INTERVAL_SIZE + 1)], # Too much noise?
        lambda history: dynamic_features.dynamic_feature_last_position_taken(history),
        lambda history: dynamic_features.dynamic_feature_real_position(history),

        # Stock price at dispersed points in time
        # lambda history: dynamic_features.dynamic_stock_price_at_time(history, time=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_stock_price_at_time(history, time=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_stock_price_at_time(history, time=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_stock_price_at_time(history, time=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_stock_price_at_time(history, time=INTERVAL_SIZE),
        #
        # # Price change over dispersed windows
        # lambda history: dynamic_features.dynamic_feature_price_change(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_price_change(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_price_change(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_price_change(history, window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_price_change(history, window=INTERVAL_SIZE),
        #
        # # Adjusted close-to-SMA ratio with spread-out time windows
        # lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history,
        #                                                                           window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_adjusted_close_sma_ratio(history, window=INTERVAL_SIZE),
        #
        # # Simple Moving Average (SMA) over more dispersed windows
        # lambda history: dynamic_features.dynamic_feature_sma(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_sma(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_sma(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_sma(history, window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_sma(history, window=INTERVAL_SIZE),
        #
        # # Exponential Moving Average (EMA) ratio
        # lambda history: dynamic_features.dynamic_feature_adjusted_close_ema_ratio(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_adjusted_close_ema_ratio(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_adjusted_close_ema_ratio(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_adjusted_close_ema_ratio(history,
        #                                                                           window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_adjusted_close_ema_ratio(history, window=INTERVAL_SIZE),
        #
        # # Momentum indicators over different time windows
        # lambda history: dynamic_features.dynamic_feature_momentum(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_momentum(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_momentum(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_momentum(history, window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_momentum(history, window=INTERVAL_SIZE),
        #
        # # Volatility indicator (ATR) over dispersed time windows
        # lambda history: dynamic_features.dynamic_feature_atr(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_atr(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_atr(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_atr(history, window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_atr(history, window=INTERVAL_SIZE),
        #
        # # Bollinger Band width over dispersed windows
        # lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_bollinger_band_width(history, window=INTERVAL_SIZE),
        #
        # # Stochastic Oscillator
        # lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_stochastic_oscillator(history, window=INTERVAL_SIZE),
        #
        # # RSI over different time windows
        # lambda history: dynamic_features.dynamic_feature_rsi(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_rsi(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_rsi(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_rsi(history, window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_rsi(history, window=INTERVAL_SIZE),
        #
        # # Price difference over dispersed time windows
        # lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_price_diff(history, window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_price_diff(history, window=INTERVAL_SIZE),
        #
        # # Volume moving averages over different windows
        # lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=INTERVAL_SIZE // 8),
        # lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=INTERVAL_SIZE // 2),
        # lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=3 * INTERVAL_SIZE // 4),
        # lambda history: dynamic_features.dynamic_feature_volume_ma(history, window=INTERVAL_SIZE),
    ]