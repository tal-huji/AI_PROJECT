import numpy as np


def dynamic_feature_last_position_taken(history):
    return history['position', -1]

def dynamic_feature_real_position(history):
    return history['real_position', -1]


def dynamic_stock_price_at_time(history, time):
    if len(history) < time + 1:
        return 0.0
    return history["data_close", -time]

# Dynamic Feature: Adjusted Close to SMA Ratio
def dynamic_feature_adjusted_close_sma_ratio(history, window):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    sma = history["data_close", -available_window:].mean()
    current_price = history["data_close", -1]
    return current_price / sma if sma != 0 else 0.0

# Dynamic Feature: Adjusted Close to EMA Ratio
def dynamic_feature_adjusted_close_ema_ratio(history, window):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    prices = history["data_close", -available_window:]
    alpha = 2 / (available_window + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append((price * alpha) + (ema[-1] * (1 - alpha)))
    return history["data_close", -1] / ema[-1] if ema[-1] != 0 else 0.0

# Dynamic Feature: RSI
def dynamic_feature_rsi(history, window=14):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    delta = np.diff(history["data_close", -available_window - 1:])

    if delta.size == 0:
        return 0.0

    gain = np.maximum(delta, 0).mean()
    loss = -np.minimum(delta, 0).mean()

    rs = gain / loss if loss != 0 else 0
    return 100 - (100 / (1 + rs))

# Dynamic Feature: MACD
def dynamic_feature_macd(history, window_short=12, window_long=26, window_signal=9):
    if len(history) < window_long + 1:
        return 0.0

    available_short = min(window_short, len(history) - 1)
    available_long = min(window_long, len(history) - 1)
    available_signal = min(window_signal, len(history) - 1)
    short_ema = history["data_close", -available_short:].mean()
    long_ema = history["data_close", -available_long:].mean()
    macd = short_ema - long_ema
    signal = history["data_close", -available_signal:].mean()
    return macd - signal

# Dynamic Feature: Bollinger Bands
def dynamic_feature_bollinger_bands(history, window=20, num_std=2):
    if len(history) < window + 1:
        return [0.0, 0.0, 0.0]

    available_window = min(window, len(history) - 1)
    sma = history["data_close", -available_window:].mean()
    std = history["data_close", -available_window:].std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    current_price = history["data_close", -1]
    return [current_price - sma, upper_band - sma, lower_band - sma]

# Dynamic Feature: On-Balance Volume (OBV)
def dynamic_feature_obv(history):
    if len(history) < 2:
        return 0.0
    obv = 0
    for i in range(1, len(history)):
        if history["data_close"][i] > history["data_close"][i-1]:
            obv += history["data_volume"][i]
        elif history["data_close"][i] < history["data_close"][i-1]:
            obv -= history["data_volume"][i]
    return obv

# Dynamic Feature: SMA
def dynamic_feature_sma(history, window):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    return history["data_close", -available_window:].mean()

# Dynamic Feature: EMA
def dynamic_feature_ema(history, window):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    prices = history["data_close", -available_window:]
    alpha = 2 / (available_window + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append((price * alpha) + (ema[-1] * (1 - alpha)))
    return ema[-1]

# Dynamic Feature: Price Rate of Change (ROC)
def dynamic_feature_roc(history, window=10):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    return ((history["data_close", -1] - history["data_close", -available_window - 1]) / history["data_close", -available_window - 1]) * 100

# Dynamic Feature: Price Difference
def dynamic_feature_price_diff(history, window=1):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    return history["data_close", -1] - history["data_close", -available_window - 1]

# Dynamic Feature: Daily Return
def dynamic_feature_daily_return(history):
    if len(history) < 2:
        return 0.0
    return (history["data_close", -1] / history["data_close", -2]) - 1

# Dynamic Feature: High-Low Range
def dynamic_feature_high_low_range(history):
    if len(history) < 1:
        return 0.0
    return history["data_high", -1] - history["data_low", -1]

# Dynamic Feature: Volume Moving Average
def dynamic_feature_volume_ma(history, window=20):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    return history["data_volume", -available_window:].mean()

# Dynamic Feature: Bollinger Band Width
def dynamic_feature_bollinger_band_width(history, window=20, num_std=2):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    sma = history["data_close", -available_window:].mean()
    std = history["data_close", -available_window:].std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band - lower_band

# Dynamic Feature: Stochastic Oscillator
def dynamic_feature_stochastic_oscillator(history, window=14):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    highest_high = np.max(history["data_high", -available_window:])
    lowest_low = np.min(history["data_low", -available_window:])
    current_close = history["data_close", -1]
    return (current_close - lowest_low) / (highest_high - lowest_low) * 100 if highest_high != lowest_low else 0.0

# Dynamic Feature: ATR
def dynamic_feature_atr(history, window=14):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    tr_high = history["data_high", -available_window:]
    tr_low = history["data_low", -available_window:]
    previous_close = history["data_close", -available_window - 1:-1]

    if len(tr_high) == 0 or len(tr_low) == 0 or len(previous_close) == 0:
        return 0.0

    tr1 = tr_high - tr_low
    tr2 = np.abs(tr_high - previous_close)
    tr3 = np.abs(tr_low - previous_close)
    true_range = np.array(np.maximum(tr1, np.maximum(tr2, tr3)), dtype=np.float64)

    if np.isnan(true_range).any() or np.isinf(true_range).any():
        return 0.0

    return np.nanmean(true_range)

# Dynamic Feature: Momentum
def dynamic_feature_momentum(history, window=10):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    return history["data_close", -1] - history["data_close", -available_window - 1]

# Dynamic Feature: Price Change
def dynamic_feature_price_change(history, window=1):
    if len(history) < window + 1:
        return 0.0

    available_window = min(window, len(history) - 1)
    price_change = (history["data_close", -1] - history["data_close", -available_window - 1]) / history[
        "data_close", -available_window - 1]
    return price_change
