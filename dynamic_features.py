import numpy as np
from matplotlib.style.core import available


# def dynamic_feature_last_position_taken(history):
#     return history['position', -1]
#
# def dynamic_feature_real_position(history):
#     return history['real_position', -1]
#
# # Dynamic Feature: Adjusted Close to SMA Ratio
# def dynamic_feature_adjusted_close_sma_ratio(history, window):
#     if len(history) < window:
#         return 0.0
#
#     available_window = min(window, len(history) - 1)
#     sma = history["data_close", -available_window:].mean()
#     current_price = history["data_close", -1]
#     return current_price / sma if sma != 0 else 0.0
#
def dynamic_feature_adjusted_close_ema_ratio(history, window):
    if len(history) < window:
        return 0.0

    available_window = min(window, len(history) - 1)
    prices = history["data_close", -available_window:]
    alpha = 2 / (available_window + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append((price * alpha) + (ema[-1] * (1 - alpha)))
    return history["data_close", -1] / ema[-1] if ema[-1] != 0 else 0.0
#
# Dynamic Feature: RSI
def dynamic_feature_rsi(history, window=14):
    if len(history) == 0:
        return 0.0

    # Ensure we have at least two data points to compute differences
    available_window = min(window, len(history) - 1)

    if available_window < 1:
        return 0.0  # Not enough data for RSI calculation

    # Calculate the price differences
    delta = np.diff(history["data_close", -available_window - 1:])

    # If delta is empty, return 0.0 to avoid runtime warnings
    if delta.size == 0:
        return 0.0

    # Compute gain and loss
    gain = np.maximum(delta, 0).mean()
    loss = -np.minimum(delta, 0).mean()

    # Avoid division by zero in case of no losses
    rs = gain / loss if loss != 0 else 0
    return 100 - (100 / (1 + rs))
#
# # Dynamic Feature: MACD
def dynamic_feature_macd(history, window_short=12, window_long=26, window_signal=9):
    if len(history) <= 1:
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
    if len(history) == 0:
        return [0.0, 0.0, 0.0]
    available_window = min(window, len(history) - 1)
    sma = history["data_close", -available_window:].mean()
    std = history["data_close", -available_window:].std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    current_price = history["data_close", -1]
    return [current_price - sma, upper_band - sma, lower_band - sma]
#
# Dynamic Feature: On-Balance Volume (OBV)
def dynamic_feature_obv(history):
    if len(history) == 0:
        return 0.0
    obv = 0
    for i in range(1, len(history)):
        if history["data_close"][i] > history["data_close"][i-1]:
            obv += history["data_volume"][i]
        elif history["data_close"][i] < history["data_close"][i-1]:
            obv -= history["data_volume"][i]
    return obv

# Dynamic Feature: ADX
def dynamic_feature_adx(history, window=14):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    high_diff = np.diff(history["data_high", -available_window - 1:])
    low_diff = np.diff(history["data_low", -available_window - 1:])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    previous_close = np.roll(history["data_close", -available_window - 1:], shift=1)
    tr = np.maximum(history["data_high", -available_window - 1:], previous_close) - np.minimum(history["data_low", -available_window - 1:], previous_close)
    atr = np.mean(tr)
    adx = (np.mean(plus_dm) - np.mean(minus_dm)) / atr if atr != 0 else 0.0
    return np.abs(adx)

# Dynamic Feature: EMA
def dynamic_feature_ema(history, window):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    prices = history["data_close", -available_window:]
    alpha = 2 / (available_window + 1)
    ema = [prices[0]]
    for price in prices[1:]:
        ema.append((price * alpha) + (ema[-1] * (1 - alpha)))
    return ema[-1]
#
# Dynamic Feature: Price Rate of Change (ROC)
def dynamic_feature_roc(history, window=10):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    return ((history["data_close", -1] - history["data_close", -available_window - 1]) / history["data_close", -available_window - 1]) * 100

# Dynamic Feature: Price Difference
def dynamic_feature_price_diff(history, window=1):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    return history["data_close", -1] - history["data_close", -available_window - 1]

# Dynamic Feature: Daily Return
def dynamic_feature_daily_return(history):
    if len(history) <= 1:
        return 0.0
    return (history["data_close", -1] / history["data_close", -2]) - 1

# Dynamic Feature: High-Low Range
def dynamic_feature_high_low_range(history):
    if len(history) == 0:
        return 0.0
    return history["data_high", -1] - history["data_low", -1]
#
# Dynamic Feature: Volume Moving Average
def dynamic_feature_volume_ma(history, window=20):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    return history["data_volume", -available_window:].mean()

# Dynamic Feature: Bollinger Band Width
def dynamic_feature_bollinger_band_width(history, window=20, num_std=2):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    sma = history["data_close", -available_window:].mean()
    std = history["data_close", -available_window:].std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return upper_band - lower_band
#
# # Dynamic Feature: Stochastic Oscillator
def dynamic_feature_stochastic_oscillator(history, window=14):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    highest_high = np.max(history["data_high", -available_window:])
    lowest_low = np.min(history["data_low", -available_window:])
    current_close = history["data_close", -1]
    return (current_close - lowest_low) / (highest_high - lowest_low) * 100 if highest_high != lowest_low else 0.0


def dynamic_feature_atr(history, window=14):
    if len(history) == 0:
        return 0.0

    available_window = min(window, len(history) - 1)

    tr_high = history["data_high", -available_window:]
    tr_low = history["data_low", -available_window:]
    previous_close = history["data_close", -available_window - 1:-1]

    # Ensure valid array lengths before calculating true ranges
    if len(tr_high) == 0 or len(tr_low) == 0 or len(previous_close) == 0:
        return 0.0

    tr1 = tr_high - tr_low
    tr2 = np.abs(tr_high - previous_close)
    tr3 = np.abs(tr_low - previous_close)

    # Calculate true range and convert to a NumPy array with float type
    true_range = np.array(np.maximum(tr1, np.maximum(tr2, tr3)), dtype=np.float64)

    # Handle possible NaN or inf values
    if np.isnan(true_range).any() or np.isinf(true_range).any():
        return 0.0

    return np.nanmean(true_range)  # Use np.nanmean to ignore NaN values if present

# Dynamic Feature: Momentum
def dynamic_feature_momentum(history, window=10):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    return history["data_close", -1] - history["data_close", -available_window - 1]

# Dynamic Feature: Skewness
def dynamic_feature_skewness(history, window=30):
    if len(history) == 0:
        return 0.0

    available_window = min(window, len(history) - 1)
    close_prices = history["data_close", -available_window:]

    # Ensure that we have valid data in the window
    if len(close_prices) == 0:
        return 0.0

    mean_close = np.mean(close_prices)
    std_close = np.std(close_prices)

    # Check if the standard deviation is zero to prevent division by zero
    if std_close == 0:
        return 0.0

    # Calculate skewness
    skewness = np.mean((close_prices - mean_close) ** 3) / (std_close ** 3)

    # Handle potential NaN values in the skewness calculation
    return skewness if not np.isnan(skewness) and not np.isinf(skewness) else 0.0
# Dynamic Feature: Kurtosis
def dynamic_feature_kurtosis(history, window=30):
    if len(history) == 0:
        return 0.0

    available_window = min(window, len(history) - 1)
    close_prices = history["data_close", -available_window:]

    # Ensure that we have valid data in the window
    if len(close_prices) == 0:
        return 0.0

    mean_close = np.mean(close_prices)
    std_close = np.std(close_prices)

    # Check if the standard deviation is zero to prevent division by zero
    if std_close == 0:
        return 0.0

    # Calculate kurtosis
    kurtosis = np.mean((close_prices - mean_close) ** 4) / (std_close ** 4)

    # Handle potential NaN values in the kurtosis calculation
    return kurtosis if not np.isnan(kurtosis) and not np.isinf(kurtosis) else 0.0
# Dynamic Feature: Z-score of Close Price
def dynamic_feature_zscore(history, window=20):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    prices = history["data_close", -available_window:]
    mean = prices.mean()
    std = prices.std()
    return (history["data_close", -1] - mean) / std if std != 0 else 0.0

# Dynamic Feature: Money Flow Index (MFI)
def dynamic_feature_mfi(history, window=14):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    typical_price = (history["data_high", -available_window:] + history["data_low", -available_window:] + history["data_close", -available_window:]) / 3
    money_flow = typical_price * history["data_volume", -available_window:]
    positive_flow = np.sum(money_flow[1:][money_flow[1:] > money_flow[:-1]])
    negative_flow = np.sum(money_flow[1:][money_flow[1:] < money_flow[:-1]])
    money_flow_ratio = positive_flow / negative_flow if negative_flow != 0 else 0
    return 100 - (100 / (1 + money_flow_ratio))


def dynamic_feature_vwap(history):
    if len(history) == 0:
        return 0.0
    return np.sum(history["data_volume", -1] * history["data_close", -1]) / np.sum(history["data_volume", -1])

def dynamic_feature_price_state(history):
    if len(history) < 2:
        return 0.0  # Not enough data to determine state
    return 1 if history["data_close", -1] >= history["data_close", -2] else -1


def dynamic_feature_transitional_probability(history):
    if len(history) < 2:
        return 0.0  # Not enough data to compute probabilities
    rise = history["data_close", -1] > history["data_close", -2]
    buy = history["position", -1] == 1
    return 1 if rise and buy else 0  # Returns 1 if both rise and buy, otherwise 0


def dynamic_feature_sharpe_ratio(history, risk_free_rate=0.0, window=14):
    if len(history) < 2:
        return 0.0
    available_window = min(window, len(history) - 1)

    # Calculate returns
    returns = np.diff(history["data_close", -available_window:]) / history["data_close", -available_window:-1]

    # Check if there are enough returns to compute mean and std
    if len(returns) < 2:
        return 0.0  # Not enough data to calculate Sharpe ratio

    excess_returns = returns - risk_free_rate

    # Safeguard against std being 0
    std_excess_returns = excess_returns.std()

    return excess_returns.mean() / std_excess_returns if std_excess_returns != 0 else 0.0

def dynamic_feature_williams_r(history, window=14):
    if len(history) == 0:
        return 0.0
    available_window = min(window, len(history) - 1)
    highest_high = np.max(history["data_high", -available_window:])
    lowest_low = np.min(history["data_low", -available_window:])
    current_close = history["data_close", -1]
    return ((highest_high - current_close) / (highest_high - lowest_low)) * -100 if highest_high != lowest_low else 0.0


def dynamic_feature_derivative_sharpe_ratio(history, risk_free_rate=0.0, window=14):
    # Ensure there are enough data points in history
    if len(history) < 2:
        return 0.0

    # Calculate the available window size
    available_window = min(window, len(history) - 1)

    # Calculate returns
    returns = np.diff(history["data_close", -available_window:]) / history["data_close", -available_window:-1]

    # Check if there are enough returns to compute Sharpe ratio
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate

    # Calculate current Sharpe ratio, with safeguard against division by zero
    std_excess_returns = excess_returns.std()
    if std_excess_returns == 0:
        return 0.0

    current_sharpe_ratio = excess_returns.mean() / std_excess_returns

    # Calculate previous Sharpe ratio (one step back), with safeguard against insufficient data
    if len(history) < available_window + 2:
        return 0.0  # Not enough history to compute previous Sharpe ratio

    previous_returns = np.diff(history["data_close", -(available_window + 1):-1]) / history["data_close", -(available_window + 1):-2]

    if len(previous_returns) < 2:
        return 0.0

    previous_excess_returns = previous_returns - risk_free_rate
    previous_std_excess_returns = previous_excess_returns.std()

    if previous_std_excess_returns == 0:
        return 0.0

    previous_sharpe_ratio = previous_excess_returns.mean() / previous_std_excess_returns

    # Return the derivative of the Sharpe ratio (difference between current and previous Sharpe ratios)
    return current_sharpe_ratio - previous_sharpe_ratio

def dynamic_feature_interval_profit(history, window=14):
    if len(history) < 2:
        return 0.0
    available_window = min(window, len(history) - 1)
    return history["data_close", -1] - history["data_close", -available_window]


def dynamic_feature_price_change(history, window=1):
    """
    Returns 1 if the percentage price rise over the specified window exceeds the threshold.
    Otherwise, returns 0.

    :param history: Historical price dataa
    :param threshold: Percentage threshold for sharp rise detection (e.g., 0.05 for 5%)
    :param window: Time window to compare the price change (default is 1, comparing the last price with the previous price)
    :return: 1 if sharp rise, 0 otherwise
    """
    if len(history) < window + 1:
        return 0.0

    # Calculate percentage price change
    price_change = (history["data_close", -1] - history["data_close", -window - 1]) / history["data_close", -window - 1]

    return price_change

