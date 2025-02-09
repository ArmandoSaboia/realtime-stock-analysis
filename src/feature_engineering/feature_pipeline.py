def calculate_moving_average(data, window=5):
    """
    Calculate the moving average of stock prices.
    :param data: DataFrame containing stock data
    :param window: Window size for moving average
    :return: DataFrame with moving average column
    """
    data['moving_average'] = data['close'].rolling(window=window).mean()
    return data