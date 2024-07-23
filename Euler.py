from Option_Base import European_Option

import numpy as np

def euler_implicit(European_Option, N):

    S, K, T, r, sigma, option_type = European_Option.Retrieve_Attributes


    dt = T / N
    asset_prices = np.zeros(N + 1)
    option_values = np.zeros(N + 1)

    for i in range(N + 1):
        asset_prices[i] = S * np.exp(r * (N - i) * dt + sigma * np.sqrt((N - i) * dt) * np.random.normal())

    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)
    else:
        option_values = np.maximum(0, K - asset_prices)

    option_price = np.mean(option_values) * np.exp(-r * T)
    return option_price


def euler_explicit(European_Option, N):

    S, K, T, r, sigma, option_type = European_Option.Retrieve_Attributes


    dt = T / N
    asset_prices = np.zeros(N + 1)
    option_values = np.zeros(N + 1)

    for i in range(N + 1):
        asset_prices[i] = S * np.exp(r * (N - i) * dt + sigma * np.sqrt((N - i) * dt) * np.random.normal())

    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)
    else:
        option_values = np.maximum(0, K - asset_prices)

    option_price = np.mean(option_values) * np.exp(-r * T)
    return option_price
