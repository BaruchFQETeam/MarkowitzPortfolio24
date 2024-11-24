import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
# Load the data
tickers = None

def load_data(file_path: str) -> None:
    global tickers
    df = pd.read_csv(file_path)
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    tickers = df.columns


def index_compiler(weights_dict: dict, title: str) -> tuple:
    '''
    The input will be a dict {ticker: weight}.
    The function calculates the percentage return from the beginning, of that portfolio given
    the chosen weights. 

    Returns a tuple of the cumulative returns df, the weights, and the title of the portfolio.
    '''
    # Load the DataFrame and set the date index
    df = pd.read_csv('sp500_5year_close_prices.csv')
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    
    # Ensure tickers in weights_dict are in the DataFrame
    missing_tickers = [ticker for ticker in weights_dict if ticker not in df.columns]
    if missing_tickers:
        raise ValueError(f"The following tickers are missing in the DataFrame: {missing_tickers}")

    # Confirm the chosen weights equal a total of 1, normalizing them if not
    total_weight = sum(weights_dict.values())
    if total_weight != 1:
        print("The weights do not add up to 1, normalizing them.")
        weights_dict = {ticker: weight / total_weight for ticker, weight in weights_dict.items()}
        print(f"Normalized weights: {weights_dict}")

    # Calculate weighted portfolio values
    portfolio_values = df[list(weights_dict.keys())].mul(
        list(weights_dict.values()), axis=1
    ).sum(axis=1)

    # Calculate cumulative returns percentage for the portfolio
    cumulative_returns = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
    return cumulative_returns, weights_dict, title 


# Plot the cumulative returns percentage
def plot_returns(returns_n_weights):
    """ 
    Parameters: returns_n_weights is a list of tuples. Each tuple contains the cumulative returns (Series), 
    weights of a portfolio, and a portfolio title, as returned by the index_compiler function. 
    """
    plt.figure(figsize=(12, 7))  # Initialize the figure once at the beginning

    # Plot each portfolio's cumulative returns
    for (cumulative_returns, weights, title) in returns_n_weights:
        plt.plot(cumulative_returns.index, cumulative_returns, label=f"{title} ")

    # Configure plot aesthetics
    plt.title("Compounded Returns Percentage Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")

    # Set x-axis tick labels at a 60-day interval
    plt.xticks(ticks=range(0, len(cumulative_returns.index), 60), 
               labels=cumulative_returns.index[::60], rotation=45)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def csv_equal_weight_portfolio(file_path: str) -> None:
    """ 
    Whichever csv of prices that is passed in, it will create an equally weighted portfolio of all the tickers in the csv
    and plot the returns of that portfolio.
    """
    global tickers
    load_data(file_path)
    equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
    equal_portfolio = index_compiler(equal_weights, 'Equally Weighted Portfolio')
    plot_returns([equal_portfolio])

# test = index_compiler({'NVDA': 0.5, 'AAPL': 0.5}, 'Daniels Artistic Portfolio')
# test2 = index_compiler({'NVDA':.9}, 'NVDA')
# test3 = index_compiler({'TSLA':.9, 'AAPL':.05, 'MSFT':.03}, 'Josephs Portfolio')
# # I used the weights that didn't equal 1 on purpose to see if the function normalizes them, tldr it does
# plot_returns([test, test2,test3])

# just plotting equally weight of the whole portfolio of whatver is in the CSV
csv_equal_weight_portfolio('sp500_5year_close_prices.csv')


