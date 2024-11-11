import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
# Load the data
# df = pd.read_csv('sp500_5year_close_prices.csv')
# df = df.set_index('Date')
# df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')

# # Define tickers
# # get tickers from the first row of the dataframe 
# tickers = df.columns
# print(tickers)
# time.sleep(100)
# tickers = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL', 'BRK.B', 'GOOG', 'AVGO', 'TSLA',
#            'JPM', 'LLY', 'UNH', 'XOM', 'V', 'MA', 'COST', 'HD', 'JNJ', 'PG', 'WMT', 'ABBV',
#            'NFLX', 'BAC', 'CRM', 'ORCL', 'CVX', 'MRK', 'KO', 'WFC', 'AMD', 'CSCO', 'PEP',
#            'ADBE', 'ACN', 'LIN', 'TMO', 'MCD', 'NOW', 'ABT', 'CAT', 'IBM', 'TXN', 'GE', 'PM',
#            'QCOM', 'GS', 'ISRG', 'INTU', 'DIS', 'CMCSA', 'VZ', 'AMGN', 'BKNG', 'AXP', 'MS',
#            'RTX', 'T', 'DHR', 'SPGI', 'UBER', 'AMAT', 'PFE', 'NEE', 'PGR', 'UNP', 'LOW', 'BLK',
#            'ETN', 'HON', 'COP', 'C', 'TJX', 'VRTX', 'BSX', 'BX', 'SYK', 'PANW', 'ADP', 'MU',
#            'FI', 'LMT', 'MDT', 'GILD', 'TMUS', 'SCHW', 'ADI', 'BMY', 'PLTR', 'MMC', 'ANET',
#            'SBUX', 'BA', 'INTC', 'CB', 'PLD', 'DE', 'KKR', 'LRCX', 'ELV', 'UPS', 'SO', 'MO',
#            'GEV', 'AMT', 'PH', 'NKE', 'KLAC', 'ICE', 'MDLZ', 'TT', 'SHW', 'CI', 'DUK', 'APH',
#            'REGN', 'SNPS', 'EQIX', 'PYPL', 'AON', 'PNC', 'CDNS', 'USB', 'WM', 'CME', 'GD',
#            'CMG', 'MSI', 'TDG', 'CVS', 'WELL', 'ZTS', 'ITW', 'CTAS', 'CRWD', 'CL', 'MMM',
#            'CEG', 'COF', 'EMR', 'EOG', 'MCO', 'NOC', 'ORLY', 'CSX', 'MCK', 'BDX', 'APD',
#            'TGT', 'WMB', 'FCX', 'ADSK', 'HCA', 'MAR', 'AJG', 'CARR', 'FDX', 'TFC', 'NSC',
#            'SLB', 'ABNB', 'ECL', 'GM', 'PCAR', 'HLT', 'ROP', 'OKE', 'NXPI', 'URI', 'TRV',
#            'BK', 'SRE', 'AMP', 'AFL', 'AZO', 'JCI', 'RCL', 'PSX', 'DLR', 'SPG', 'GWW', 'MPC',
#            'FTNT', 'PSA', 'AEP', 'FICO', 'NEM', 'KMI', 'ALL', 'O', 'AIG', 'MET', 'DHI',
#            'CMI', 'LHX', 'CPRT', 'D', 'FAST', 'PAYX', 'FIS']

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


test = index_compiler({'NVDA': 0.5, 'AAPL': 0.5}, 'Daniels Artistic Portfolio')
test2 = index_compiler({'NVDA':.9}, 'NVDA')
test3 = index_compiler({'TSLA':.9, 'AAPL':.05, 'MSFT':.03}, 'Josephs Portfolio')
# I used the weights that didn't equal 1 on purpose to see if the function normalizes them, tldr it does
plot_returns([test, test2,test3])





# # Ensure tickers are in the dataframe
# available_tickers = [ticker for ticker in tickers if ticker in df.columns]
# df = df[available_tickers]

# # Generate random weights and normalize them for Portfolio 1
# random_weights = np.random.random(len(available_tickers))
# normalized_weights = random_weights / random_weights.sum()
# ticker_weights = {ticker: weight for ticker, weight in zip(available_tickers, normalized_weights)}

# # Generate random weights and normalize them for Portfolio 2
# random_weights2 = np.random.random(len(available_tickers))
# normalized_weights2 = random_weights2 / random_weights2.sum()
# ticker_weights2 = {ticker: weight for ticker, weight in zip(available_tickers, normalized_weights2)}

# # Calculate weighted portfolio values
# portfolio_values = df.mul([ticker_weights[ticker] for ticker in df.columns], axis=1).sum(axis=1)
# portfolio_values2 = df.mul([ticker_weights2[ticker] for ticker in df.columns], axis=1).sum(axis=1)

# # Calculate cumulative returns percentage for both portfolios
# cumulative_returns = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
# cumulative_returns2 = (portfolio_values2 / portfolio_values2.iloc[0] - 1) * 100

# # Calculate cumulative returns percentage for Nvidia (NVDA)
# nvidia_cumulative_returns = (df["NVDA"] / df["NVDA"].iloc[0] - 1) * 100
# aapl_cumulative_returns = (df["AAPL"] / df["AAPL"].iloc[0] - 1) * 100
