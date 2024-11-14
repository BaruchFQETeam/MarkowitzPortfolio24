import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tickers = None

def load_data(file_path: str) -> None:
    global tickers
    df = pd.read_csv(file_path)
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    tickers = df.columns

def calculate_portfolio_value(df, weights, initial_investment=1000):
    """
    Calculate the portfolio value as a weighted sum of stock prices, scaled by an initial investment.
    This assumes daily rebalancing.
    """
    weights = {stock: weight for stock, weight in weights.items() if stock in df.columns}
    portfolio_value = sum(df[stock] * weight for stock, weight in weights.items())
    portfolio_value *= initial_investment / portfolio_value.iloc[0]  # Scale to start with initial investment
    return portfolio_value

def calculate_portfolio_value_no_rebalancing(df, weights, initial_investment=1000):
    """
    Calculate portfolio value with initial weights, allowing the weights to drift over time.
    This does not assume daily rebalancing.
    """
    # Calculate initial investment distribution
    initial_investment_per_stock = {stock: initial_investment * weight for stock, weight in weights.items() if stock in df.columns}
    
    # Calculate the portfolio value by allowing weights to drift
    portfolio_value = sum(df[stock] * (initial_investment_per_stock[stock] / df[stock].iloc[0]) for stock in initial_investment_per_stock)
    
    return portfolio_value

def add_ewma_bollinger_bands(portfolio_values, halflife_days):
    """
    Add EWMA and Bollinger Bands to the portfolio values.
    """
    df = pd.DataFrame({'portfolio_value': portfolio_values})
    df['ewma'] = df['portfolio_value'].ewm(halflife=halflife_days).mean()
    df['std_dev'] = df['portfolio_value'].rolling(window=halflife_days).std()
    df['bollinger_upper'] = df['ewma'] + (2 * df['std_dev'])
    df['bollinger_lower'] = df['ewma'] - (2 * df['std_dev'])
    df.drop(columns=['std_dev'], inplace=True)
    return df

def index_compiler(weights_dict: dict, title: str, halflife_days: int = 20, initial_investment=1000, rebalance=True) -> tuple:
    """
    Computes portfolio value and adds EWMA and Bollinger Bands.
    By default, assumes daily rebalancing unless rebalance is set to False.
    """
    df = pd.read_csv('sp500_5year_close_prices.csv')
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')

    missing_tickers = [ticker for ticker in weights_dict if ticker not in df.columns]
    if missing_tickers:
        raise ValueError(f"The following tickers are missing in the DataFrame: {missing_tickers}")

    total_weight = sum(weights_dict.values())
    if total_weight != 1:
        weights_dict = {ticker: weight / total_weight for ticker, weight in weights_dict.items()}

    if rebalance:
        portfolio_values = calculate_portfolio_value(df, weights_dict, initial_investment)
    else:
        portfolio_values = calculate_portfolio_value_no_rebalancing(df, weights_dict, initial_investment)

    ewma_bollinger_df = add_ewma_bollinger_bands(portfolio_values, halflife_days)
    return ewma_bollinger_df, weights_dict, title 

def plot_returns(returns_n_weights):
    """
    Plots portfolio values, EWMA, and Bollinger Bands based on initial investment.
    """
    plt.figure(figsize=(12, 7))

    for ewma_bollinger_df, weights, title in returns_n_weights:
        plt.plot(ewma_bollinger_df.index, ewma_bollinger_df['portfolio_value'], label=f"{title} Portfolio Value")
        plt.plot(ewma_bollinger_df.index, ewma_bollinger_df['ewma'], label=f"{title} EWMA", linestyle='--')
        plt.fill_between(ewma_bollinger_df.index, ewma_bollinger_df['bollinger_upper'], 
                         ewma_bollinger_df['bollinger_lower'], color='gray', alpha=0.2, 
                         label=f"{title} Bollinger Bands")
        plt.plot(ewma_bollinger_df.index, ewma_bollinger_df['bollinger_upper'], color='green', linestyle=':', label="Upper Band")
        plt.plot(ewma_bollinger_df.index, ewma_bollinger_df['bollinger_lower'], color='red', linestyle=':', label="Lower Band")

    plt.title("Portfolio Value with EWMA and Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.xticks(ticks=range(0, len(ewma_bollinger_df.index), 60), labels=ewma_bollinger_df.index[::60], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def csv_equal_weight_portfolio(file_path: str, halflife_days: int = 20, initial_investment=1000) -> None:
    """
    Creates an equally weighted portfolio and plots its value with EWMA and Bollinger Bands.
    """
    global tickers
    load_data(file_path)
    equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
    print(f"Equal weights: {equal_weights}")
    equal_portfolio_rebalanced = index_compiler(equal_weights, 'Equally Weighted Portfolio (Rebalanced Daily)', halflife_days, initial_investment, rebalance=True)
    equal_portfolio_no_rebalancing = index_compiler(equal_weights, 'Equally Weighted Portfolio (No Rebalancing)', halflife_days, initial_investment, rebalance=False)
    nvda_portfolio = index_compiler({'NVDA': 1}, 'NVDA Portfolio', halflife_days, initial_investment, rebalance=True)
    wba_portfolio = index_compiler({'WBA': 1}, 'WBA Portfolio', halflife_days, initial_investment, rebalance=True)
    plot_returns([equal_portfolio_rebalanced, equal_portfolio_no_rebalancing, nvda_portfolio,])
    
    # plot_returns([equal_portfolio_rebalanced])

# Equally weighted portfolio example with initial investment of $1000
csv_equal_weight_portfolio('sp500_5year_close_prices.csv')
