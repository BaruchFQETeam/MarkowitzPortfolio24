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

def calculate_portfolio_value_no_rebalancing(df, weights, initial_investment=1000):
    """
    Calculate portfolio value with initial weights, allowing the weights to drift over time.
    This does not assume daily rebalancing.
    
    Returns:
    - pd.Series: Series of portfolio values over time.
    """
    # Calculate initial investment distribution per stock
    initial_investment_per_stock = {
        stock: initial_investment * weight for stock, weight in weights.items() if stock in df.columns
    }
    
    # Calculate portfolio value by allowing weights to drift over time
    portfolio_values = []
    for date, prices in df.iterrows():
        portfolio_value = sum(prices[stock] * (initial_investment_per_stock[stock] / df[stock].iloc[0]) for stock in initial_investment_per_stock)
        portfolio_values.append(portfolio_value)
    
    return pd.Series(portfolio_values, index=df.index)

def calculate_portfolio_value_with_rebalancing(df, weights, initial_investment=1000, rebalance_frequency=1):
    """
    Calculate portfolio value with periodic rebalancing.
    
    Args:
    - df (pd.DataFrame): DataFrame with Date as the index and stock prices as columns.
    - weights (dict): Dictionary with stock tickers as keys and weights as values.
    - initial_investment (float): Starting amount to invest in the portfolio.
    - rebalance_frequency (int): Number of days between rebalancing. If set to 0, no rebalancing occurs.
    
    Returns:
    - pd.Series: Series of portfolio values over time.
    """
    portfolio_values = []
    initial_investment_per_stock = {stock: initial_investment * weight for stock, weight in weights.items() if stock in df.columns}
    
    # Initial shares bought based on initial weights and first day's prices
    shares = {stock: initial_investment_per_stock[stock] / df[stock].iloc[0] for stock in initial_investment_per_stock}
    
    for i, (date, prices) in enumerate(df.iterrows()):
        # Calculate current portfolio value
        portfolio_value = sum(prices[stock] * shares[stock] for stock in shares)
        portfolio_values.append(portfolio_value)
        
        # Rebalance if needed (based on frequency) and not on the first date
        if rebalance_frequency > 0 and i % rebalance_frequency == 0 and i != 0:
            # Rebalance portfolio: calculate new investment per stock based on current portfolio value
            new_investment_per_stock = {stock: portfolio_value * weight for stock, weight in weights.items()}
            shares = {stock: new_investment_per_stock[stock] / prices[stock] for stock in new_investment_per_stock}
    
    return pd.Series(portfolio_values, index=df.index)

def add_ewma_bollinger_bands(portfolio_values, halflife_days):
    """
    Add EWMA and Bollinger Bands to the portfolio values.
    """
    df = pd.DataFrame({'portfolio_value': portfolio_values})
    
    # Calculate EWMA
    df['ewma'] = df['portfolio_value'].ewm(halflife=halflife_days).mean()
    
    # Calculate standard deviation and Bollinger Bands
    df['std_dev'] = df['portfolio_value'].rolling(window=halflife_days).std()
    df['bollinger_upper'] = df['ewma'] + (2 * df['std_dev'])
    df['bollinger_lower'] = df['ewma'] - (2 * df['std_dev'])
    
    df.drop(columns=['std_dev'], inplace=True)
    return df

# Test code
def index_compiler(weights_dict: dict, title: str, halflife_days: int = 20, initial_investment=1000, rebalance=True, rebalance_frequency=1) -> tuple:
    """
    Parameters:
    - weights_dict: Dictionary of stock tickers and their weights.
    - title: Title of the portfolio.
    - halflife_days: Halflife in days for the EWMA calculation.
    - initial_investment: Initial investment amount.
    - rebalance: Boolean flag for rebalancing.
    - rebalance_frequency: Frequency of rebalancing in days.
        - 5 days for weekly rebalancing. Since the market is open 5 days a week.
        - Default is daily rebalancing.

    Computes portfolio value and adds EWMA and Bollinger Bands.

    Returns a tuple of the DataFrame with EWMA and Bollinger Bands, the weights dictionary, and the title.
    """
    df = pd.read_csv('sp500_5year_close_prices.csv')
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')

    # Validate that weights sum to 1
    total_weight = sum(weights_dict.values())
    if total_weight != 1:
        weights_dict = {ticker: weight / total_weight for ticker, weight in weights_dict.items()}

    if rebalance:
        portfolio_values = calculate_portfolio_value_with_rebalancing(df, weights_dict, initial_investment, rebalance_frequency)
    else:
        portfolio_values = calculate_portfolio_value_no_rebalancing(df, weights_dict, initial_investment)

    ewma_bollinger_df = add_ewma_bollinger_bands(portfolio_values, halflife_days)
    return ewma_bollinger_df, weights_dict, title

def plot_returns(returns_n_weights):
    """
    Parameters: returns_n_weights is a list of tuples. Each tuple contains the DataFrame with EWMA and Bollinger Bands, weights of a portfolio, and a portfolio title, as returned by the index_compiler function.
    Plots portfolio values, EWMA, and Bollinger Bands based on initial investment.
    """
    plt.figure(figsize=(12, 7))

    labeled_bands = False

    for ewma_bollinger_df, weights, title in returns_n_weights:
        if not labeled_bands:
            plt.fill_between(ewma_bollinger_df.index, ewma_bollinger_df['bollinger_upper'], 
                            ewma_bollinger_df['bollinger_lower'], color='gray', alpha=0.2, 
                            label=f"Bollinger Bands Range")
            plt.plot(ewma_bollinger_df.index, ewma_bollinger_df['bollinger_upper'], color='green', linestyle=':', label="Upper Band")
            plt.plot(ewma_bollinger_df.index, ewma_bollinger_df['bollinger_lower'], color='red', linestyle=':', label="Lower Band")
            labeled_bands = True
        else:
            plt.fill_between(ewma_bollinger_df.index, ewma_bollinger_df['bollinger_upper'], 
                ewma_bollinger_df['bollinger_lower'], color='gray', alpha=0.2)
            plt.plot(ewma_bollinger_df.index, ewma_bollinger_df['bollinger_upper'], color='green', linestyle=':')
            plt.plot(ewma_bollinger_df.index, ewma_bollinger_df['bollinger_lower'], color='red', linestyle=':')
        plt.plot(ewma_bollinger_df.index, ewma_bollinger_df['portfolio_value'], label=f"{title} Portfolio Value")
        plt.plot(ewma_bollinger_df.index, ewma_bollinger_df['ewma'], label=f"{title} EWMA", linestyle='--')

            

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
    Creates an equally weighted portfolio based on the CSV and plots its value with EWMA and Bollinger Bands.
    """
    global tickers
    load_data(file_path)
    equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
    print(f"Equal weights: {equal_weights}")
    equal_portfolio_rebalanced = index_compiler(equal_weights, 'Equally Weighted Portfolio (Rebalanced Daily)', halflife_days, initial_investment, rebalance=True, rebalance_frequency=1)
    equal_portfolio_no_rebalancing = index_compiler(equal_weights, 'Equally Weighted Portfolio (No Rebalancing)', halflife_days, initial_investment, rebalance=False,)
    equal_portfolio_rebalanced_weekly = index_compiler(equal_weights, 'Equally Weighted Portfolio (Rebalanced Weekly)', halflife_days, initial_investment, rebalance=True, rebalance_frequency=5)
    return [equal_portfolio_rebalanced_weekly]
    # plot_returns([equal_portfolio_rebalanced])

def individual_stock_prep_plot(tickers_recieved, halflife_days: int = 20, initial_investment=1000) -> None:
    """
    Creates a portfolio for each stock in the csv and plots its value with EWMA and Bollinger Bands.

    """
    #specific ticker plot data
    ticker_data = []
    for ticker in tickers_recieved:
        indv_dict = {ticker: 1}
        ind_stock_data = index_compiler(indv_dict, f'{ticker}', halflife_days, initial_investment, rebalance=True)
        ticker_data.append(ind_stock_data)
    return ticker_data


data = csv_equal_weight_portfolio('sp500_5year_close_prices.csv')
tickers_data = individual_stock_prep_plot(['WBA','AAPL'], halflife_days=20, initial_investment=1000)
combined_data = data + tickers_data 
plot_returns(combined_data)
