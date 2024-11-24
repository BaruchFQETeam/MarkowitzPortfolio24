import pandas as pd

def calculate_portfolio_value(df, weights):
    """
    Calculate the portfolio value as a weighted sum of stock prices.
    
    Args:
    - df (pd.DataFrame): DataFrame with Date as the index and stock prices as columns.
    - weights (dict): Dictionary with stock tickers as keys and weights as values.
    
    Returns:
    - pd.Series: Series of portfolio values over time.
    """
    # Ensure weights only include stocks present in df columns
    weights = {stock: weight for stock, weight in weights.items() if stock in df.columns}
    
    # Calculate the weighted portfolio value
    portfolio_value = sum(df[stock] * weight for stock, weight in weights.items())
    return portfolio_value

def add_ewma_bollinger_bands(portfolio_values, halflife_days):
    """
    Add EWMA and Bollinger Bands to the portfolio values.
    
    Args:
    - portfolio_values (pd.Series): Series of portfolio values over time.
    - halflife_days (int): Halflife in days for the EWMA calculation.
    
    Returns:
    - pd.DataFrame: DataFrame with Date, portfolio_value, ewma, bollinger_upper, and bollinger_lower columns.
    """
    df = pd.DataFrame({'portfolio_value': portfolio_values})
    
    # Calculate EWMA
    df['ewma'] = df['portfolio_value'].ewm(halflife=halflife_days).mean()
    
    # Calculate rolling standard deviation
    df['std_dev'] = df['portfolio_value'].rolling(window=halflife_days).std()
    
    # Calculate Bollinger Bands
    df['bollinger_upper'] = df['ewma'] + (2 * df['std_dev'])
    df['bollinger_lower'] = df['ewma'] - (2 * df['std_dev'])
    
    # Drop the std_dev column as it's intermediate
    df.drop(columns=['std_dev'], inplace=True)
    
    return df

# Example usage
if __name__ == "__main__":


    df = pd.read_csv('sp500_5year_close_prices.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)

    # Define weights for each stock (adjust these as necessary)
    weights = {
        'NVDA': 0.1,
        'AAPL': 0.15,
        'MSFT': 0.1,
        'AMZN': 0.2,
        'META': 0.05,
        'GOOGL': 0.1,
        'BRK.B': 0.1,
        'GOOG': 0.1,
        'AVGO': 0.05,
        'TSLA': 0.05
    }
    # Calculate the portfolio value time series
    portfolio_value = calculate_portfolio_value(df, weights)

    # Calculate EWMA and Bollinger Bands (for example, 20 days halflife)
    result_df = add_ewma_bollinger_bands(portfolio_value, halflife_days=20)    
    # Display the result
    print(result_df)
