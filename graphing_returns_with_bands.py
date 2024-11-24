import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from Robinhood.RobhinhoodQuotes import write_sp500_data
#seaborn is a library for making statistical graphics in Python. It is built on top of matplotlib and closely integrated with pandas data structures.

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


def plot_returns(returns_n_weights, trade_logs=None, trades_tally=0):
    """
    Parameters: 
    - returns_n_weights: A list of tuples with DataFrame, weights, and title as returned by index_compiler.
    - trade_logs: List of tuples with (trade_date, trade_price, trade_type, total_trade_pnl).
    
    Plots portfolio values, EWMA, Bollinger Bands, and cumulative PnL from trade_logs, with trade markers on the PnL tracker.
    """
    plt.figure(figsize=(12, 7))

    labeled_bands = False
    cumulative_pnl = []  # To store cumulative PnL over time
    pnl_dates = []

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

    # Add trade points and PnL line if available
    if trade_logs:
        # Extract cumulative PnL directly from trade logs
        for trade_date, trade_price, trade_type, total_trade_pnl in trade_logs:
            pnl_dates.append(trade_date)
            cumulative_pnl.append(total_trade_pnl)

        # Plot the PnL line
        plt.plot(pnl_dates, cumulative_pnl, label="Cumulative PnL", color='grey', linewidth=2)

        # Plot trade markers on the PnL tracker
        used_labels = set()  # Track used labels for legend only
        for trade_date, trade_price, trade_type, total_trade_pnl in trade_logs:
            if trade_type == 'long_entry':
                plt.scatter(trade_date, total_trade_pnl, color='blue', marker='^', s=50, 
                            label='Open Long' if 'Open Long' not in used_labels else "", alpha=0.8, zorder=5)
                used_labels.add('Open Long')
            elif trade_type == 'long_exit':
                plt.scatter(trade_date, total_trade_pnl, color='orange', marker='v', s=50, 
                            label='Close Long' if 'Close Long' not in used_labels else "", alpha=0.8, zorder=5)
                used_labels.add('Close Long')
            elif trade_type == 'short_entry':
                plt.scatter(trade_date, total_trade_pnl, color='green', marker='^', s=50, 
                            label='Open Short' if 'Open Short' not in used_labels else "", alpha=0.8, zorder=5)
                used_labels.add('Open Short')
            elif trade_type == 'short_exit':
                plt.scatter(trade_date, total_trade_pnl, color='red', marker='v', s=50, 
                            label='Close Short' if 'Close Short' not in used_labels else "", alpha=0.8, zorder=5)
                used_labels.add('Close Short')
            

    plt.title("Cumulative PnL with Trades")
    plt.xlabel("Date")
    plt.ylabel("PnL ($)")
    plt.xticks(ticks=range(0, len(pnl_dates), max(1, len(pnl_dates) // 10)), rotation=45)
    weights_dict = returns_n_weights[0][1] # Get the weights of the portfolio
    weights_list = [f"{ticker}: {weight:.2%}" for ticker, weight in weights_dict.items()]  # Format each stock weight
    weights_text = "\n".join(weights_list)  # Combine into a multi-line string

    # Add legend
    plt.legend(loc='upper left')

    # Add weights and total trades as additional text, 1 weight per line
    plt.text(1.02, 0.5, f"Stock Weights:\n{weights_text}\n\nTotal Trades: {trades_tally}", 
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='center', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))

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


def track_trades(df, trade_size=1000):
    """
    Returns a tuple of trade logs and the total number of trades.
    trade_logs is a list of tuples with (trade_date, trade_price, trade_type, total_trade_pnl).
    trades_tally is an integer representing the total number of trades.
    """
    current_position = None
    entry_price = 0
    units_traded = 0
    trades_tally = 0

    # Ensure these columns exist in your DataFrame
    df['trade_pnl'] = 0.0
    df['total_trade_pnl'] = 0.0
    df['position'] = None


    # Lists for tracking trades
    short_position_open = []
    short_position_close = []
    long_position_open = []
    long_position_close = []
    trade_logs = []  # Log of all trades for plotting

    for i in range(len(df)):
        price = df.loc[df.index[i], 'portfolio_value']
        ewma = df.loc[df.index[i], 'ewma']
        upper = df.loc[df.index[i], 'bollinger_upper']
        lower = df.loc[df.index[i], 'bollinger_lower']

        if i > 0:
            # Start with the realized PnL carried forward
            df.loc[df.index[i], 'total_trade_pnl'] = df.loc[df.index[i - 1], 'total_trade_pnl']

        else:
            # Initialize for the first row
            df.loc[df.index[i], 'total_trade_pnl'] = 0
        
        # Calculate unrealized PnL (temporary, not added to total_trade_pnl)
        unrealized_pnl = 0
        if current_position == 'short':
            unrealized_pnl = units_traded * (entry_price - price)  # Profit from price decrease
        elif current_position == 'long':
            unrealized_pnl = units_traded * (price - entry_price)  # Profit from price increase

        # Append to trade_logs
        trade_logs.append((df.index[i], price, 'neither', df.loc[df.index[i], 'total_trade_pnl'] + unrealized_pnl))

        # Enter a short position
        if current_position is None and price > upper:
            print(f"Short position entered at {price}")
            short_position_open.append(price)
            trade_logs[i] = (trade_logs[i][0], trade_logs[i][1], 'short_entry', trade_logs[i][3])  # Overwrite trade type
            current_position = 'short'
            entry_price = price
            units_traded = trade_size / entry_price  # Calculate units for fixed trade size
            df.loc[df.index[i], 'position'] = 'short'

        # Enter a long position
        elif current_position is None and price < lower:
            print(f"Long position entered at {price}")
            long_position_open.append(price)
            trade_logs[i] = (trade_logs[i][0], trade_logs[i][1], 'long_entry', trade_logs[i][3])  #overwriting the trade type
            current_position = 'long'
            entry_price = price
            units_traded = trade_size / entry_price  # Calculate units for fixed trade size
            df.loc[df.index[i], 'position'] = 'long'

        # Exit short position
        elif current_position == 'short' and price < ewma:
            print(f"Short position exited at {price}")
            short_position_close.append(price)
            trade_logs[i] = (trade_logs[i][0], trade_logs[i][1], 'short_exit', trade_logs[i][3])  #overwriting the trade type
            pnl = units_traded * (entry_price - price)  # Use units_traded for PnL
            df.loc[df.index[i], 'total_trade_pnl'] += pnl
            current_position = None
            entry_price = 0
            units_traded = 0
            trades_tally += 1

        # Exit long position
        elif current_position == 'long' and price > ewma:
            print(f"Long position exited at {price}")
            long_position_close.append(price)
            trade_logs[i] = (trade_logs[i][0], trade_logs[i][1], 'long_exit', trade_logs[i][3]) #overwriting the trade type
            pnl = units_traded * (price - entry_price)  # Use units_traded for PnL
            df.loc[df.index[i], 'total_trade_pnl'] += pnl
            current_position = None
            entry_price = 0
            units_traded = 0
            trades_tally += 1

    print(df[['portfolio_value', 'position', 'total_trade_pnl']])

    # Debug output
    print("Short Position Open:", short_position_open)
    print("Short Position Close:", short_position_close)
    print("Long Position Open:", long_position_open)
    print("Long Position Close:", long_position_close)

    # Calculate overall PnL
    # short_pnl = (np.array(short_position_open) - np.array(short_position_close)) * trade_size / np.array(short_position_open)
    # long_pnl = (np.array(long_position_close) - np.array(long_position_open)) * trade_size / np.array(long_position_open)

    # total_short_pnl = np.sum(short_pnl)
    # total_long_pnl = np.sum(long_pnl)
    # combined_pnl = total_short_pnl + total_long_pnl

    # print("Total Short PnL:", total_short_pnl)
    # print("Total Long PnL:", total_long_pnl)
    # print("Total Combined PnL:", combined_pnl)
    print("Total Trades:", trades_tally)
    if current_position == 'Open':
        print("Position still open, This means the graph ends with an open position")

    return trade_logs, trades_tally # Return the trade logs for plotting


#pick stocks to use
stock_picks = ['GOLD','SRPT','DAL','AAPL','WBA','TSLA','MSFT','NVDA','AMZN','GOOGL','GOOG','BRK.B','AVGO','META']
write_sp500_data(stock_picks) #write the data to a csv file
data = csv_equal_weight_portfolio('sp500_5year_close_prices.csv') #access the data from the csv file
# print(data[0][0])
trade_logs, tradesTally = track_trades(data[0][0], trade_size=1000)
# print(trade_logs)
# tickers_data = individual_stock_prep_plot(['WBA','AAPL'], halflife_days=20, initial_investment=1000)
tickers_data = [] #for now we will not use this data


combined_data = data + tickers_data 
plot_returns(combined_data, trade_logs=trade_logs, trades_tally=tradesTally) #plot the data
