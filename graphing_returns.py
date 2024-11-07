import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv('sp500_5year_close_prices.csv')
df = df.set_index('Date')
df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')

# Define tickers
tickers = ['NVDA', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL', 'BRK.B', 'GOOG', 'AVGO', 'TSLA',
           'JPM', 'LLY', 'UNH', 'XOM', 'V', 'MA', 'COST', 'HD', 'JNJ', 'PG', 'WMT', 'ABBV',
           'NFLX', 'BAC', 'CRM', 'ORCL', 'CVX', 'MRK', 'KO', 'WFC', 'AMD', 'CSCO', 'PEP',
           'ADBE', 'ACN', 'LIN', 'TMO', 'MCD', 'NOW', 'ABT', 'CAT', 'IBM', 'TXN', 'GE', 'PM',
           'QCOM', 'GS', 'ISRG', 'INTU', 'DIS', 'CMCSA', 'VZ', 'AMGN', 'BKNG', 'AXP', 'MS',
           'RTX', 'T', 'DHR', 'SPGI', 'UBER', 'AMAT', 'PFE', 'NEE', 'PGR', 'UNP', 'LOW', 'BLK',
           'ETN', 'HON', 'COP', 'C', 'TJX', 'VRTX', 'BSX', 'BX', 'SYK', 'PANW', 'ADP', 'MU',
           'FI', 'LMT', 'MDT', 'GILD', 'TMUS', 'SCHW', 'ADI', 'BMY', 'PLTR', 'MMC', 'ANET',
           'SBUX', 'BA', 'INTC', 'CB', 'PLD', 'DE', 'KKR', 'LRCX', 'ELV', 'UPS', 'SO', 'MO',
           'GEV', 'AMT', 'PH', 'NKE', 'KLAC', 'ICE', 'MDLZ', 'TT', 'SHW', 'CI', 'DUK', 'APH',
           'REGN', 'SNPS', 'EQIX', 'PYPL', 'AON', 'PNC', 'CDNS', 'USB', 'WM', 'CME', 'GD',
           'CMG', 'MSI', 'TDG', 'CVS', 'WELL', 'ZTS', 'ITW', 'CTAS', 'CRWD', 'CL', 'MMM',
           'CEG', 'COF', 'EMR', 'EOG', 'MCO', 'NOC', 'ORLY', 'CSX', 'MCK', 'BDX', 'APD',
           'TGT', 'WMB', 'FCX', 'ADSK', 'HCA', 'MAR', 'AJG', 'CARR', 'FDX', 'TFC', 'NSC',
           'SLB', 'ABNB', 'ECL', 'GM', 'PCAR', 'HLT', 'ROP', 'OKE', 'NXPI', 'URI', 'TRV',
           'BK', 'SRE', 'AMP', 'AFL', 'AZO', 'JCI', 'RCL', 'PSX', 'DLR', 'SPG', 'GWW', 'MPC',
           'FTNT', 'PSA', 'AEP', 'FICO', 'NEM', 'KMI', 'ALL', 'O', 'AIG', 'MET', 'DHI',
           'CMI', 'LHX', 'CPRT', 'D', 'FAST', 'PAYX', 'FIS']

# Ensure tickers are in the dataframe
available_tickers = [ticker for ticker in tickers if ticker in df.columns]
df = df[available_tickers]

# Generate random weights and normalize them for Portfolio 1
random_weights = np.random.random(len(available_tickers))
normalized_weights = random_weights / random_weights.sum()
ticker_weights = {ticker: weight for ticker, weight in zip(available_tickers, normalized_weights)}

# Generate random weights and normalize them for Portfolio 2
random_weights2 = np.random.random(len(available_tickers))
normalized_weights2 = random_weights2 / random_weights2.sum()
ticker_weights2 = {ticker: weight for ticker, weight in zip(available_tickers, normalized_weights2)}

# Calculate weighted portfolio values
portfolio_values = df.mul([ticker_weights[ticker] for ticker in df.columns], axis=1).sum(axis=1)
portfolio_values2 = df.mul([ticker_weights2[ticker] for ticker in df.columns], axis=1).sum(axis=1)

# Calculate cumulative returns percentage for both portfolios
cumulative_returns = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
cumulative_returns2 = (portfolio_values2 / portfolio_values2.iloc[0] - 1) * 100

# Calculate cumulative returns percentage for Nvidia (NVDA)
nvidia_cumulative_returns = (df["NVDA"] / df["NVDA"].iloc[0] - 1) * 100
aapl_cumulative_returns = (df["AAPL"] / df["AAPL"].iloc[0] - 1) * 100

# Plot the cumulative returns percentage
plt.figure(figsize=(12, 7))
cumulative_returns.plot(label="Portfolio 1")
cumulative_returns2.plot(label="Portfolio 2")
# nvidia_cumulative_returns.plot(label="Nvidia (NVDA)", linestyle="--", color="green")
aapl_cumulative_returns.plot(label="Apple (AAPL)", linestyle="--", color="red")
plt.title("Compounded Returns Percentage Over Time")
plt.xlabel("Date")
plt.xticks(ticks=range(0, len(df.index), 60), labels=df.index[::60], rotation=45)
plt.ylabel("Cumulative Return (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
