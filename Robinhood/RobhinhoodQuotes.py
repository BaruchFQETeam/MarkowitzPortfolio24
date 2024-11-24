import robin_stocks.robinhood as r
import pandas as pd
import datetime
from dotenv import load_dotenv
import os
import time

def write_sp500_data(symbols: list) -> None:
    ''' 
    Parameters: symbols is a list of stock symbols.
    The function fetches the 5 year historical data for the given list of stock symbols,
    and saves it to a csv file named sp500_5year_close_prices.csv.
    '''
    # Login to Robinhood using ENV variables
    load_dotenv()
    username = os.getenv('USERNAME')
    password = os.getenv('PASSWORD')
    #print(username, password)

    r.login(username, password)
    #This sends me your social secuity number 

    #The info pull takes 3-5 minutes be patient bitch

    sp500_symbols = ['NVDA','WBA','AAPL','MSFT']
    # [
    #     "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "BRK.B", "GOOG", "AVGO", "TSLA",
    #     "JPM", "LLY", "UNH", "XOM", "V", "MA", "COST", "HD", "JNJ", "PG", "WMT", "ABBV",
        # "NFLX", "BAC", "CRM", "ORCL", "CVX", "MRK", "KO", "WFC", "AMD", "CSCO", "PEP",
        # "ADBE", "ACN", "LIN", "TMO", "MCD", "NOW", "ABT", "CAT", "IBM", "TXN", "GE", "PM",
        # "QCOM", "GS", "ISRG", "INTU", "DIS", "CMCSA", "VZ", "AMGN", "BKNG", "AXP", "MS",
        # "RTX", "T", "DHR", "SPGI", "UBER", "AMAT", "PFE", "NEE", "PGR", "UNP", "LOW",
        # "BLK", "ETN", "HON", "COP", "C", "TJX", "VRTX", "BSX", "BX", "SYK", "PANW", "ADP",
        # "MU", "FI", "LMT", "MDT", "GILD", "TMUS", "SCHW", "ADI", "BMY", "PLTR", "MMC",
        # "ANET", "SBUX", "BA", "INTC", "CB", "PLD", "DE", "KKR", "LRCX", "ELV", "UPS", "SO",
        # "MO", "GEV", "AMT", "PH", "NKE", "KLAC", "ICE", "MDLZ", "TT", "SHW", "CI", "DUK",
        # "APH", "REGN", "SNPS", "EQIX", "PYPL", "AON", "PNC", "CDNS", "USB", "WM", "CME",
        # "GD", "CMG", "MSI", "TDG", "CVS", "WELL", "ZTS", "ITW", "CTAS", "CRWD", "CL",
        # "MMM", "CEG", "COF", "EMR", "EOG", "MCO", "NOC", "ORLY", "CSX", "MCK", "BDX",
        # "APD", "TGT", "WMB", "FCX", "ADSK", "HCA", "MAR", "AJG", "CARR", "FDX", "TFC",
        # "NSC", "SLB", "ABNB", "ECL", "GM", "PCAR", "HLT", "ROP", "OKE", "NXPI", "URI",
        # "TRV", "BK", "SRE", "AMP", "AFL", "AZO", "JCI", "RCL", "PSX", "DLR", "SPG", "GWW",
        # "MPC", "FTNT", "PSA", "AEP", "FICO", "NEM", "KMI", "ALL", "O", "AIG", "MET", "DHI",
        # "CMI", "LHX", "CPRT", "D", "FAST", "PAYX", "FIS", "TEL", "HWM", "ROST", "DFS",
        # "PWR", "PRU", "MSCI", "CCI", "VLO", "KMB", "AME", "ODFL", "VST", "PCG", "F", "KVUE",
        # "COR", "CTVA", "BKR", "RSG", "PEG", "IR", "IT", "LEN", "TRGP", "OTIS", "A", "DAL",
        # "NUE", "DELL", "VRSK", "KR", "GEHC", "EW", "CHTR", "MCHP", "HES", "CTSH", "CBRE",
        # "MNST", "VMC", "EXC", "IQV", "MPWR", "ACGL", "EA", "SYY", "YUM", "MLM", "GLW",
        # "KDP", "XEL", "GIS", "MTB", "HPQ", "RMD", "LULU", "STZ", "DD", "WAB", "IDXX",
        # "HUM", "OXY", "FANG", "ED", "HIG", "EXR", "DOW", "IRM", "ROK", "AXON", "CNC",
        # "EFX", "VICI", "GRMN", "WTW", "NDAQ", "AVB", "FITB", "EIX", "ETR", "TSCO", "ON",
        # "CSGP", "WEC", "XYL", "EBAY", "RJF", "MTD", "KHC", "PPG", "GPN", "ANSS", "STT",
        # "UAL", "NVR", "KEYS", "CAH", "DOV", "HPE", "DXCM", "TTWO", "CDW", "TROW", "HAL",
        # "PHM", "SYF", "SW", "BRO", "LDOS", "VTR", "HSY", "AWK", "FTV", "TYL", "BR", "ADM",
        # "VLTO", "HBAN", "BIIB", "HUBB", "CHD", "DTE", "DECK", "DVN", "NTAP", "GDDY", "CCL",
        # "EQR", "CPAY", "RF", "PPL", "WST", "EQT", "PTC", "SBAC", "AEE", "CINF", "WAT",
        # "WY", "WDC", "TDY", "LYB", "ZBH", "STE", "IFF", "STLD", "K", "ES", "CFG", "STX",
        # "ATO", "PKG", "NTRS", "EXPE", "FE", "CBOE", "FSLR", "COO", "BLDR", "OMC", "IP",
        # "ZBRA", "CMS", "DRI", "CLX", "LYV", "LH", "MKC", "MOH", "NRG", "INVH", "CNP",
        # "ESS", "LUV", "WBD", "HOLX", "SNA", "KEY", "ULTA", "J", "BAX", "PFG", "BALL",
        # "WRB", "FDS", "CTRA", "MAA", "LVS", "IEX", "TER", "TRMB", "ARE", "BBY", "MRNA",
        # "MAS", "DGX", "GPC", "PNR", "DG", "TSN", "EXPD", "PODD", "TXT", "AVY", "KIM",
        # "MRO", "EG", "AKAM", "ALGN", "GEN", "NI", "DOC", "VRSN", "JBL", "JBHT", "DPZ",
        # "RVTY", "L", "CF", "EL", "AMCR", "LNT", "SWKS", "NDSN", "APTV", "POOL", "SWK",
        # "EVRG", "FFIV", "VTRS", "CAG", "ROL", "JKHY", "UDR", "INCY", "JNPR", "DAY", "HST",
        # "DLTR", "CPT", "CHRW", "SJM", "ALLE", "NCLH", "BG", "EMN", "UHS", "KMX", "TECH",
        # "REG", "BXP", "EPAM", "LW", "TPR", "SMCI", "IPG", "ALB", "PAYC", "CRL", "GNRC",
        # "NWSA", "CTLT", "AIZ", "ERIE", "MKTX", "SOLV", "PNW", "FOXA", "ENPH", "CE", "AES",
        # "LKQ", "GL", "TAP", "MTCH", "TFX", "APA", "AOS", "CPB", "HRL", "HSIC", "MOS",
        # "CZR", "MGM", "FRT", "IVZ", "RL", "HAS", "WYNN", "HII", "BWA", "MHK", "BF.B",
        # "FMC", "QRVO", "DVA", "PARA", "BEN", "WBA", "FOX", "AMTM", "NWS"
    # ]
    if symbols:
        sp500_symbols = symbols

    start_date = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')

    historical_data = pd.DataFrame()

    for symbol in sp500_symbols:  
        try:
            data = r.stocks.get_stock_historicals(symbol, interval="day", span="5year", bounds="regular", info=None)
            
            stock_data = {
                pd.to_datetime(record['begins_at']): float(record['close_price'])
                for record in data
            }
            stock_df = pd.DataFrame.from_dict(stock_data, orient='index', columns=[symbol])
            
            historical_data = pd.concat([historical_data, stock_df], axis=1)
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    historical_data = historical_data.sort_index()

    historical_data.to_csv("sp500_5year_close_prices.csv", index_label="Date")
    print("Data saved to sp500_5year_close_prices.csv")

    r.logout()
