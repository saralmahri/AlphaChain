import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import timedelta, datetime
import chardet
from tqdm import tqdm

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set default figure size and style
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titlepad'] = 15
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['font.size'] = 12

# Step 1: Define Investment Parameters
investment_start_date = '2023-01-05'
investment_periods = [1, 2]  # Investment periods in years
initial_investment = 1_000_000
correlation_threshold = 0.5  # Positive correlation threshold
negative_correlation_threshold = -0.3  # Negative correlation threshold
risk_free_rate = 0.01  # 1% annual risk-free rate

# Define TRADING_DAYS_PER_YEAR as a variable
TRADING_DAYS_PER_YEAR = 252  # Approximate number of trading days in a year

# OEM tickers
oem_tickers = ['F', 'GM', 'VOW3.DE', '7267.T', '7203.T']  # OEM tickers in yfinance
oem_supplier_mapping = {
    'F': 'F',
    'GM': 'GM',
    'VOW3.DE': 'XTRA:VOW3',
    '7267.T': 'TSE:7267',
    '7203.T': 'TSE:7203'
}

# Exchange Suffix Mapping for yfinance
exchange_suffix_mapping = {
    'AMEX': '',        # American Stock Exchange
    'ASX': '.AX',      # Australian Securities Exchange
    'BATS': '',        # BATS Global Markets
    'BCBA': '.BA',     # Buenos Aires Stock Exchange
    'BME': '.MC',      # Bolsas y Mercados Españoles (Madrid Stock Exchange)
    'BSE': '.BO',      # Bombay Stock Exchange
    'CVE': '.V',       # TSX Venture Exchange (Canada)
    'FRA': '.F',       # Frankfurt Stock Exchange
    'EPA': '.PA',      # Euronext Paris
    'ETR': '.DE',      # Deutsche Börse Xetra (Germany)
    'HKG': '.HK',      # Hong Kong Stock Exchange
    'ICX': '.IC',      # Iceland Stock Exchange
    'INDEXFTSE': '',   # FTSE Index (UK)
    'INDEXNASDAQ': '', # NASDAQ Index (USA)
    'INDEXSP': '',     # S&P Index (USA)
    'IST': '.IS',      # Istanbul Stock Exchange
    'JSE': '.JO',      # Johannesburg Stock Exchange
    'KRX': '.KS',      # Korea Exchange
    'LON': '.L',       # London Stock Exchange
    'MCX': '.ME',      # Moscow Exchange
    'MUTUALFUND': '',  # Mutual Funds (USA)
    'NASDAQ': '',      # NASDAQ Stock Exchange
    'NSE': '.NS',      # National Stock Exchange of India
    'NYSE': '',        # New York Stock Exchange
    'NZSE': '.NZ',     # New Zealand Stock Exchange
    'SGX': '.SI',      # Singapore Exchange
    'SHA': '.SS',      # Shanghai Stock Exchange
    'SHE': '.SZ',      # Shenzhen Stock Exchange
    'STO': '.ST',      # Stockholm Stock Exchange
    'SWX': '.SW',      # Swiss Exchange
    'TAI': '.TW',      # Taiwan Stock Exchange
    'TLV': '.TA',      # Tel Aviv Stock Exchange
    'TOR': '.TO',      # Toronto Stock Exchange
    'TSE': '.T',       # Tokyo Stock Exchange
    'VIE': '.VI',      # Vienna Stock Exchange
    'XETR': '.DE',     # Deutsche Börse Xetra (Germany)
    'ICE': '.IC',      # Nasdaq Iceland
    'BVMF': '.SA',     # B3 - Brasil Bolsa Balcão (São Paulo Stock Exchange)
    'BMV': '.MX',      # Mexican Stock Exchange
    'BEL': '.BR',      # Euronext Brussels
    'CPH': '.CO',      # Nasdaq Copenhagen
    'HEL': '.HE',      # Nasdaq Helsinki
    'OSL': '.OL',      # Oslo Stock Exchange
    'SGO': '.SN',      # Santiago Stock Exchange
    'AMS': '.AS',      # Euronext Amsterdam
    'LIS': '.LS',      # Euronext Lisbon
    'MIL': '.MI',      # Borsa Italiana (Milan Stock Exchange)
    'KRX': '.KS',      # Korea Exchange
    'KOSDAQ': '.KQ',   # Korean Securities Dealers Automated Quotations
    'TWO': '.TWO',     # Taipei Exchange (Taiwan OTC Exchange)
    'OTC': '.OTC',     # Over-the-Counter Markets
    'PNK': '.PK',      # Pink Sheets (OTC)
    'BUD': '.BD',      # Budapest Stock Exchange
    'WAR': '.WA',      # Warsaw Stock Exchange
    'PRG': '.PR',      # Prague Stock Exchange
    'ATH': '.AT',      # Athens Stock Exchange
    'CSE': '.CN',      # Canadian Securities Exchange
    'IEX': '',         # Investors Exchange (USA)
    'TSXV': '.V',      # TSX Venture Exchange (Canada)
    'IST': '.IS',      # Borsa Istanbul
    'SAU': '.SAU',     # Saudi Stock Exchange (Tadawul)
    'NSEI': '.NS',     # National Stock Exchange of India
    'BSE': '.BO',      # Bombay Stock Exchange
    'HKEX': '.HK',     # Hong Kong Stock Exchange
    'OMX': '.ST',      # Nasdaq Stockholm
    'BCS': '.BC',      # Bolsa de Comercio de Santiago
    'FKA': '.F',       # Frankfurt Stock Exchange
    'VSE': '.VI',      # Vienna Stock Exchange
    'CNQ': '.CN',      # Canadian Securities Exchange
    'AQSE': '.AQSE',   # Aquis Stock Exchange (UK)
    'OTCMKTS': '',     # OTC Markets Group
    'EBS': '.ZU',      # SIX Swiss Exchange (Zurich)
    'JPX': '.T',       # Japan Exchange Group
    'NYSEAMERICAN': '',# NYSE American (formerly AMEX)
    'NYSEARCA': '',    # NYSE ARCA
    'BKK': '.BK',      # Stock Exchange of Thailand
    'TAI': '.TW',      # Taiwan Stock Exchange
    'BIST': '.IS',     # Borsa Istanbul
    'MEX': '.MX',      # Mexican Stock Exchange
    'BRU': '.BR',      # Euronext Brussels
    'STU': '.SG',      # Börse Stuttgart
    'HAM': '.HM',      # Hamburg Stock Exchange
    'HAN': '.HA',      # Hannover Stock Exchange
    'MUN': '.MU',      # Munich Stock Exchange
    'DUS': '.DU',      # Düsseldorf Stock Exchange
    'BER': '.BE',      # Berlin Stock Exchange
    'QUOTEMEDIA': '',  # General placeholder for various exchanges
    'LIT': '.VI',      # Vienna Stock Exchange
    'OSLO': '.OL',     # Oslo Stock Exchange
    'TAE': '.TA',      # Tel Aviv Stock Exchange
    'SAO': '.SA',      # São Paulo Stock Exchange (B3)
    'KLS': '.KL',      # Bursa Malaysia
    'STO': '.ST',      # Nasdaq Stockholm
    'HEL': '.HE',      # Nasdaq Helsinki
    'CPH': '.CO',      # Nasdaq Copenhagen
    'ISE': '.IR',      # Irish Stock Exchange
    'SG': '.SG',       # Börse Stuttgart
    'DE': '.DE',       # Deutsche Börse Xetra
    'JP': '.T',        # Tokyo Stock Exchange
    'HK': '.HK',       # Hong Kong Stock Exchange
    'IN': '.NS',       # National Stock Exchange of India
    'CN': '.SS',       # Shanghai Stock Exchange
    'KS': '.KS',       # Korea Exchange
    'KQ': '.KQ',       # KOSDAQ
    'TWO': '.TWO',     # Taipei Exchange
    'SS': '.SS',       # Shanghai Stock Exchange
    'SZ': '.SZ',       # Shenzhen Stock Exchange
    'TW': '.TW',       # Taiwan Stock Exchange
    'SHG': '.SS',      # Shanghai Stock Exchange
    'NSE': '.NS',      # National Stock Exchange of India
    'BOM': '.BO',      # Bombay Stock Exchange
    'BCBA': '.BA',     # Buenos Aires Stock Exchange
    'SA': '.SA',       # B3 - Brasil Bolsa Balcão
    'MX': '.MX',       # Mexican Stock Exchange
    'JKSE': '.JK',     # Indonesia Stock Exchange
    'KLSE': '.KL',     # Bursa Malaysia
    'BSE': '.BO',      # Bombay Stock Exchange
    'PSE': '.PS',      # Philippine Stock Exchange
    'TADAWUL': '.SAU', # Saudi Stock Exchange
    'SIX': '.SW',      # SIX Swiss Exchange
    'ASX': '.AX',      # Australian Securities Exchange
    'IDX': '.JK',      # Indonesia Stock Exchange
    'TSX': '.TO',      # Toronto Stock Exchange
    'TSXV': '.V',      # TSX Venture Exchange
    'CNSX': '.CN',     # Canadian Securities Exchange
    'BSP': '.PM',      # Philippine Stock Exchange
    'CRYPTO': '-USD',  # Cryptocurrencies (e.g., BTC-USD)
    'FOREX': '=X',     # Foreign Exchange Rates (e.g., EURUSD=X)
}


# Step 2: Fetch Stock Data with Consistent Trading Days
def fetch_data(tickers, start, end):
    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(tickers, start=start, end=end, progress=False)['Adj Close']
    
    # Check if data is empty
    if data.empty:
        print(f"No data fetched for tickers {tickers} between {start} and {end}.")
        return pd.DataFrame()
    
    # Forward-fill and backward-fill missing prices
    data = data.fillna(method='ffill').fillna(method='bfill')
    return data

# Portfolio Optimization
def optimize_weights(returns, method, annualization_factor, bounds=None):
    num_assets = returns.shape[1]
    mean_returns = returns.mean() * annualization_factor
    cov_matrix = returns.cov() * annualization_factor

    if method == "equal":
        weights = np.ones(num_assets) / num_assets
        return weights
    elif method == "sharpe":
        def negative_sharpe(weights):
            port_return = np.sum(mean_returns * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -((port_return - risk_free_rate) / port_volatility)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        if bounds is None:
            bounds = [(0, 1) for _ in range(num_assets)]
        result = minimize(negative_sharpe, np.ones(num_assets) / num_assets, method='SLSQP', bounds=bounds, constraints=constraints)
        return result.x

# Process Supplier Data
def process_supplier_data(oem_ticker, csv_path, training_start_date, investment_end_date):
    with open(csv_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    suppliers = pd.read_csv(csv_path, encoding=encoding).fillna('')

    # Extract ticker symbols from Customer Name
    suppliers['Ticker Extracted'] = suppliers['Customer Name'].str.extract(r'\((?:NYSE:|XTRA:|TSE:)?([^)]+)\)')

    # Check if the extracted ticker matches the OEM ticker
    supplier_oem_ticker = oem_supplier_mapping.get(oem_ticker, oem_ticker)

    # Filter relevant suppliers based on the extracted ticker
    relevant_suppliers = suppliers[suppliers['Ticker Extracted'] == supplier_oem_ticker.split(':')[-1]]
    relevant_suppliers = relevant_suppliers[relevant_suppliers['Exchange:Ticker Symbol'] != '-']

    supplier_tickers = []
    failed_tickers = []

    for _, row in tqdm(relevant_suppliers.iterrows(), total=relevant_suppliers.shape[0]):
        exchange_ticker = row['Exchange:Ticker Symbol']
        if ':' in exchange_ticker:
            exchange, ticker = exchange_ticker.split(':', 1)
            suffix = exchange_suffix_mapping.get(exchange.strip().upper())
            if suffix is not None:
                adjusted_ticker = ticker.strip() + suffix
                try:
                    # Fetch data to validate ticker
                    test_data = fetch_data([adjusted_ticker], training_start_date, investment_end_date)
                    if not test_data.empty:
                        supplier_tickers.append(adjusted_ticker)
                    else:
                        failed_tickers.append(adjusted_ticker)
                except Exception:
                    failed_tickers.append(adjusted_ticker)
            else:
                failed_tickers.append(exchange_ticker)
        else:
            failed_tickers.append(exchange_ticker)
    
    print(f"For OEM {oem_ticker}, added suppliers: {len(supplier_tickers)}, failed: {len(failed_tickers)}")
    return supplier_tickers, relevant_suppliers

# Perform Analysis
def perform_analysis(period_years):
    print(f"\n=== Performing analysis for {period_years}-year investment period ===\n")
    investment_start = pd.to_datetime(investment_start_date)
    investment_end = investment_start + pd.DateOffset(years=period_years)
    investment_end_date = investment_end.strftime('%Y-%m-%d')

    # Define training start date with buffer
    buffer_days = 30  # Additional buffer days to account for any initial missing data
    training_start = investment_start - pd.DateOffset(years=period_years) - timedelta(days=buffer_days)
    training_start_date = training_start.strftime('%Y-%m-%d')

    # Fetch OEM data
    print("Fetching OEM data...")
    oem_data = fetch_data(oem_tickers, training_start_date, investment_end_date)
    if oem_data.empty:
        print("No OEM data available for the given date range.")
        return

    supplier_tickers = []
    supplier_details = pd.DataFrame()
    for oem_ticker in oem_tickers:
        csv_path = f"G:/My Drive/SarAI/Demo/Investment Scenario/{oem_ticker}_suppliers.csv"
        suppliers, details = process_supplier_data(oem_ticker, csv_path, training_start_date, investment_end_date)
        supplier_tickers += suppliers
        supplier_details = pd.concat([supplier_details, details])

    # Remove duplicates
    supplier_tickers = list(set(supplier_tickers))
    print(f"Total unique suppliers: {len(supplier_tickers)}")

    # Fetch Supplier data
    print("Fetching Supplier data...")
    supplier_data = fetch_data(supplier_tickers, training_start_date, investment_end_date)
    if supplier_data.empty:
        print("No supplier data available for the given date range.")
        return

    # Combine OEM and Supplier data
    all_data = pd.concat([oem_data, supplier_data], axis=1)

    # Calculate returns
    returns = all_data.pct_change()

    # Split returns into training and investment periods
    training_returns = returns.loc[training_start_date:investment_start_date]
    investment_returns = returns.loc[investment_start_date:investment_end_date]

    # Adjust investment start date if data is not available for some stocks
    # Find the earliest date where data is available for all stocks
    first_valid_dates = investment_returns.apply(lambda col: col.first_valid_index())
    adjusted_investment_start_date = first_valid_dates.max()
    if pd.isnull(adjusted_investment_start_date):
        print("No valid data available for investment period.")
        return

    investment_returns = investment_returns.loc[adjusted_investment_start_date:]

    # Recalculate the number of investment days
    num_investment_days = investment_returns.shape[0]
    if num_investment_days == 0:
        print("No investment returns available after adjusting for data availability.")
        return

    # Annualization factor
    investment_annualization_factor = num_investment_days / period_years

    # Calculate correlations with OEMs
    positive_suppliers = set()
    negative_suppliers = set()
    for oem_ticker in oem_tickers:
        if oem_ticker in training_returns.columns:
            oem_return = training_returns[oem_ticker]
            supplier_corr = training_returns[supplier_data.columns].corrwith(oem_return)
            pos_corr = supplier_corr[supplier_corr > correlation_threshold]
            neg_corr = supplier_corr[supplier_corr < negative_correlation_threshold]
            positive_suppliers.update(pos_corr.index.tolist())
            negative_suppliers.update(neg_corr.index.tolist())

    print(f"Positive Correlated Suppliers: {len(positive_suppliers)}")
    print(f"Negative Correlated Suppliers: {len(negative_suppliers)}")

    # Create Supply Chain-Enriched Portfolio
    supply_chain_tickers = list(set(oem_tickers + list(positive_suppliers) + list(negative_suppliers)))
    supply_chain_returns = investment_returns[supply_chain_tickers]

    # Handle missing data by filling NaN returns with zero (assuming no return on missing data)
    supply_chain_returns = supply_chain_returns.fillna(0)

    # Non-Supply Chain Portfolio (Only OEMs)
    non_supply_chain_returns = investment_returns[oem_tickers]
    non_supply_chain_returns = non_supply_chain_returns.fillna(0)

    # ====================== Portfolio Optimization with Constraints ======================

    # Define bounds for Non Supply Chain Portfolio (minimum weight of 0.05 for each OEM)
    num_oems = len(oem_tickers)
    bounds_non_supply_chain = [(0.05, 1) for _ in range(num_oems)]

    # Optimize Non Supply Chain Portfolio
    sharpe_weights_non_supply_chain = optimize_weights(
        non_supply_chain_returns,
        "sharpe",
        investment_annualization_factor,
        bounds=bounds_non_supply_chain
    )

    # Normalize weights to ensure they sum to 1 (in case of numerical issues)
    sharpe_weights_non_supply_chain /= sharpe_weights_non_supply_chain.sum()

    # Define bounds for Supply Chain-Enriched Portfolio
    num_assets_supply_chain = len(supply_chain_tickers)
    bounds_supply_chain = []

    # Identify indices of OEMs and suppliers
    for ticker in supply_chain_returns.columns:
        if ticker in oem_tickers:
            bounds_supply_chain.append((0.05, 1))  # Minimum weight of 0.05 for OEMs
        else:
            bounds_supply_chain.append((0, 1))     # No minimum weight for suppliers

    # Optimize Supply Chain-Enriched Portfolio
    sharpe_weights_supply_chain = optimize_weights(
        supply_chain_returns,
        "sharpe",
        investment_annualization_factor,
        bounds=bounds_supply_chain
    )

    # Normalize weights to ensure they sum to 1
    sharpe_weights_supply_chain /= sharpe_weights_supply_chain.sum()

    # Exclude suppliers with weight less than 0.03
    weights_df = pd.DataFrame({
        'Ticker': supply_chain_returns.columns,
        'Weight': sharpe_weights_supply_chain
    })

    low_weight_suppliers = weights_df[(weights_df['Weight'] < 0.03) & (~weights_df['Ticker'].isin(oem_tickers))]
    if not low_weight_suppliers.empty:
        print(f"Excluding {len(low_weight_suppliers)} suppliers with weights less than 0.03.")

        # Remove low-weight suppliers
        supply_chain_returns_filtered = supply_chain_returns.drop(columns=low_weight_suppliers['Ticker'])
        sharpe_weights_supply_chain_filtered = optimize_weights(
            supply_chain_returns_filtered,
            "sharpe",
            investment_annualization_factor,
            bounds=[bounds_supply_chain[i] for i, ticker in enumerate(supply_chain_returns.columns) if ticker not in low_weight_suppliers['Ticker'].values]
        )

        # Normalize weights
        sharpe_weights_supply_chain_filtered /= sharpe_weights_supply_chain_filtered.sum()

        # Update weights_df
        weights_df_filtered = pd.DataFrame({
            'Ticker': supply_chain_returns_filtered.columns,
            'Weight': sharpe_weights_supply_chain_filtered
        })

        # Update variables
        sharpe_weights_supply_chain = sharpe_weights_supply_chain_filtered
        supply_chain_returns = supply_chain_returns_filtered
        weights_df = weights_df_filtered
    else:
        print("No suppliers with weights less than 0.03 to exclude.")

    # ====================== End of Portfolio Optimization with Constraints ======================

    # Calculate Portfolio Returns
    portfolio_sharpe_supply_chain = supply_chain_returns.dot(sharpe_weights_supply_chain)
    portfolio_sharpe_non_supply_chain = non_supply_chain_returns.dot(sharpe_weights_non_supply_chain)

    # Cumulative Returns
    cumulative_returns_supply_chain = (1 + portfolio_sharpe_supply_chain).cumprod()
    cumulative_returns_non_supply_chain = (1 + portfolio_sharpe_non_supply_chain).cumprod()

    # Check if cumulative returns are empty
    if cumulative_returns_supply_chain.empty or cumulative_returns_non_supply_chain.empty:
        print("Cumulative returns are empty. Cannot proceed with plotting.")
        return

    # Calculate total returns (Moved before export code)
    total_return_supply_chain = cumulative_returns_supply_chain.iloc[-1] - 1
    total_return_non_supply_chain = cumulative_returns_non_supply_chain.iloc[-1] - 1

    # Export Results
    export_directory = os.path.join(os.getcwd(), "exports")
    if not os.path.exists(export_directory):
        os.makedirs(export_directory)

    # Save portfolio weights for both portfolios
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ====================== Export Detailed Portfolio Information ======================

    # Prepare data for Supply Chain-Enriched Portfolio
    supply_chain_portfolio_df = pd.DataFrame({
        'Ticker': supply_chain_returns.columns,
        'Weight': sharpe_weights_supply_chain
    })

    # Determine roles and supplier relationships
    supply_chain_portfolio_df['Role'] = supply_chain_portfolio_df['Ticker'].apply(
        lambda x: 'OEM' if x in oem_tickers else 'Supplier'
    )

    # Create a mapping of supplier tickers to OEMs
    supplier_to_oems = {}
    for oem_ticker in oem_tickers:
        # Get the suppliers for this OEM
        csv_path = f"G:/My Drive/SarAI/Demo/Investment Scenario/{oem_ticker}_suppliers.csv"
        with open(csv_path, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
        suppliers = pd.read_csv(csv_path, encoding=encoding).fillna('')

        # Extract ticker symbols from Customer Name
        suppliers['Ticker Extracted'] = suppliers['Customer Name'].str.extract(r'\((?:NYSE:|XTRA:|TSE:)?([^)]+)\)')

        # Filter relevant suppliers based on the extracted ticker
        supplier_oem_ticker = oem_supplier_mapping.get(oem_ticker, oem_ticker)
        relevant_suppliers = suppliers[suppliers['Ticker Extracted'] == supplier_oem_ticker.split(':')[-1]]
        relevant_suppliers = relevant_suppliers[relevant_suppliers['Exchange:Ticker Symbol'] != '-']

        for _, row in relevant_suppliers.iterrows():
            exchange_ticker = row['Exchange:Ticker Symbol']
            if ':' in exchange_ticker:
                exchange, ticker = exchange_ticker.split(':', 1)
                suffix = exchange_suffix_mapping.get(exchange.strip().upper())
                if suffix is not None:
                    adjusted_ticker = ticker.strip() + suffix
                    supplier_to_oems.setdefault(adjusted_ticker, set()).add(oem_ticker)

    # Function to get the OEM(s) a supplier supplies
    def get_supplier_oems(ticker):
        oems = supplier_to_oems.get(ticker, set())
        return ', '.join(oems)

    supply_chain_portfolio_df['Supplier of'] = supply_chain_portfolio_df.apply(
        lambda row: get_supplier_oems(row['Ticker']) if row['Role'] == 'Supplier' else '', axis=1
    )

    # Attempt to get company names using yfinance
    def get_company_name(ticker):
        try:
            info = yf.Ticker(ticker).info
            return info.get('longName') or info.get('shortName') or ''
        except Exception:
            return ''

    supply_chain_portfolio_df['Company Name'] = supply_chain_portfolio_df['Ticker'].apply(get_company_name)

    # Reorder columns
    supply_chain_portfolio_df = supply_chain_portfolio_df[['Ticker', 'Company Name', 'Role', 'Supplier of', 'Weight']]

    # Prepare data for Non-Supply Chain Portfolio
    non_supply_chain_portfolio_df = pd.DataFrame({
        'Ticker': non_supply_chain_returns.columns,
        'Weight': sharpe_weights_non_supply_chain
    })

    non_supply_chain_portfolio_df['Role'] = 'OEM'  # Only OEMs in this portfolio
    non_supply_chain_portfolio_df['Supplier of'] = ''  # No suppliers

    # Get company names
    non_supply_chain_portfolio_df['Company Name'] = non_supply_chain_portfolio_df['Ticker'].apply(get_company_name)

    # Reorder columns
    non_supply_chain_portfolio_df = non_supply_chain_portfolio_df[['Ticker', 'Company Name', 'Role', 'Supplier of', 'Weight']]

    # Add summary information
    investment_period_str = f"{adjusted_investment_start_date.strftime('%Y-%m-%d')} to {investment_end_date}"
    summary_info = pd.DataFrame({
        'Investment Period': [investment_period_str],
        'Total Companies in Supply Chain Portfolio': [len(supply_chain_portfolio_df)],
        'Total Companies in Non Supply Chain Portfolio': [len(non_supply_chain_portfolio_df)],
        'Total Return Supply Chain Portfolio (%)': [total_return_supply_chain * 100],
        'Total Return Non Supply Chain Portfolio (%)': [total_return_non_supply_chain * 100]
    })

    # Export to Excel with multiple sheets
    export_file = os.path.join(export_directory, f"Portfolio_Details_{timestamp}.xlsx")

    with pd.ExcelWriter(export_file) as writer:
        # Write summary information
        summary_info.to_excel(writer, sheet_name='Summary', index=False)
        
        # Write supply chain portfolio details
        supply_chain_portfolio_df.to_excel(writer, sheet_name='Supply Chain Portfolio', index=False)
        
        # Write non-supply chain portfolio details
        non_supply_chain_portfolio_df.to_excel(writer, sheet_name='Non Supply Chain Portfolio', index=False)

    print(f"Detailed portfolio information exported to {export_file}.")

    # ====================== End of Export Code ======================

    print(f"Processed {len(oem_tickers)} OEMs and {len(supplier_tickers)} suppliers.")
    print(f"Positive Correlations: {len(positive_suppliers)}, Negative Correlations: {len(negative_suppliers)}.")

    # Plotting Cumulative Returns
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_returns_supply_chain.index, cumulative_returns_supply_chain.values, label='Supply Chain-Enriched Portfolio', color='#1f77b4', linewidth=2)
    plt.plot(cumulative_returns_non_supply_chain.index, cumulative_returns_non_supply_chain.values, label='Non Supply Chain Portfolio', color='#ff7f0e', linewidth=2)
    plt.title(f"Cumulative Returns Comparison ({period_years}-Year Investment Period)", fontsize=18, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Returns", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Percentage Increase Bar Plot
    # total_return_supply_chain and total_return_non_supply_chain are already calculated above

    # Handle division by zero
    if total_return_non_supply_chain == 0:
        percentage_increase = np.nan
        print("Total return for Non Supply Chain portfolio is zero. Cannot calculate percentage increase.")
    else:
        percentage_increase = ((total_return_supply_chain - total_return_non_supply_chain) / abs(total_return_non_supply_chain)) * 100

    plt.figure(figsize=(10, 6))
    bar_colors = ['#ff7f0e', '#1f77b4']  # Consistent colors
    plt.bar(['Non Supply Chain', 'Supply Chain-Enriched'], [total_return_non_supply_chain * 100, total_return_supply_chain * 100], color=bar_colors, alpha=0.9)
    if not np.isnan(percentage_increase):
        plt.title(f"Total Portfolio Return Comparison\nSupply Chain-Enriched Return is {percentage_increase:.2f}% Higher", fontsize=16, fontweight='bold')
    else:
        plt.title("Total Portfolio Return Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Portfolio Type", fontsize=14)
    plt.ylabel("Total Return (%)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Run Analysis
for period in investment_periods:
    perform_analysis(period)
