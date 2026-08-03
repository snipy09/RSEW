"""
Data loading module for fetching and preprocessing market data.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Optional


class DataLoader:
    """Load and preprocess market data from yfinance."""

    def __init__(self, symbol: str = 'SPY', period_years: int = 10):
        """
        Initialize DataLoader.

        Parameters:
        -----------
        symbol : str, default='SPY'
            Stock symbol (e.g., 'SPY', 'NIFTY' formats supported)
        period_years : int, default=10
            Number of years of historical data to fetch
        """
        self.symbol = symbol
        self.period_years = period_years
        self.data = None

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical price data from yfinance.

        Returns:
        --------
        data : pd.DataFrame
            DataFrame with columns: Date, Open, High, Low, Close, Volume, Adj Close
        """
        print(f"Fetching {self.period_years} years of {self.symbol} data...")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.period_years)

        # Fetch data
        try:
            data = yf.download(
                self.symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise

        # Reset index to make Date a column
        data.reset_index(inplace=True)

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns
            data.columns = [col[0] if isinstance(
                col, tuple) else col for col in data.columns]

        # Handle column naming (yfinance sometimes uses different names)
        try:
            data.columns = [str(col).lower() for col in data.columns]
        except:
            pass

        # Remove rows with NaN prices
        data = data.dropna(subset=['close', 'volume'])

        # Ensure date is datetime
        data['date'] = pd.to_datetime(data['date'])

        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)

        self.data = data
        print(
            f"Loaded {len(data)} trading days from {data['date'].min()} to {data['date'].max()}")

        return data

    def get_price_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get price series and dates.

        Returns:
        --------
        prices : np.ndarray
            Closing prices
        dates : np.ndarray
            Dates corresponding to prices
        """
        if self.data is None:
            self.fetch_data()

        prices = self.data['close'].values
        dates = self.data['date'].values

        return prices, dates

    def get_ohlcv_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get OHLCV data and dates.

        Returns:
        --------
        open_prices, high_prices, low_prices, close_prices, volumes, dates : tuple of np.ndarray
        """
        if self.data is None:
            self.fetch_data()

        return (
            self.data['open'].values,
            self.data['high'].values,
            self.data['low'].values,
            self.data['close'].values,
            self.data['volume'].values,
            self.data['date'].values
        )

    def add_volume_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-related features.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with 'volume' column

        Returns:
        --------
        data : pd.DataFrame
            DataFrame with volume features added
        """
        # Volume change
        data['volume_change'] = data['volume'].pct_change()

        # Volume MA ratio
        data['volume_ma_10'] = data['volume'].rolling(10).mean()
        data['volume_ma_ratio'] = data['volume'] / \
            (data['volume_ma_10'] + 1e-8)

        return data

    @staticmethod
    def validate_data(data: pd.DataFrame) -> bool:
        """
        Validate data quality.

        Parameters:
        -----------
        data : pd.DataFrame
            Data to validate

        Returns:
        --------
        is_valid : bool
            True if data passes validation
        """
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']

        # Check columns
        for col in required_columns:
            if col not in data.columns:
                print(f"Missing column: {col}")
                return False

        # Check for sufficient data
        if len(data) < 365:
            print(f"Insufficient data: {len(data)} rows (need at least 365)")
            return False

        # Check for gaps in dates
        date_diffs = data['date'].diff().dt.days
        if (date_diffs > 3).sum() > 5:  # Allow a few gaps, but not many
            print("Possible data gaps detected")

        return True


def fetch_market_data(
    symbol: str = 'SPY',
    period_years: int = 10,
    validate: bool = True
) -> pd.DataFrame:
    """
    Convenience function to fetch market data.

    Parameters:
    -----------
    symbol : str, default='SPY'
        Stock symbol
    period_years : int, default=10
        Period in years
    validate : bool, default=True
        Validate data after loading

    Returns:
    --------
    data : pd.DataFrame
        Historical price data
    """
    loader = DataLoader(symbol=symbol, period_years=period_years)
    data = loader.fetch_data()

    if validate:
        if not loader.validate_data(data):
            print("Warning: Data validation failed!")

    return data
