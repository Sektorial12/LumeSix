import os
import pandas as pd
from dotenv import load_dotenv
from nixtla import NixtlaClient
import logging

# --- Setup Logger ---
logger = logging.getLogger(__name__)

# Load environment variables from .env file
# Ensure .env is in the project root (lumesix)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)

NIXTLA_API_KEY = os.getenv('NIXTLA_API_KEY')

# --- TimeGPT Initialization ---
def get_timegpt_client():
    """Initializes and returns the TimeGPT client."""
    if not NIXTLA_API_KEY:
        logger.error("Error: NIXTLA_API_KEY not found in environment variables.")
        logger.info(f"Looked for .env at: {dotenv_path}")
        return None
    try:
        logger.info("Initializing Nixtla client...")
        timegpt = NixtlaClient(api_key=NIXTLA_API_KEY)
        # Optional: Test connection with a simple method if available
        # For example, listing models (if the API/client supports it easily)
        # logger.info(timegpt.validate_api_key()) # Hypothetical validation
        logger.info("Nixtla client initialized successfully.")
        return timegpt
    except Exception as e:
        logger.error(f"Error initializing Nixtla client: {e}", exc_info=True)
        return None

# --- Forecasting Function ---
def get_timegpt_forecast(client, df, horizon, freq='5min', target_col='close'):
    """
    Generates a forecast using the Nixtla API.

    Args:
        client (NixtlaClient): The initialized Nixtla client.
        df (pd.DataFrame): Input DataFrame with historical data.
                           Requires a datetime index or column named 'ds'
                           and a target column named 'y'.
        horizon (int): The number of steps to forecast ahead.
        freq (str): The frequency of the time series data (e.g., '5min', '1h').
        target_col (str): The name of the column in 'df' to forecast.

    Returns:
        pd.DataFrame: A DataFrame containing the forecast, or None if an error occurs.
    """
    if client is None:
        logger.error("Error: Nixtla client is not initialized.")
        return None
    if df is None or df.empty:
        logger.error("Error: Input DataFrame is empty or None.")
        return None

    # --- Prepare DataFrame for Nixtla ---
    # Nixtla expects columns 'ds' (datetime) and 'y' (target variable)
    # It also needs a 'unique_id' column if forecasting multiple series (we have one: the symbol)
    df_prep = df.copy()
    
    # 1. Ensure datetime column exists and is named 'ds'
    if 'ds' in df_prep.columns:
        # If 'ds' already exists, ensure it's datetime
        df_prep['ds'] = pd.to_datetime(df_prep['ds'])
        logger.info("Using existing 'ds' column.")
    elif 'datetime' in df_prep.columns:
        df_prep['ds'] = pd.to_datetime(df_prep['datetime'])
        logger.info("Converting 'datetime' column to 'ds'.")
    elif 'timestamp' in df_prep.columns:
         # Assuming timestamp is milliseconds Unix epoch
        df_prep['ds'] = pd.to_datetime(df_prep['timestamp'], unit='ms')
        logger.info("Converting 'timestamp' column (ms) to 'ds'.")
    else:
        logger.error("Error: DataFrame must contain a 'ds', 'datetime', or 'timestamp' column.")
        return None

    # 2. Ensure target column is named 'y'
    if target_col not in df_prep.columns:
        logger.error(f"Error: Target column '{target_col}' not found in DataFrame.")
        return None
    df_prep['y'] = df_prep[target_col]

    # 3. Add 'unique_id' (e.g., the trading symbol)
    # We'll assume a single symbol for now. If multiple, this needs adjustment.
    df_prep['unique_id'] = 'series_0' # Placeholder ID

    # 4. Select required columns
    required_cols = ['unique_id', 'ds', 'y']
    df_timegpt = df_prep[required_cols]

    # 5. Sort by time
    df_timegpt = df_timegpt.sort_values(by='ds')

    logger.info(f"DataFrame prepared for Nixtla. Shape: {df_timegpt.shape}, Horizon: {horizon}, Freq: {freq}")
    logger.info("Nixtla Input DataFrame Head:")
    logger.info(f"\n{df_timegpt.head().to_string()}") # Log DataFrame as string

    # --- Fill potential NaN values --- 
    # Add this step before calling the API
    initial_nan_count = df_timegpt['y'].isna().sum()
    if initial_nan_count > 0:
        logger.warning(f"Warning: Found {initial_nan_count} NaN values in target column 'y'. Filling using ffill/bfill.")
        df_timegpt['y'] = df_timegpt['y'].ffill().bfill() # Forward fill, then backfill for safety
        if df_timegpt['y'].isna().sum() > 0:
             logger.error("Error: Still found NaNs after filling. Check data source.")
             return None

    # --- Call Nixtla API ---
    try:
        logger.info(f"Requesting forecast from Nixtla API for {horizon} steps...")
        # Note: Finetuning, level parameters, etc., can be added later
        forecast_df = client.forecast(df=df_timegpt, h=horizon, freq=freq)
        logger.info("Forecast received successfully.")
        return forecast_df
    except Exception as e:
        logger.error(f"Error during Nixtla forecast API call: {e}", exc_info=True)
        # Add more specific error handling based on nixtla exceptions if available
        return None

# --- Example Usage (Placeholder) ---
if __name__ == '__main__':
    # Setup basic logging for standalone script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("--- Testing Nixtla Forecaster Module with Real Bybit Data ---")
    
    # Import Bybit functions
    # Assumes script is run from the project root (lumesix)
    try:
        from src.connectors.bybit_connector import get_bybit_client, fetch_ohlcv, DEFAULT_TRADING_PAIR, DEFAULT_TIMEFRAME
    except ImportError:
        logger.error("Error: Could not import Bybit connector functions.")
        logger.info("Ensure you are running this script from the project root directory (lumesix).")
        exit()

    # 1. Initialize Clients
    nixtla_client = get_timegpt_client()
    bybit_client = get_bybit_client() # Use defaults (including testnet from .env)

    if nixtla_client and bybit_client:
        logger.info("\n--- Fetching Real Market Data from Bybit ---")
        # Fetch recent data (e.g., last 200 5-min candles = ~16 hours)
        ohlcv_df = fetch_ohlcv(
            client=bybit_client,
            symbol=DEFAULT_TRADING_PAIR, 
            timeframe=DEFAULT_TIMEFRAME, # Default is '5m'
            limit=200 # Fetch enough data for the model
        )

        if ohlcv_df is not None and not ohlcv_df.empty:
            logger.info(f"Fetched {len(ohlcv_df)} records for {DEFAULT_TRADING_PAIR} ({DEFAULT_TIMEFRAME}).")
            logger.info("Fetched DataFrame (tail):")
            logger.info(f"\n{ohlcv_df.tail().to_string()}") # Log DataFrame as string

            # 2. Get Forecast using fetched data
            logger.info("\n--- Generating Forecast with Nixtla ---")
            forecast_horizon = 12 # Forecast next 12 steps (1 hour for 5min timeframe)
            forecast_result = get_timegpt_forecast(
                client=nixtla_client,
                df=ohlcv_df, # Pass the fetched OHLCV data
                horizon=forecast_horizon,
                freq=DEFAULT_TIMEFRAME.replace('m','min'), # Convert '5m' to '5min' for Nixtla
                target_col='close' # OHLCV data has a 'close' column
            )

            if forecast_result is not None:
                logger.info("\nForecast Result:")
                logger.info(f"\n{forecast_result.to_string()}") # Log DataFrame as string
            else:
                logger.error("\nFailed to get forecast from Nixtla.")
        else:
            logger.error("\nFailed to fetch OHLCV data from Bybit.")
    else:
        logger.error("\nSkipping forecast test due to client initialization failure.")
        if not nixtla_client:
            logger.error("- Nixtla client failed to initialize.")
        if not bybit_client:
            logger.error("- Bybit client failed to initialize.")

    logger.info("\n--- Nixtla Forecaster Module Test Complete ---")
