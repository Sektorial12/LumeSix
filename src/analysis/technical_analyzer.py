import pandas as pd
import pandas_ta as ta
import logging

# --- Setup Logger ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---


def add_indicators(df, ta_config):
    """
    Adds common technical indicators to the OHLCV DataFrame using configured parameters.

    Args:
        df (pd.DataFrame): DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        ta_config (dict): Configuration dictionary for technical indicators, e.g., 
                          {
                              'rsi': {'length': 14},
                              'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                              'emas': {'short_period': 20, 'long_period': 50},
                              'atr': {'length': 14}
                          }

    Returns:
        pd.DataFrame: The original DataFrame with added indicator columns, 
                     or the original DataFrame if an error occurs or input is invalid.
    """
    if df is None or df.empty:
        logger.error("TA: Input DataFrame is empty or None. Skipping indicator calculation.")
        return df

    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        logger.error("TA: Input DataFrame missing required columns (open, high, low, close, volume).")
        return df

    try:
        logger.info(f"TA: Calculating indicators with config: RSI({ta_config['rsi']['length']}), MACD({ta_config['macd']['fast_period']},{ta_config['macd']['slow_period']},{ta_config['macd']['signal_period']}), EMAs({ta_config['emas']['short_period']},{ta_config['emas']['long_period']}), ATR({ta_config['atr']['length']})...")
        
        # Calculate RSI
        df.ta.rsi(length=ta_config['rsi']['length'], append=True) 

        # Calculate MACD
        df.ta.macd(fast=ta_config['macd']['fast_period'], 
                   slow=ta_config['macd']['slow_period'], 
                   signal=ta_config['macd']['signal_period'], 
                   append=True)

        # Calculate EMAs
        # Ensure EMA names are unique if periods are the same, though pandas_ta handles this by default (e.g., EMA_20)
        df.ta.ema(length=ta_config['emas']['short_period'], append=True) 
        df.ta.ema(length=ta_config['emas']['long_period'], append=True)
        
        # Calculate ATR
        df.ta.atr(length=ta_config['atr']['length'], append=True)

        # Calculate Bollinger Bands (optional, uncomment and configure if needed)
        # if 'bollinger_bands' in ta_config:
        #     df.ta.bbands(length=ta_config['bollinger_bands']['length'], 
        #                   std=ta_config['bollinger_bands']['std_dev'], 
        #                   append=True)

        logger.info("TA: Indicators calculated successfully.")
        return df

    except Exception as e:
        logger.error(f"TA: An error occurred during indicator calculation: {e}")
        # Return the original DataFrame without crashing
        return df

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    logger.info("Testing technical_analyzer module...")
    # Create a dummy DataFrame similar to what fetch_ohlcv would return
    # Using realistic-looking price data
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='5min').astype(int) // 10**6, # Timestamps in ms
        'open': [100 + i*0.1 + ((i % 5 - 2) * 0.2) for i in range(100)],
        'high': [100 + i*0.1 + 0.5 + ((i % 3) * 0.3) for i in range(100)],
        'low': [100 + i*0.1 - 0.3 - ((i % 4) * 0.2) for i in range(100)],
        'close': [100 + i*0.1 + 0.1 + ((i % 6 - 3) * 0.15) for i in range(100)],
        'volume': [1000 + i*10 + ((i % 10 - 5) * 50) for i in range(100)]
    }
    dummy_df = pd.DataFrame(data)
    # Ensure no negative volumes just in case
    dummy_df['volume'] = dummy_df['volume'].abs()
    # Ensure low is the minimum and high is the maximum of ohlc for the period
    dummy_df['low'] = dummy_df[['open', 'high', 'low', 'close']].min(axis=1)
    dummy_df['high'] = dummy_df[['open', 'high', 'low', 'close']].max(axis=1)
    
    # Dummy TA configuration for testing
    dummy_ta_config = {
        'rsi': {'length': 14},
        'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'emas': {'short_period': 20, 'long_period': 50},
        'atr': {'length': 14}
        # 'bollinger_bands': {'length': 20, 'std_dev': 2} # if testing bbands
    }

    logger.info("Original Dummy DataFrame (tail):")
    logger.info(dummy_df.tail())

    # Add indicators using the dummy config
    df_with_indicators = add_indicators(dummy_df.copy(), dummy_ta_config) 

    if df_with_indicators is not None and not df_with_indicators.empty:
        logger.info("\nDataFrame with Indicators (tail):")
        # Print columns including the newly added ones
        pd.set_option('display.max_columns', None) # Show all columns
        pd.set_option('display.width', 1000) # Wider display
        logger.info(df_with_indicators.tail())
        logger.info("\nColumns added:")
        logger.info([col for col in df_with_indicators.columns if col not in dummy_df.columns])
    else:
        logger.error("\nFailed to add indicators or returned empty DataFrame.")
