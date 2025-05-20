from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file first

import asyncio
import time
import pandas as pd
import pandas_ta as ta # Import pandas_ta for EMA calculation
import os
import sys  # Added for sys.stderr
import io   # Added for io.TextIOWrapper
import ccxt
import yaml  # Added for YAML config loading
import logging # Added for logging
from src.notifications.telegram_notifier import send_telegram_message_async

# --- Setup Logger ---
# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO) # Set level for root logger

# Create handlers
# Wrap sys.stderr with UTF-8 encoding for the console handler
utf8_stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
c_handler = logging.StreamHandler(utf8_stderr)
f_handler = logging.FileHandler('lumesix_bot.log', encoding='utf-8') # Specify UTF-8 for file handler
# Handlers will inherit level from root_logger unless set explicitly, which is fine.

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the root_logger
# Check if handlers already exist to prevent duplication if this script were re-run in some scenarios (e.g. Jupyter)
if not root_logger.hasHandlers() or not any(isinstance(h, logging.FileHandler) and h.baseFilename == f_handler.baseFilename for h in root_logger.handlers):
    if not root_logger.handlers: # Add handlers if no handlers exist
        root_logger.addHandler(c_handler)
        root_logger.addHandler(f_handler)
    else: # If some handlers exist, be more careful - only add if not already present
        # This logic is a bit more complex to avoid duplicate handlers on re-runs in some environments
        # For a typical script run, the initial `if not root_logger.hasHandlers()` would suffice.
        console_handler_exists = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
        file_handler_exists = any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath('lumesix_bot.log') for h in root_logger.handlers)
        if not console_handler_exists:
            root_logger.addHandler(c_handler)
        if not file_handler_exists:
            root_logger.addHandler(f_handler)

# This logger instance will inherit from the root logger's configuration
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

# Load environment variables (for API keys primarily)
# load_dotenv() # Moved to the top

# --- Configuration Loading ---
def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_path}' not found. Exiting.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}. Exiting.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading config: {e}. Exiting.", exc_info=True)
        return None

def check_macro_trend_alignment(bybit_client, symbol: str, macro_timeframe: str, macro_ema_period: int, gpt_decision: str):
    """Checks if the GPT-4 decision aligns with the macro trend determined by an EMA.

    Args:
        bybit_client: The initialized Bybit client.
        symbol (str): Trading symbol (e.g., 'ETH/USDT').
        macro_timeframe (str): Timeframe for macro trend analysis (e.g., '4h', '1d').
        macro_ema_period (int): Period for the EMA on the macro timeframe.
        gpt_decision (str): The trading decision from GPT-4 ('BUY', 'SELL', 'HOLD').

    Returns:
        bool: True if aligned or if GPT decision is 'HOLD', False otherwise.
              Returns True if data fetching or EMA calculation fails, to not block trades by default on error.
    """
    if gpt_decision == 'HOLD':
        return True # No trend alignment needed for HOLD

    logger.info(f"\n--- Checking Macro Trend Alignment ({macro_timeframe} EMA{macro_ema_period}) for {gpt_decision} on {symbol} ---")
    
    # Fetch enough data for EMA calculation + a little buffer
    # fetch_ohlcv is synchronous
    macro_ohlcv_df = fetch_ohlcv(bybit_client, symbol, macro_timeframe, limit=macro_ema_period + 50)

    if macro_ohlcv_df is None or macro_ohlcv_df.empty or len(macro_ohlcv_df) < macro_ema_period:
        logger.warning(f"Could not fetch sufficient data for {macro_timeframe} EMA{macro_ema_period} calculation. Skipping trend check.")
        return True # Default to true to not block trades if data is unavailable

    try:
        # Calculate EMA
        ema_col_name = f'EMA_{macro_ema_period}'
        macro_ohlcv_df.ta.ema(length=macro_ema_period, append=True, col=ema_col_name) # Use 'col' if pandas_ta ver > 0.3.14b0, else use f'EMA_{macro_ema_period}' directly
        
        if ema_col_name not in macro_ohlcv_df.columns:
            logger.warning(f"Failed to calculate {ema_col_name} for macro trend. Skipping trend check.")
            # Try with a common naming convention if 'col' argument behaves differently
            # This is a fallback, ideally the 'col' argument should work with recent pandas_ta versions.
            # For older versions, the column name is automatically generated as 'EMA_LENGTH'.
            # We are explicitly naming it, so this check is mostly for safety.
            if f'EMA_{macro_ema_period}' in macro_ohlcv_df.columns:
                 ema_col_name = f'EMA_{macro_ema_period}'
            else:
                return True # Default to true if EMA calculation fails

        latest_close = macro_ohlcv_df['close'].iloc[-1]
        latest_ema = macro_ohlcv_df[ema_col_name].iloc[-1]

        if pd.isna(latest_ema):
            logger.warning(f"Latest {ema_col_name} is NaN. Skipping trend check.")
            return True

        macro_trend = "UPTREND" if latest_close > latest_ema else "DOWNTREND" if latest_close < latest_ema else "SIDEWAYS"
        logger.info(f"Macro Trend ({macro_timeframe}): Latest Close={latest_close:.2f}, {ema_col_name}={latest_ema:.2f} => {macro_trend}")

        if gpt_decision == 'BUY' and macro_trend == 'UPTREND':
            logger.info(f"Macro trend ({macro_trend}) aligns with GPT BUY decision.")
            return True
        elif gpt_decision == 'SELL' and macro_trend == 'DOWNTREND':
            logger.info(f"Macro trend ({macro_trend}) aligns with GPT SELL decision.")
            return True
        else:
            logger.info(f"Macro trend ({macro_trend}) does NOT align with GPT {gpt_decision} decision.")
            return False
            
    except Exception as e:
        logger.error(f"Error during macro trend check: {e}. Skipping trend check.", exc_info=True)
        return True # Default to true on error to not unduly block trades

# --- Import Core Modules ---
from src.connectors.bybit_connector import (
    get_bybit_client, 
    fetch_ohlcv,
    fetch_positions,       # <-- Import new function
    place_market_order,    # <-- Import new function
    close_position,        # <-- Import new function
    get_available_usdt_balance, # <-- Import new function
    timeframe_to_milliseconds
)
from src.analysis.technical_analyzer import add_indicators
from src.forecasting.timegpt_forecaster import get_timegpt_client, get_timegpt_forecast
from src.evaluation.gpt_evaluator import get_openai_client, evaluate_market_conditions
from src.analysis.confluence_checker import check_timegpt_alignment, check_ta_confirmation # New import

async def main():
    """Main function to run the trading bot logic."""
    logger.info("--- Initializing AI Trading Bot ---")

    # --- Load Configuration ---
    config = load_config()
    if config is None:
        return # Exit if config loading failed

    # --- Extract Config Values ---
    # Trading parameters
    trading_symbol = config['trading']['symbol']
    exec_timeframe = config.get('trading', {}).get('timeframe_execution', '5m')
    # trend_timeframe = config.get('trading', {}).get('timeframe_trend', '1h') # For future use

    # Strategy parameters
    confidence_threshold = config.get('strategy', {}).get('min_confidence_threshold', 80) # Default 80 if not found
    tp_perc_config = config['strategy'].get('target_gain_percent') # Renamed to avoid conflict
    sl_perc_config = config['strategy'].get('stop_loss_percent')   # Renamed to avoid conflict

    # Confluence settings from config
    require_timegpt_alignment = config.get('strategy', {}).get('require_timegpt_trend_alignment', True)
    require_ta_confirm = config.get('strategy', {}).get('require_ta_confirmation', True)

    # Macro Trend Filter settings from config
    require_macro_trend_config = config.get('strategy', {}).get('require_macro_trend_alignment', False)
    macro_timeframe_config = config.get('strategy', {}).get('macro_trend_timeframe', '4h')
    macro_ema_period_config = config.get('strategy', {}).get('macro_ema_period', 200)

    # ATR TP/SL settings from config
    use_atr_tp_sl_config = config.get('strategy', {}).get('use_atr_tp_sl', False)
    atr_tp_mult_config = config.get('strategy', {}).get('atr_tp_multiplier', 2.0)
    atr_sl_mult_config = config.get('strategy', {}).get('atr_sl_multiplier', 1.5)

    # Technical Analysis Indicator Parameters & Confirmation Rules
    technical_analysis_config = config.get('technical_analysis', {})
    ta_confirmation_rules_config = config.get('strategy', {}).get('ta_confirmation_rules', {})
    if not technical_analysis_config or not ta_confirmation_rules_config:
        logger.warning("Warning: 'technical_analysis' or 'ta_confirmation_rules' section missing in config. Using default TA behavior or it might fail.")
        # Potentially set default fallbacks here if crucial, or let it fail if config is essential

    # API/Internal settings
    initial_fetch_limit = config['api']['initial_fetch_limit']
    forecast_horizon = config['api']['forecast_horizon'] # Corrected path
    min_confidence_threshold = config['strategy']['min_confidence_threshold'] # Corrected path
    max_data_length = config['api']['max_data_length']
    loop_sleep_seconds = config['api']['loop_sleep_seconds']

    # --- Translate timeframe for Nixtla frequency parameter ---
    # Nixtla/Pandas expect frequency strings like '5min', '1H', '1D'
    timeframe_map = {'m': 'min', 'h': 'H', 'd': 'D', 'w': 'W'} # Add others if needed
    try:
        value = int(exec_timeframe[:-1])
        unit = exec_timeframe[-1].lower()
        nixtla_freq = f"{value}{timeframe_map[unit]}"
    except (KeyError, ValueError, IndexError):
        logger.error(f"Could not parse execution_timeframe '{exec_timeframe}' into Nixtla frequency. Using original.", exc_info=True)
        nixtla_freq = exec_timeframe # Fallback, might still cause issues

    # --- Initialize Clients ---
    logger.info("Initializing clients...")
    bybit_client = get_bybit_client() # Uses .env for keys, testnet setting
    nixtla_client = get_timegpt_client() # Corrected function call
    openai_client = get_openai_client() # Uses .env for key

    if not bybit_client or not nixtla_client or not openai_client:
        logger.critical("Failed to initialize one or more clients. Exiting.")
        return
        
    # --- Set Bybit Account Configuration ---
    try:
        logger.info(f"Setting margin mode to 'cross' for {trading_symbol}...")
        await bybit_client.set_margin_mode('cross', symbol=trading_symbol, params={'category': 'linear'})
        logger.info(f"Setting leverage to {config['strategy']['leverage']}x for {trading_symbol}...")
        await bybit_client.set_leverage(config['strategy']['leverage'], symbol=trading_symbol, params={'category': 'linear'})
        logger.info("Bybit margin mode and leverage set successfully.")
    except ccxt.ExchangeError as e:
        logger.warning(f"Could not set margin mode or leverage: {e}. Check permissions or if already set.")
    except Exception as e:
        logger.error(f"An unexpected error occurred setting margin/leverage: {e}", exc_info=True)

    logger.info("Clients initialized successfully.")

    # --- Initial Data Fetch and Preparation ---
    logger.info(f"\n--- Fetching initial market data for {trading_symbol} ({exec_timeframe}) ---")
    historical_data = await fetch_ohlcv(bybit_client, trading_symbol, exec_timeframe, limit=initial_fetch_limit)
    if historical_data is None or historical_data.empty:
        logger.error("Failed to fetch initial historical data. Exiting.")
        if bybit_client: 
            if hasattr(bybit_client, 'close') and asyncio.iscoroutinefunction(bybit_client.close):
                logger.info("Closing async Bybit connection...")
                await bybit_client.close()
            elif hasattr(bybit_client, 'close'): # For synchronous clients that might have close
                logger.info("Closing sync Bybit connection...")
                bybit_client.close() 
            else:
                logger.info("Bybit client does not require explicit closing or no close method found.")
        return
    
    logger.info(f"Initial historical data fetched. Shape: {historical_data.shape}")
    historical_data = add_indicators(historical_data, technical_analysis_config) # Pass TA config
    logger.info("Initial indicators calculated.")
    logger.info("Updated Data (tail):")
    logger.info(historical_data.tail().to_string())

    # --- Initial Forecast ---
    logger.info("\n--- Generating initial forecast ---")
    forecast_data = get_timegpt_forecast(
        nixtla_client,
        historical_data,
        horizon=forecast_horizon,
        freq=nixtla_freq # Use translated frequency
    )
    if forecast_data is not None:
        logger.info("Initial forecast generated successfully:")
        logger.info(forecast_data.to_string())
    else:
        logger.warning("Failed to generate initial forecast.")

    # --- Initial Evaluation ---
    logger.info("\n--- Generating initial evaluation ---")
    initial_decision, initial_confidence = evaluate_market_conditions(openai_client, historical_data, forecast_data, trading_symbol)

    if initial_decision and initial_confidence is not None:
        logger.info(f"Initial Evaluation: Decision={initial_decision}, Confidence={initial_confidence}")

        if initial_confidence >= min_confidence_threshold:
            # Perform confluence checks for initial trade
            latest_indicators_initial = historical_data.iloc[-1]
            timegpt_aligned_initial = True # Assume alignment if not required or for first trade if logic dictates
            if require_timegpt_alignment and forecast_data is not None:
                timegpt_aligned_initial = check_timegpt_alignment(forecast_data, initial_decision, latest_indicators_initial['close'])
                logger.info(f"Initial Trade - TimeGPT Alignment Check ({initial_decision}): {timegpt_aligned_initial}")
            
            ta_confirmed_initial = True # Assume TA confirmed if not required
            if require_ta_confirm:
                ta_confirmed_initial = check_ta_confirmation(latest_indicators_initial, initial_decision, ta_confirmation_rules_config, technical_analysis_config)
                logger.info(f"Initial Trade - TA Confirmation Check ({initial_decision}): {ta_confirmed_initial}")

            macro_trend_aligned_initial = True # Assume alignment if not required
            if require_macro_trend_config:
                macro_trend_aligned_initial = check_macro_trend_alignment(bybit_client, trading_symbol, macro_timeframe_config, macro_ema_period_config, initial_decision)
                logger.info(f"Initial Trade - Macro Trend Alignment Check ({initial_decision}): {macro_trend_aligned_initial}")

            if timegpt_aligned_initial and ta_confirmed_initial and macro_trend_aligned_initial:
                # Check balance before placing order
                available_balance = await get_available_usdt_balance(bybit_client)
                min_cost = 5.0 # Example minimum
                required_amount = config['strategy']['order_size_usdt']
                current_price = historical_data['close'].iloc[-1]
                latest_atr = historical_data['ATRr_14'].iloc[-1] if 'ATRr_14' in historical_data.columns else None

                if available_balance >= required_amount and available_balance >= min_cost:
                    logger.info(f"Sufficient balance ({available_balance:.2f} USDT) available.")
                    if initial_decision == 'BUY':
                        logger.info(f"Action: Placing initial MARKET BUY order for {trading_symbol}.")
                        order_result = await place_market_order(bybit_client, trading_symbol, 'buy', required_amount, 
                                               current_price_arg=current_price, 
                                               tp_percent=tp_perc_config, sl_percent=sl_perc_config,
                                               use_atr_tp_sl=use_atr_tp_sl_config, atr_value=latest_atr,
                                               atr_tp_multiplier=atr_tp_mult_config, atr_sl_multiplier=atr_sl_mult_config,
                                               notifier=send_telegram_message_async)
                        if order_result:
                            logger.info(f"Initial BUY order placed: {order_result.get('id')}")
                        else:
                            logger.error("Failed to place initial BUY order.")
                    elif initial_decision == 'SELL':
                        logger.info(f"Action: Placing initial MARKET SELL order for {trading_symbol}.")
                        order_result = await place_market_order(bybit_client, trading_symbol, 'sell', required_amount, 
                                               current_price_arg=current_price, 
                                               tp_percent=tp_perc_config, sl_percent=sl_perc_config,
                                               use_atr_tp_sl=use_atr_tp_sl_config, atr_value=latest_atr,
                                               atr_tp_multiplier=atr_tp_mult_config, atr_sl_multiplier=atr_sl_mult_config,
                                               notifier=send_telegram_message_async)
                        if order_result:
                            logger.info(f"Initial SELL order placed: {order_result.get('id')}")
                        else:
                            logger.error("Failed to place initial SELL order.")
                else:
                    logger.warning(f"Insufficient balance. Available: {available_balance:.2f} USDT, Required Order Size: {required_amount} USDT, Min Exchange Cost: {min_cost} USDT. Skipping trade.")
            else:
                logger.info(f"Initial Trade - Confluence checks failed (TimeGPT: {timegpt_aligned_initial}, TA: {ta_confirmed_initial}, Macro Trend: {macro_trend_aligned_initial}). No trade placed.")
        else:
            logger.info(f"Confidence ({initial_confidence}%) below threshold ({min_confidence_threshold}%). Holding.")
    else:
        logger.error("Initial evaluation failed or returned invalid data.")

    # --- Main Loop ---
    logger.info("\n--- Starting Main Bot Loop --- ")
    timeframe_ms = timeframe_to_milliseconds(exec_timeframe)
    if timeframe_ms is None:
        logger.error(f"Error: Could not convert timeframe '{exec_timeframe}' to milliseconds. Using default 60s loop.")
        timeframe_ms = 60 * 1000

    while True:
        try:
            current_time_utc = pd.Timestamp.utcnow()
            logger.info(f"\n[{current_time_utc}] Loop iteration starting...")

            # --- Fetch Updated Data ---
            logger.info(f"Fetching latest {config['trading']['candles_limit']} {exec_timeframe} candles for {trading_symbol}...")
            new_candle_data = await fetch_ohlcv(bybit_client, trading_symbol, exec_timeframe, limit=config['trading']['candles_limit'])

            if new_candle_data is not None and not new_candle_data.empty:
                logger.info(f"Fetched {len(new_candle_data)} new candle data points.")
                # Append new data and remove oldest if exceeding max_data_length
                historical_data = pd.concat([historical_data, new_candle_data], ignore_index=True)
                historical_data.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                historical_data.sort_values(by='timestamp', inplace=True)
                if len(historical_data) > max_data_length:
                    historical_data = historical_data.iloc[-max_data_length:]
                historical_data.reset_index(drop=True, inplace=True)
                logger.info(f"Historical data updated. Total candles: {len(historical_data)}, Latest candle: {pd.to_datetime(historical_data['timestamp'].iloc[-1], unit='ms')}")

                # Recalculate indicators on the updated historical data
                logger.info("Recalculating technical indicators...")
                historical_data = add_indicators(historical_data, technical_analysis_config) # Pass TA config
                logger.info("Indicators recalculated.")
                logger.info("Updated Data (tail):")
                logger.info(historical_data.tail().to_string())
            else:
                logger.warning("No new market data returned from the latest fetch attempt.")
                # Decide whether to continue or wait longer if no data repeatedly
                # For now, we'll just proceed with the existing data

            # --- Generate Forecast & Evaluation --- (Proceed even if no new data, using last known state)
            if len(historical_data) < config['trading']['min_data_points_for_forecast']:
                logger.warning(f"Not enough historical data ({len(historical_data)}) to generate forecast. Need {config['trading']['min_data_points_for_forecast']}.")
            else:
                # --- Recalculate Indicators & Forecast --- 
                logger.info("Generating new forecast...")
                forecast_data = get_timegpt_forecast(
                    nixtla_client,
                    historical_data,
                    horizon=forecast_horizon,
                    freq=nixtla_freq # Use translated frequency
                )
                if forecast_data is not None:
                    logger.info("New forecast generated successfully:")
                    logger.info(forecast_data.to_string())
                else:
                    logger.warning("Failed to generate forecast in loop.")

                # GPT-4 Market Evaluation
                logger.info("Evaluating market conditions with GPT-4...")
                current_decision, current_confidence = evaluate_market_conditions(openai_client, historical_data, forecast_data, trading_symbol)
                if current_decision and current_confidence is not None:
                    logger.info(f"Evaluation: Decision={current_decision}, Confidence={current_confidence}")

                    if current_confidence >= min_confidence_threshold:
                        # --- Confluence Checks for Loop Trade --- 
                        latest_indicators_loop = historical_data.iloc[-1]
                        timegpt_aligned_loop = True
                        if require_timegpt_alignment and forecast_data is not None:
                            timegpt_aligned_loop = check_timegpt_alignment(forecast_data, current_decision, latest_indicators_loop['close'])
                            logger.info(f"Loop Trade - TimeGPT Alignment Check ({current_decision}): {timegpt_aligned_loop}")

                        ta_confirmed_loop = True
                        if require_ta_confirm:
                            ta_confirmed_loop = check_ta_confirmation(latest_indicators_loop, current_decision, ta_confirmation_rules_config, technical_analysis_config)
                            logger.info(f"Loop Trade - TA Confirmation Check ({current_decision}): {ta_confirmed_loop}")

                        macro_trend_aligned_loop = True
                        if require_macro_trend_config:
                            macro_trend_aligned_loop = check_macro_trend_alignment(bybit_client, trading_symbol, macro_timeframe_config, macro_ema_period_config, current_decision)
                            logger.info(f"Loop Trade - Macro Trend Alignment Check ({current_decision}): {macro_trend_aligned_loop}")

                        # --- Position Management & Trading Logic ---
                        # fetch_positions is synchronous, so no await here
                        open_positions = await fetch_positions(bybit_client, trading_symbol, category="linear")

                        if open_positions is None: # Error case
                            logger.error("Error: Could not fetch position data. Skipping trade action.")
                        elif not open_positions: # No open position
                            logger.info("No open position found.")
                            if timegpt_aligned_loop and ta_confirmed_loop and macro_trend_aligned_loop: # Check confluence for new trade
                                logger.info(f"Loop Trade - Confluence passed for new {current_decision} trade (no existing position).")
                                available_balance = await get_available_usdt_balance(bybit_client)
                                min_cost = 5.0 # Example minimum
                                required_amount = config['strategy']['order_size_usdt']
                                current_price = historical_data['close'].iloc[-1]
                                latest_atr = historical_data['ATRr_14'].iloc[-1] if 'ATRr_14' in historical_data.columns and use_atr_tp_sl_config else None

                                if available_balance >= required_amount and available_balance >= min_cost:
                                    logger.info(f"Sufficient balance ({available_balance:.2f} USDT) available.")
                                    if current_decision == 'BUY':
                                        logger.info(f"Action: Placing MARKET BUY order for {trading_symbol}.")
                                        order_result = await place_market_order(bybit_client, trading_symbol, 'buy', required_amount, 
                                                               current_price_arg=current_price, 
                                                               tp_percent=tp_perc_config, sl_percent=sl_perc_config,
                                                               use_atr_tp_sl=use_atr_tp_sl_config, atr_value=latest_atr,
                                                               atr_tp_multiplier=atr_tp_mult_config, atr_sl_multiplier=atr_sl_mult_config,
                                                               notifier=send_telegram_message_async)
                                        if order_result:
                                            logger.info(f"BUY order placed: {order_result.get('id')}")
                                        else:
                                            logger.error("Failed to place BUY order.")
                                    elif current_decision == 'SELL':
                                        logger.info(f"Action: Placing MARKET SELL order for {trading_symbol}.")
                                        order_result = await place_market_order(bybit_client, trading_symbol, 'sell', required_amount, 
                                                               current_price_arg=current_price, 
                                                               tp_percent=tp_perc_config, sl_percent=sl_perc_config,
                                                               use_atr_tp_sl=use_atr_tp_sl_config, atr_value=latest_atr,
                                                               atr_tp_multiplier=atr_tp_mult_config, atr_sl_multiplier=atr_sl_mult_config,
                                                               notifier=send_telegram_message_async)
                                        if order_result:
                                            logger.info(f"SELL order placed: {order_result.get('id')}")
                                        else:
                                            logger.error("Failed to place SELL order.")
                                else:
                                    logger.warning(f"Insufficient balance. Available: {available_balance:.2f} USDT, Required Order Size: {required_amount} USDT, Min Exchange Cost: {min_cost} USDT. Skipping trade.")
                            else:
                                logger.info(f"Loop Trade - Confluence checks failed for new {current_decision} trade (TimeGPT: {timegpt_aligned_loop}, TA: {ta_confirmed_loop}, Macro Trend: {macro_trend_aligned_loop}). No trade placed.")
                        else: # Position exists
                            current_position = open_positions[0] # Assuming only one position per symbol
                            current_position_side = current_position.get('side') # 'long' or 'short'
                            contracts = current_position.get('contracts')
                            logger.info(f"Existing {current_position_side} position found ({contracts} contracts).")

                            if current_position_side == 'long' and current_decision == 'SELL':
                                logger.info(f"Action: Closing existing LONG position based on SELL signal.")
                                close_order_result = await close_position(bybit_client, current_position)
                                if close_order_result:
                                    logger.info(f"Position closed: {close_order_result.get('id')}")
                                else:
                                    logger.error(f"Failed to close position for {trading_symbol}.")
                            elif current_position_side == 'short' and current_decision == 'BUY':
                                logger.info(f"Action: Closing existing SHORT position based on BUY signal.")
                                close_order_result = await close_position(bybit_client, current_position)
                                if close_order_result:
                                    logger.info(f"Position closed: {close_order_result.get('id')}")
                                else:
                                    logger.error(f"Failed to close position for {trading_symbol}.")
                            else:
                                logger.info(f"Action: Holding existing {current_position_side} position (Signal: {current_decision}).")
                    else:
                        logger.info(f"Confidence ({current_confidence}%) below threshold ({min_confidence_threshold}%). Holding.")
                else:
                    logger.error("Evaluation failed or returned invalid data (current_decision or current_confidence is None). No trade action taken.")

            # --- Sleep ---
            logger.info(f"Loop finished. Sleeping for {loop_sleep_seconds} seconds...")
            await asyncio.sleep(loop_sleep_seconds)

        except KeyboardInterrupt:
            logger.info("\nKeyboardInterrupt received. Shutting down...")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the main loop: {e}")
            logger.info("--- Full Traceback ---")
            logger.error(traceback.format_exc()) # Print the full traceback
            logger.info("--- End Traceback ---")
            logger.info("Continuing loop after a short delay...")
            await asyncio.sleep(10) # Wait before retrying

    # --- Cleanup ---
    logger.info("\nClosing connections...")
    if bybit_client:
        if hasattr(bybit_client, 'close') and asyncio.iscoroutinefunction(bybit_client.close):
            logger.info("Closing async Bybit connection...")
            await bybit_client.close()
        elif hasattr(bybit_client, 'close'): # For synchronous clients that might have close
            logger.info("Closing sync Bybit connection...")
            bybit_client.close() 
        else:
            logger.info("Bybit client does not require explicit closing or no close method found.")
    logger.info("Bot shutdown complete.")

if __name__ == '__main__':
    # Ensure config.yaml exists before running
    if not os.path.exists('config.yaml'):
        logger.error("Error: config.yaml not found in the project root.")
        logger.info("Please ensure the configuration file exists before running main.py.")
    else:
        try:
            asyncio.run(main())
        except Exception as e:
            logger.error(f"Unhandled exception in main execution: {e}")
