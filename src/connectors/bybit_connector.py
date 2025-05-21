import ccxt
import ccxt.pro as ccxtpro # Import ccxt.pro for async/websocket
import pandas as pd
import time
import asyncio 
import traceback # For detailed error logging if needed
import math # For TP/SL price formatting print
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Ensure that your config.py is in the lumesix directory, or adjust path accordingly
# If config.py is in lumesix, and src is a subfolder, this might need adjustment
# For now, assuming config.py can be imported directly or PYTHONPATH is set.
# A better way for larger projects is to ensure 'lumesix' is on PYTHONPATH or use relative imports if structured.
# Let's try to import from the parent directory. Add 'lumesix' to sys.path if needed.
import sys
import os

# Add the parent directory (lumesix) to sys.path to find the config module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # This should be 'lumesix'
sys.path.insert(0, project_root)

from config import BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET, DEFAULT_TRADING_PAIR, DEFAULT_TIMEFRAME
# Import the TA function
from src.analysis import add_indicators

def get_bybit_client(testnet=BYBIT_TESTNET):
    """
    Initializes and returns a CCXT Bybit exchange client.

    Args:
        testnet (bool): If True, initializes the client for Bybit's testnet.
                        Defaults to the value set in config.py (BYBIT_TESTNET).

    Returns:
        ccxt.bybit: An instance of the CCXT Bybit exchange class.
                    Returns None if API keys are not configured.
    """
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        logger.error("Error: Bybit API Key or Secret not found or not configured. Check environment variables and config.")
        logger.info("Please ensure BYBIT_API_KEY and BYBIT_API_SECRET are available to the application.")
        return None

    exchange_config = {
        'apiKey': BYBIT_API_KEY,
        'secret': BYBIT_API_SECRET,
        'enableRateLimit': True, # Add rate limiting
        'options': {
            'defaultType': 'linear', # Use 'linear' for USDT perpetuals
            'adjustForTimeDifference': True, # Recommended by ccxt
            'v5': True              # Explicitly request API v5
        }
    }

    exchange = ccxtpro.bybit(exchange_config) # Changed to use ccxtpro for async client

    exchange.timeout = 30000 # Set timeout to 30 seconds (30000 ms)

    if testnet:
        logger.info("Initializing Bybit client for TESTNET.")
        exchange.set_sandbox_mode(True)
    else:
        logger.info("Initializing Bybit client for MAINNET.")

    return exchange

def set_trading_conditions(client, symbol, leverage, margin_mode='cross'):
    """Sets margin mode and leverage for a given symbol."""
    try:
        client.load_markets(True) # Ensure markets are loaded
        market = client.market(symbol)

        if not market or market['type'] != 'swap' or not market.get('linear'):
            logger.error(f"Cannot set leverage/margin for non-linear swap symbol: {symbol}. Market type: {market.get('type')}, Linear: {market.get('linear')}")
            return False

        # Set Margin Mode
        # For Bybit unified accounts (and many others), margin mode is typically set per symbol or per category (e.g. linear)
        # ccxt unified method: client.set_margin_mode(margin_mode, symbol, params)
        # Bybit specific: may need params={'category': 'linear'} or similar depending on account type & ccxt version
        # For 'cross' margin on linear contracts, it's often the account default or set for the category.
        # For 'isolated', it's set with leverage.
        
        # Attempt to set cross margin for linear category if applicable
        # Note: For Bybit Unified Trading Account (UTA), 'cross' is the default for the entire account.
        # Setting it explicitly per symbol might not be available or might apply to isolated margin only.
        # We'll try to set isolated margin with leverage if margin_mode is 'isolated',
        # otherwise assume 'cross' is default or managed at account level.

        current_leverage = None
        if margin_mode.lower() == 'isolated':
            logger.info(f"Attempting to set ISOLATED margin mode for {symbol} with leverage {leverage}x.")
            # For isolated margin, leverage is set simultaneously.
            client.set_leverage(leverage, symbol, {'marginMode': 'isolated'})
            logger.info(f"Successfully set ISOLATED margin and leverage to {leverage}x for {symbol}.")
            current_leverage = leverage
        elif margin_mode.lower() == 'cross':
            logger.info(f"Attempting to set leverage to {leverage}x for {symbol} (assuming CROSS margin or account default).")
            # For cross margin, just set leverage. The marginMode param might not be needed or supported here by all exchanges.
            # Bybit setLeverage for linear contracts typically handles this for the contract under cross margin.
            client.set_leverage(leverage, symbol) # No specific marginMode param for cross, it applies to the symbol under current acc margin mode
            logger.info(f"Successfully set leverage to {leverage}x for {symbol} (CROSS margin). Call may fail if symbol already in isolated.")
            current_leverage = leverage
        else:
            logger.warning(f"Unsupported margin mode: {margin_mode}. Skipping margin/leverage settings.")
            return False

        # Verify leverage (optional, as set_leverage would error out if failed)
        # positions = client.fetch_positions([symbol]) # May not be available pre-trade
        # if positions:
        #     pos_leverage = positions[0].get('leverage')
        #     logger.info(f"Verified leverage for {symbol}: {pos_leverage}x")
        # else: # Fallback if no position, try to get from account/market config if API supports
        #     logger.info(f"Leverage set to {leverage}x for {symbol}. Verification may require fetching positions or specific API endpoint.")

        return True

    except ccxt.NetworkError as e:
        logger.error(f"Network error setting trading conditions for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error setting trading conditions for {symbol}: {e} - Check API permissions, symbol settings on exchange, or if leverage is already set.")
    except Exception as e:
        logger.error(f"Unexpected error setting trading conditions for {symbol}: {e}")
    return False

async def fetch_ohlcv(client, symbol=DEFAULT_TRADING_PAIR, timeframe=DEFAULT_TIMEFRAME, since=None, limit=100):
    """
    Fetches historical OHLCV data for a given symbol and timeframe.

    Args:
        client (ccxt.bybit): The initialized Bybit client.
        symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        timeframe (str): The timeframe string (e.g., '1m', '5m', '1h').
        since (int): Timestamp in milliseconds for the earliest candle to fetch.
        limit (int): The maximum number of candles to fetch (Bybit limit might apply, often 1000 or 200).

    Returns:
        pandas.DataFrame: A DataFrame containing OHLCV data with columns 
                          ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                          sorted by timestamp descending. Returns None if an error occurs.
                          Returns an empty DataFrame if no data is available for the period.
    """
    if not client:
        logger.error("Error: Bybit client is not initialized.")
        return None

    try:
        logger.info(f"Fetching {limit} {timeframe} candles for {symbol}...")
        # CCXT fetch_ohlcv returns a list of lists:
        # [[timestamp, open, high, low, close, volume], ...]
        ohlcv_data = await client.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)

        if not ohlcv_data:
            logger.info(f"No OHLCV data found for {symbol} with the given parameters.")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # --- Fix: Check for and correct negative timestamps ---
        if (df['timestamp'] < 0).any():
            logger.warning("Negative timestamps detected in fetched data. Applying abs().")
            df['timestamp'] = df['timestamp'].abs()
        # --- End Fix ---

        # Ensure correct data types
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        # Convert timestamp to integer AFTER abs() if needed, or keep as is if unit='ms' conversion handles it
        df['timestamp'] = df['timestamp'].astype('int64') # Ensure it's integer for potential downstream use

        logger.info(f"Successfully fetched {len(df)} candles.")
        return df

    except ccxt.NetworkError as e:
        logger.error(f"Network error fetching OHLCV for {symbol}: {e}")
        return None
    except ccxt.ExchangeError as e:
        # Handle specific Bybit rate limit errors if needed
        if 'ratelimit' in str(e).lower() or 'too many visits' in str(e).lower():
            logger.info(f"Rate limit hit fetching OHLCV for {symbol}. Consider adding delays. Error: {e}")
        else:
             logger.error(f"Exchange error fetching OHLCV for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching OHLCV for {symbol}: {e}")
        return None


async def stream_realtime_data(client, symbol=DEFAULT_TRADING_PAIR, data_type='ticker'):
    """
    Connects to Bybit WebSocket and streams real-time data (ticker or kline).
    Uses ccxt.pro client.

    Args:
        client (ccxt.pro.bybit): The initialized ccxt.pro Bybit client.
        symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        data_type (str): Type of data to stream ('ticker' or 'kline').

    Note: This function runs indefinitely until interrupted (e.g., Ctrl+C).
    """
    if not client:
        logger.error("WS Error: Bybit client (ccxt.pro) is not initialized.")
        return

    # Check for specific async methods from ccxt.pro
    if not hasattr(client, 'watch_ticker') or not hasattr(client, 'watch_ohlcv'):
        logger.error("WS Error: The provided client does not seem to support WebSocket methods (watch_ticker/watch_ohlcv).")
        return

    logger.info(f"Starting WebSocket stream for {symbol} {data_type}...")
    logger.info("(Press Ctrl+C to stop)")

    while True:
        try:
            if data_type == 'ticker':
                # logger.info("WS: Awaiting ticker data...") # Optional: Verbose logging
                ticker_data = await client.watch_ticker(symbol)
                # logger.info("WS: Received ticker data:", ticker_data) # Optional: Log full data
                timestamp = ticker_data.get('iso8601', 'N/A')
                last_price = ticker_data.get('last', 'N/A')
                bid = ticker_data.get('bid', 'N/A')
                ask = ticker_data.get('ask', 'N/A')
                logger.info(f"{timestamp} - {symbol}: Last={last_price}, Bid={bid}, Ask={ask}")
            
            # Add 'kline'/'ohlcv' handling here if needed (using watch_ohlcv)

            else:
                logger.error(f"WS Error: Unsupported data_type '{data_type}' for streaming.")
                break # Exit loop if type is wrong

        except asyncio.CancelledError:
            logger.info("WS: Stream cancelled by task cancellation.")
            break # Exit loop cleanly on cancellation
        except ccxt.RequestTimeout as e:
            logger.error(f"WS RequestTimeout: {e}. Attempting to continue...")
            # ccxt.pro often handles reconnection automatically
            await asyncio.sleep(1) # Brief pause before next loop iteration
        except ccxt.NetworkError as e:
            # Includes ConnectionClosed, etc.
            logger.error(f"WS NetworkError: {e}. Reconnecting attempt might happen automatically...")
            await asyncio.sleep(5) # Wait before potentially retrying
        except ccxt.ExchangeError as e:
            logger.error(f"WS ExchangeError: {e}. Stopping stream.")
            break # Exit loop on specific exchange errors
        except Exception as e:
            # Catch any other unexpected error
            logger.error(f"WS: An unexpected error occurred in stream loop: {type(e).__name__}: {e}")
            # Optionally re-raise if needed: raise
            break # Exit loop on other errors
        
    logger.info("WS: Exited streaming loop.")
    # Note: Client closure should happen in the calling function's finally block


# Updated function to support both percentage-based and ATR-based TP/SL [v2.0]  
async def place_market_order(client, symbol, side, cost_usdt, current_price_arg,
                       tp_percent=None, sl_percent=None, 
                       use_atr_tp_sl=False, atr_value=None, 
                       atr_tp_multiplier=None, atr_sl_multiplier=None, notifier=None):
    """Places a market order with optional Take Profit (TP) and Stop Loss (SL).
    
    TP/SL can be percentage-based or ATR-based.
    Args:
        client: The Bybit API client.
        symbol (str): The trading symbol (e.g., 'BTC/USDT').
        side (str): 'buy' or 'sell'.
        cost_usdt (float): The desired cost of the order in USDT (approximate for market orders).
        current_price_arg (float): The current market price, used for quantity calculation and fallback for TP/SL.
        tp_percent (float, optional): Take profit percentage from the entry price.
        sl_percent (float, optional): Stop loss percentage from the entry price.
        use_atr_tp_sl (bool, optional): If True, use ATR for TP/SL calculations.
        atr_value (float, optional): The current ATR value. Required if use_atr_tp_sl is True.
        atr_tp_multiplier (float, optional): Multiplier for ATR to set TP. Required if use_atr_tp_sl is True.
        atr_sl_multiplier (float, optional): Multiplier for ATR to set SL. Required if use_atr_tp_sl is True.
        notifier (function, optional): A function to send a notification on successful order placement.
    Returns:
        dict: The order response from the exchange, or None if an error occurred.
    """
    if current_price_arg is None or current_price_arg <= 0:
        logger.error("Error: current_price_arg must be a positive value for order placement.")
        return None

    try:
        logger.info(f"Attempting to place {side} market order for {symbol} with cost ~{cost_usdt} USDT...")
        
        live_ticker_price = await fetch_current_price(client, symbol)
        price_for_calculation = live_ticker_price if live_ticker_price else current_price_arg

        if price_for_calculation <= 0:
            logger.error(f"Error: Price for calculation ({price_for_calculation}) is not valid.")
            return None
        
        logger.info(f"Using live ticker price for order calculations: {price_for_calculation}")

        raw_quantity = cost_usdt / price_for_calculation
        logger.info(f"Calculated raw quantity ({cost_usdt} USDT / {price_for_calculation} live ticker price): {raw_quantity}")

        markets = await client.load_markets()
        market = markets.get(symbol)
        if not market:
            logger.error(f"Market details for {symbol} not found.")
            return None

        logger.info(f"Market Raw Data for {symbol}: {market}") # Log raw market data
        logger.info(f"Market Limits for {symbol}: Amount Min={market.get('limits', {}).get('amount', {}).get('min')}, Amount Max={market.get('limits', {}).get('amount', {}).get('max')}")
        logger.info(f"Market Precision for {symbol}: Amount Precision={market.get('precision', {}).get('amount')}, Price Precision={market.get('precision', {}).get('price')}")

        amount_precision_val = market.get('precision', {}).get('amount')
        price_precision_val = market.get('precision', {}).get('price')

        # Fallback for precision if not directly available
        if amount_precision_val is None:
            min_amount_step = market.get('limits', {}).get('amount', {}).get('min', 0.001) 
            amount_precision_val = abs(int(round(math.log10(min_amount_step)))) if min_amount_step > 0 else 3
            logger.warning(f"Amount precision not found, inferred to {amount_precision_val} based on min step.")
        if price_precision_val is None:
            min_price_step = market.get('limits', {}).get('price', {}).get('min', 0.01)
            price_precision_val = abs(int(round(math.log10(min_price_step)))) if min_price_step > 0 else 2
            logger.warning(f"Price precision not found, inferred to {price_precision_val} based on min step.")

        quantity = float(client.amount_to_precision(symbol, raw_quantity))
        logger.info(f"Formatted quantity (amount): {quantity} (Using Precision/Step: {market.get('precision', {}).get('amount')})")

        if quantity <= 0:
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            logger.error(f"Error: Calculated quantity ({quantity}) is zero or less. Min amount for {symbol} is {min_amount}. Cost ({cost_usdt} USDT) might be too low for current price ({price_for_calculation}).")
            return None

        params = {'category': 'linear'}
        tp_price_formatted = None
        sl_price_formatted = None

        entry_price_for_tp_sl = price_for_calculation

        if use_atr_tp_sl and atr_value is not None and atr_tp_multiplier is not None and atr_sl_multiplier is not None and atr_value > 0:
            logger.info(f"Calculating TP/SL using ATR (Value: {atr_value}, TP Mult: {atr_tp_multiplier}, SL Mult: {atr_sl_multiplier})")
            if side == 'buy':
                tp_price = entry_price_for_tp_sl + (atr_value * atr_tp_multiplier)
                sl_price = entry_price_for_tp_sl - (atr_value * atr_sl_multiplier)
            else:  # sell
                tp_price = entry_price_for_tp_sl - (atr_value * atr_tp_multiplier)
                sl_price = entry_price_for_tp_sl + (atr_value * atr_sl_multiplier)
            
            # Determine num_decimals for logging based on price_tick_size
            price_tick_size = market.get('precision', {}).get('price')
            num_decimals_for_log = 2 # Default
            if price_tick_size is not None and price_tick_size > 0:
                if price_tick_size < 1: # e.g. 0.01 -> 2 decimals
                    num_decimals_for_log = abs(int(round(math.log10(price_tick_size))))
                else: # e.g. 1.0 -> 0 decimals
                    num_decimals_for_log = 0
            else: # Fallback if market['precision']['price'] is missing
                min_price_tick_from_limits = market.get('limits', {}).get('price', {}).get('min')
                if min_price_tick_from_limits is not None and min_price_tick_from_limits > 0:
                    if min_price_tick_from_limits < 1:
                        num_decimals_for_log = abs(int(round(math.log10(min_price_tick_from_limits))))
                    else:
                        num_decimals_for_log = 0
                    logger.warning(f"Used min_price_tick {min_price_tick_from_limits} for {num_decimals_for_log} log decimals based on limits.")
                else:
                    logger.warning(f"Defaulting to {num_decimals_for_log} log decimals due to missing precision info.")

            tp_price_formatted = client.price_to_precision(symbol, tp_price)
            sl_price_formatted = client.price_to_precision(symbol, sl_price)
            logger.info(f"ATR-based TP Price (raw): {tp_price:.{num_decimals_for_log}f} -> Formatted by ccxt: {tp_price_formatted}")
            logger.info(f"ATR-based SL Price (raw): {sl_price:.{num_decimals_for_log}f} -> Formatted by ccxt: {sl_price_formatted}")
            
            params['takeProfit'] = tp_price_formatted
            params['tpTriggerBy'] = 'MarkPrice'  # Options: MarkPrice, IndexPrice, LastPrice
        elif tp_percent is not None and sl_percent is not None:
            logger.info(f"Calculating TP/SL using percentages (TP: {tp_percent}%, SL: {sl_percent}%)")
            if side == 'buy':
                tp_price = entry_price_for_tp_sl * (1 + tp_percent / 100)
                sl_price = entry_price_for_tp_sl * (1 - sl_percent / 100)
            else:  # sell
                tp_price = entry_price_for_tp_sl * (1 - tp_percent / 100)
                sl_price = entry_price_for_tp_sl * (1 + sl_percent / 100)

            tp_price_formatted = client.price_to_precision(symbol, tp_price)
            sl_price_formatted = client.price_to_precision(symbol, sl_price)
            logger.info(f"Percentage-based TP Price: {tp_price:.{price_precision_val if price_precision_val else 2}f} -> Formatted: {tp_price_formatted}")
            logger.info(f"Percentage-based SL Price: {sl_price:.{price_precision_val if price_precision_val else 2}f} -> Formatted: {sl_price_formatted}")
        
        if tp_price_formatted and float(tp_price_formatted) > 0:
            params['takeProfit'] = tp_price_formatted
            params['tpTriggerBy'] = 'MarkPrice'  # Options: MarkPrice, IndexPrice, LastPrice
        if sl_price_formatted and float(sl_price_formatted) > 0:
            # Ensure SL is not through the current price for a buy order, or vice versa
            valid_sl = True
            if side == 'buy' and float(sl_price_formatted) >= entry_price_for_tp_sl:
                logger.warning(f"Stop loss ({sl_price_formatted}) for BUY order is at or above entry price ({entry_price_for_tp_sl}). SL will not be set.")
                valid_sl = False
            elif side == 'sell' and float(sl_price_formatted) <= entry_price_for_tp_sl:
                logger.warning(f"Stop loss ({sl_price_formatted}) for SELL order is at or below entry price ({entry_price_for_tp_sl}). SL will not be set.")
                valid_sl = False
            
            if valid_sl:
                params['stopLoss'] = sl_price_formatted
                params['slTriggerBy'] = 'MarkPrice'
            else:
                # Remove sl_price_formatted if it's invalid to prevent sending it
                sl_price_formatted = None 

        # Final check to prevent orders with TP=SL or other invalid TP/SL relative to entry
        if tp_price_formatted and sl_price_formatted and tp_price_formatted == sl_price_formatted:
            logger.warning(f"TP price ({tp_price_formatted}) and SL price ({sl_price_formatted}) are identical. Removing TP and SL from order.")
            if 'takeProfit' in params: del params['takeProfit']
            if 'stopLoss' in params: del params['stopLoss']
        elif side == 'buy' and tp_price_formatted and float(tp_price_formatted) <= entry_price_for_tp_sl:
            logger.warning(f"TP price ({tp_price_formatted}) for BUY order is at or below entry price ({entry_price_for_tp_sl}). Removing TP.")
            if 'takeProfit' in params: del params['takeProfit']
        elif side == 'sell' and tp_price_formatted and float(tp_price_formatted) >= entry_price_for_tp_sl:
            logger.warning(f"TP price ({tp_price_formatted}) for SELL order is at or above entry price ({entry_price_for_tp_sl}). Removing TP.")
            if 'takeProfit' in params: del params['takeProfit']

        logger.info(f"Submitting MARKET {side.upper()} order: symbol={symbol}, quantity={quantity}, params={params}")
        order = await client.create_order(symbol, 'market', side, quantity, params=params)
        logger.info(f"Order placed successfully for {symbol}: {order}")
        
        if notifier and order:
            try:
                tp_val = params.get('takeProfit', 'N/A')
                sl_val = params.get('stopLoss', 'N/A')
                # Use actual fill price for notification if available, else the price used for calculation
                notif_entry_price = order.get('price') if order.get('price') else price_for_calculation
                # logger.info(f"Order {order.get('id')} for {symbol} submitted to exchange.")
                message_parts = [
                    f"✅ **Trade Alert on Bybit** ✅",
                    f"**Symbol:** {symbol}",
                    f"**Side:** {side.upper()}",
                    f"**Quantity:** {client.amount_to_precision(symbol, quantity)}",
                    f"**Approx. Cost:** {cost_usdt:.2f} USDT",
                    f"**Entry Price (approx):** {client.price_to_precision(symbol, notif_entry_price)}",
                    f"**Order ID:** {order.get('id')}"
                ]
                if tp_val != 'N/A':
                    message_parts.append(f"**TP:** {tp_val}")
                if sl_val != 'N/A':
                    message_parts.append(f"**SL:** {sl_val}")
                message = "\n".join(message_parts)
                await notifier(message)
                logger.info(f"Telegram notification attempt for order {order.get('id')} completed.")
            except Exception as notify_e:
                logger.error(f"Failed to send Telegram notification for order {order.get('id')}: {notify_e}")
        
        return order

    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds to place order for {symbol}: {e}")
        return None
    except ccxt.NetworkError as e:
        logger.error(f"Network error while placing order for {symbol}: {e}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error while placing order for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in place_market_order for {symbol}: {e}", exc_info=True)
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_current_price(client, symbol):
    """Fetches the current mark price for a given symbol."""
    try:
        ticker = await client.fetch_ticker(symbol)
        # Bybit often uses 'mark' for linear perpetuals in the info structure or specific methods
        current_price = ticker.get('mark', ticker.get('last')) 
        if current_price is None:
            logger.warning(f"Could not determine mark or last price from ticker for {symbol}. Ticker: {ticker}")
            return None
        return float(current_price)
    except ccxt.NetworkError as e:
        logger.error(f"Network error fetching price for {symbol}: {e}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error fetching price for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching price for {symbol}: {e}", exc_info=True)
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_positions(client, symbol=None, category='linear'): # Added category argument with default
    """Fetches open positions, optionally filtered by symbol and category."""
    try:
        await client.load_markets(True) # Force reload markets for fresh data
        # Market details logging removed for brevity, can be added back if needed for debugging
        
        params = {'category': category}
        if hasattr(client, 'fetch_positions_v5'):
            positions = await client.fetch_positions_v5(symbol=symbol, params=params)
        elif hasattr(client, 'fetch_positions'): 
            positions = await client.fetch_positions(symbols=[symbol] if symbol else None) # Simpler fallback
        else:
            logger.error("Client does not support fetch_positions_v5 or fetch_positions.")
            return None
        
        open_positions = [p for p in positions if p.get('contracts') is not None and float(p.get('contracts', 0)) != 0]
        return open_positions
    except ccxt.NetworkError as e:
        logger.error(f"Network error fetching positions: {e}")
        return None
    except ccxt.ExchangeError as e:
        # Specific handling for 'position not found' or similar benign errors
        if 'position not found' in str(e).lower() or 'OrderNotExists' in str(e) or '30016' in str(e): # 30016 for Bybit: position not found
            logger.info(f"No open positions found for {symbol} (category: {category}).")
            return [] # Return empty list for no positions
        logger.error(f"Exchange error fetching positions for {symbol} (category: {category}): {e}")
        return None # For other exchange errors
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching positions for {symbol} (category: {category}): {e}", exc_info=True)
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def close_position(client, position_info):
    """Closes an open position based on its information."""
    try:
        symbol = position_info.get('symbol')
        side = position_info.get('side') # 'long' or 'short'
        contracts = position_info.get('contracts')
        
        if not all([symbol, side, contracts]):
            logger.error(f"Error: Incomplete position_info for closing: {position_info}")
            return None

        close_side = 'sell' if side == 'long' else 'buy'
        order_amount = float(contracts)
        # Ensure 'category' is passed if required by your Bybit account type (e.g., 'linear')
        order = await client.create_order(symbol, 'market', close_side, order_amount, params={'category': 'linear', 'reduceOnly': True})
        logger.info(f"Position for {symbol} closed successfully: {order.get('id') if order else 'N/A'}")
        return order
    except ccxt.InsufficientFunds as e: 
        logger.error(f"Insufficient funds error when trying to close position for {symbol} (should be reduceOnly): {e}")
        return None
    except ccxt.NetworkError as e:
        logger.error(f"Network error while closing position for {symbol}: {e}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error while closing position for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while closing position for {symbol}: {e}", exc_info=True)
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def get_available_usdt_balance(client):
    """Fetches the available USDT balance from a unified trading account."""
    try:
        balance = await client.fetch_balance(params={'accountType': 'UNIFIED'}) # or 'CONTRACT' for some older setups
        # For UNIFIED, USDT is usually under total, free, or used with coin 'USDT'
        # The structure can be complex, let's try to find USDT specifically
        usdt_balance = 0
        if 'info' in balance and 'list' in balance['info']:
            for asset_balance in balance['info']['list']:
                if asset_balance.get('coin') == 'USDT':
                    # 'walletBalance' or 'availableToWithdraw' or 'equity' could be relevant
                    # 'availableToWithdraw' is often a good proxy for what's truly free for new trades
                    # Let's use 'availableBalance' if present for the specific coin, else a more general one.
                    # For Bybit v5 API, it's often 'availableBalance' in the coin's details or 'totalAvailableBalance'
                    # Looking at `balance['USDT']['free']` is a common ccxt structure after parsing
                    if 'USDT' in balance and 'free' in balance['USDT'] and balance['USDT']['free'] is not None:
                        usdt_balance = float(balance['USDT']['free'])
                        break
                    # Fallback to info list if direct USDT free balance not found
                    usdt_balance = float(asset_balance.get('availableToWithdraw', asset_balance.get('walletBalance', 0)))
                    break # Found USDT
        elif 'USDT' in balance and 'free' in balance['USDT'] and balance['USDT']['free'] is not None: # More direct access
             usdt_balance = float(balance['USDT']['free'])
        else:
            logger.warning("Could not find detailed USDT balance in 'list' or direct 'USDT' free entry. Checking total 'free'.")
            usdt_balance = float(balance.get('free', {}).get('USDT', 0))

        logger.info(f"Available USDT balance: {usdt_balance}")
        return usdt_balance
    except ccxt.NetworkError as e:
        logger.error(f"Network error fetching balance: {e}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error fetching balance: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching balance: {e}", exc_info=True)
        return None

async def fetch_historical_data_async(client, symbol, timeframe, limit):
    """ Asynchronously fetches historical OHLCV data. """
    logger.info(f"Fetching {limit} {timeframe} candles for {symbol}...")
    try:
        # Make sure the client is configured for async operations if necessary
        # Some ccxt methods might not be async by default or require specific setup
        # For Bybit, fetch_ohlcv is typically blocking but works in an async context
        # if the client itself was instantiated with async capabilities or is managed by an event loop.
        loop = asyncio.get_event_loop()
        # Use run_in_executor to run the blocking ccxt call in a separate thread
        ohlcv = await loop.run_in_executor(None, lambda: client.fetch_ohlcv(symbol, timeframe, limit=limit))
        # logger.debug(f"Raw OHLCV data sample (first entry): {ohlcv[0] if ohlcv else 'N/A'}")
        logger.info(f"Successfully fetched {len(ohlcv) if ohlcv else 0} candles.")
        return ohlcv
    except ccxt.NetworkError as e:
        logger.error(f"Network error fetching historical data for {symbol}: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error fetching historical data for {symbol}: {e}")
    except Exception as e:
        logger.error(f"An error occurred during async data fetch for {symbol}: {e}", exc_info=True)
    return [] # Return empty list on failure

async def run_connector_tests():
    # --- Section 1: Sync Tests (Initialization, OHLCV Fetch, TA) ---
    # Note: "Sync Tests" is now a misnomer as we're using an async client (ccxtpro)
    # and async functions. Renaming to "Module Tests" or similar might be clearer.
    logger.info("--- Running Connector Module Tests ---")
    
    # get_bybit_client() returns an async ccxtpro client
    # so variable name 'sync_client' is kept for minimal diff but it's an async client.
    async_client = get_bybit_client() 

    if async_client:
        logger.info("Bybit async client initialized successfully for tests.")
        try:
            # Test 1: Fetch Server Time
            server_time = await async_client.fetch_time() # Client is async, so await fetch_time
            logger.info(f"Bybit Server Time: {async_client.iso8601(server_time)}")

            # Test 2: Fetch OHLCV Data
            logger.info("\nTesting fetch_ohlcv function...")
            symbol_to_fetch = DEFAULT_TRADING_PAIR
            timeframe_to_fetch = DEFAULT_TIMEFRAME
            num_candles = 50
            ohlcv_df = await fetch_ohlcv(async_client, symbol=symbol_to_fetch, timeframe=timeframe_to_fetch, limit=num_candles)

            if ohlcv_df is not None and not ohlcv_df.empty:
                logger.info(f"\nSuccessfully fetched {len(ohlcv_df)} OHLCV data points for {symbol_to_fetch} ({timeframe_to_fetch}).")
                ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
                logger.info(f"Latest data point:\n{ohlcv_df.tail(1)}")

                # Test 3: Add Technical Indicators (using the local add_indicators)
                logger.info("\nTesting add_indicators function...")
                indicators_df = add_indicators(ohlcv_df.copy()) # Pass a copy to avoid modifying original df
                if indicators_df is not None and not indicators_df.empty:
                    logger.info("Successfully added technical indicators.")
                    logger.info(f"Latest data with indicators:\n{indicators_df.tail(1)}")
                else:
                    logger.error("Failed to add technical indicators or DataFrame was empty.")
            else:
                logger.error("Failed to fetch OHLCV data for indicator testing or DataFrame was empty.")

            # --- Test 4: Fetch Available Balance ---
            logger.info("\nTesting get_available_usdt_balance...")
            balance = await get_available_usdt_balance(async_client)
            if balance is not None:
                logger.info(f"Available USDT balance: {balance}")
            else:
                logger.error("Failed to fetch available USDT balance.")

            # --- Test 5: Fetch Positions (Optional - may require open positions) ---
            logger.info("\nTesting fetch_positions...")
            # Test with a common symbol, e.g., BTC/USDT:USDT
            positions = await fetch_positions(async_client, symbol=DEFAULT_TRADING_PAIR, category='linear')
            if positions is not None:
                if positions: # If list is not empty
                    logger.info(f"Fetched {len(positions)} open position(s) for {DEFAULT_TRADING_PAIR}:")
                    for pos in positions:
                        logger.info(f"  - {pos}")
                else:
                    logger.info(f"No open positions found for {DEFAULT_TRADING_PAIR}.")
            else:
                logger.error(f"Failed to fetch positions for {DEFAULT_TRADING_PAIR}.")

            # --- Placeholder for Place Order and Close Position Tests (CAUTION: USES REAL FUNDS IF NOT TESTNET) ---
            # These require careful handling, especially on mainnet.
            # Ensure BYBIT_TESTNET=True in .env before uncommenting/running for real orders.
            if BYBIT_TESTNET:
                logger.info("\n--- Testnet Order Placement and Closing Tests (SKIPPED FOR NOW - Requires manual setup/verification) ---")
                # current_price_for_test = await fetch_current_price_async(async_client, DEFAULT_TRADING_PAIR)
                # if current_price_for_test:
                #     logger.info(f"Current price for {DEFAULT_TRADING_PAIR} for order test: {current_price_for_test}")
                #     # Test place_market_order
                #     # test_order = await place_market_order(async_client, DEFAULT_TRADING_PAIR, 'buy', 10, current_price_for_test, notifier=lambda msg: logger.info(f"TestNotif: {msg}"))
                #     # if test_order:
                #     #     logger.info(f"Test order placed: {test_order.get('id')}")
                #     #     # Test close_position (would need to fetch the position info first)
                #     # else:
                #     #     logger.error("Test order placement failed.")
                # else:
                #     logger.error(f"Could not fetch current price for {DEFAULT_TRADING_PAIR}, skipping order placement test.")
            else:
                logger.warning("Skipping order placement/closing tests as NOT on TESTNET.")

        except Exception as e:
            logger.error(f"Error during connector module tests: {e}", exc_info=True)
        finally:
            if async_client:
                logger.info("Closing Bybit async client after tests.")
                await async_client.close()
    else:
        logger.error("Failed to initialize Bybit async client for tests.")

if __name__ == '__main__':
    # Setup basic logging for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    asyncio.run(run_connector_tests())

# --- Utility Functions ---
def timeframe_to_milliseconds(timeframe):
    """Converts a timeframe string (e.g., '1m', '5m', '1h', '1d') to milliseconds."""
    multipliers = {'m': 60, 'h': 60 * 60, 'd': 60 * 60 * 24, 'w': 60 * 60 * 24 * 7}
    try:
        unit = timeframe[-1].lower()
        if unit in multipliers:
            value = int(timeframe[:-1])
            return value * multipliers[unit] * 1000
        else:
            logger.warning(f"Unsupported timeframe unit: {unit} in '{timeframe}'")
            return None
    except (ValueError, IndexError, TypeError) as e:
        logger.error(f"Error parsing timeframe '{timeframe}': {e}")
        return None

# --- Main Execution / Examples (if run directly) ---
# Example usage for direct execution and testing of the connector functions
async def run_examples():
    logger.info("--- Testing Bybit Connector --- ")
    # testnet = True # Set to False for mainnet (USE WITH CAUTION)
    client = get_bybit_client(testnet=True) # Always use testnet for examples

    if client:
        logger.info("Bybit async client initialized successfully for tests.")
        try:
            # Test 1: Fetch Server Time
            server_time = await client.fetch_time() # Client is async, so await fetch_time
            logger.info(f"Bybit Server Time: {client.iso8601(server_time)}")

            # Test 2: Fetch OHLCV Data
            logger.info("\nTesting fetch_ohlcv function...")
            symbol_to_fetch = DEFAULT_TRADING_PAIR
            timeframe_to_fetch = DEFAULT_TIMEFRAME
            num_candles = 50
            ohlcv_df = await fetch_ohlcv(client, symbol=symbol_to_fetch, timeframe=timeframe_to_fetch, limit=num_candles)

            if ohlcv_df is not None and not ohlcv_df.empty:
                logger.info(f"\nSuccessfully fetched {len(ohlcv_df)} OHLCV data points for {symbol_to_fetch} ({timeframe_to_fetch}).")
                ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
                logger.info("\nDataFrame tail BEFORE adding indicators:")
                logger.info(ohlcv_df[['datetime', 'open', 'high', 'low', 'close', 'volume']].tail())
                
                # Test 3: Add Technical Indicators
                logger.info("\nTesting add_indicators function...")
                df_with_indicators = add_indicators(ohlcv_df.copy())
                
                if df_with_indicators is not None and not df_with_indicators.empty:
                    logger.info("\nDataFrame tail AFTER adding indicators:")
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', 120)
                    logger.info(df_with_indicators.tail())
                    added_cols = [col for col in df_with_indicators.columns if col not in ohlcv_df.columns and col != 'datetime']
                    logger.info(f"\nIndicators added: {added_cols}")
                else:
                    logger.info("Failed to add indicators or result was empty.")
            elif ohlcv_df is not None and ohlcv_df.empty:
                logger.info(f"No OHLCV data returned for {symbol_to_fetch}.")
            else:
                logger.info("Failed to fetch OHLCV data.")

        except ccxt.NetworkError as e:
            logger.error(f"Sync Test Network error: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Sync Test Exchange error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during sync tests: {e}")
        # Close the synchronous client if it was initialized
        # Although ccxt sync clients often don't require explicit close unless resources are held
        # logger.info("Closing sync client...") # Optional
    else:
        logger.info("Failed to initialize Bybit client for sync tests.")

    logger.info("\n--- Synchronous Tests Complete ---")

    # --- Section 2: Async WebSocket Test ---
    logger.info("\n--- Running Asynchronous WebSocket Test ---")
    
    async def main_websocket_test():
        # Initialize ccxt.pro client specifically for WebSocket
        async_client = None # Ensure it's defined for the finally block
        try:
            logger.info("Initializing async client (ccxt.pro.bybit)...")
            # Use same credentials, but with ccxt.pro
            exchange_config = {
                'apiKey': BYBIT_API_KEY,
                'secret': BYBIT_API_SECRET,
                'options': {
                    'defaultType': 'spot',
                }
            }
            async_client = ccxtpro.bybit(exchange_config)
            if BYBIT_TESTNET:
                logger.info("Setting sandbox mode for async client.")
                async_client.set_sandbox_mode(True)
            else:
                 logger.info("Using mainnet for async client.")
            
            logger.info("Async client initialized.")
            await stream_realtime_data(async_client, symbol=DEFAULT_TRADING_PAIR, data_type='ticker')
        except Exception as e:
            logger.error(f"Error during async client init or stream setup: {type(e).__name__}: {e}")
        finally:
            if async_client:
                logger.info("Closing async client connection...")
                await async_client.close()
                logger.info("Async client closed.")
            else:
                logger.info("Async client was not initialized, no connection to close.")

    # Run the main async test function
    try:
        asyncio.run(main_websocket_test())
    except KeyboardInterrupt:
        logger.info("\nWebSocket test interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"Unexpected error running main_websocket_test: {type(e).__name__}: {e}")
    finally:
        logger.info("WebSocket test section finished.")
