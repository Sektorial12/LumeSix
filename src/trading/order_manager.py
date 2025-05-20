import os
# --- Load .env early ---
from dotenv import load_dotenv
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"WARNING: .env file not found at {dotenv_path}. Environment variables for Telegram might not be loaded.")
# --- End Load .env early ---

import ccxt.async_support as ccxt
import logging
from src.notifications.telegram_notifier import send_telegram_message_async

# --- Setup Logger ---
logger = logging.getLogger(__name__)
# Note: Logger level and handlers will be configured by the root application (main.py)
# --- End Logger Setup ---

# --- Configuration --- 
# Example: Get a default order size from .env or use a fixed value
DEFAULT_ORDER_SIZE_USDT = float(os.getenv('DEFAULT_ORDER_SIZE_USDT', 10)) # Default $10 order size
DEFAULT_TRADING_PAIR = os.getenv('DEFAULT_TRADING_PAIR', 'BTC/USDT')

async def place_market_order(client: ccxt.Exchange, symbol: str, side: str, amount_usdt: float, notifier=None):
    """ 
    Places a market order on Bybit (linear perpetuals).

    Args:
        client: Initialized async ccxt Bybit client.
        symbol: Trading symbol (e.g., 'BTC/USDT').
        side: 'buy' or 'sell'.
        amount_usdt: The approximate amount in USDT to trade.
                     Note: Bybit requires order size in base currency (e.g., BTC).
                     We'll calculate this based on the current price.

    Returns:
        dict: The order information dict from ccxt, or None if an error occurred.
    """
    logger.info(f"\n--- Attempting to place {side} market order for {symbol} ({amount_usdt} USDT) ---")
    if side not in ['buy', 'sell']:
        logger.error(f"Error: Invalid order side '{side}'. Must be 'buy' or 'sell'.")
        return None

    try:
        # 1. Fetch ticker to get current price for size calculation
        logger.info(f"Fetching ticker for {symbol}...")
        ticker = await client.fetch_ticker(symbol)
        last_price = ticker.get('last')
        if not last_price:
            logger.error(f"Error: Could not fetch last price for {symbol} to calculate order size.")
            return None
        logger.info(f"Current price for {symbol}: {last_price}")
        
        # --- Add this new logging block ---
        if symbol in client.markets:
            market = client.markets[symbol]
            min_amount = market['limits']['amount']['min'] if market['limits']['amount'] and 'min' in market['limits']['amount'] else 'N/A'
            max_amount = market['limits']['amount']['max'] if market['limits']['amount'] and 'max' in market['limits']['amount'] else 'N/A'
            amount_step = market['precision']['amount'] # Smallest increment for amount
            price_step = market['precision']['price']   # Smallest increment for price

            logger.info(f"Market {symbol} Details (from CCXT client):")
            logger.info(f"  - Amount Precision (step size): {amount_step}")
            logger.info(f"  - Price Precision (step size): {price_step}")
            logger.info(f"  - Minimum Order Amount (base currency): {min_amount}")
            logger.info(f"  - Maximum Order Amount (base currency): {max_amount}")
        else:
            logger.warning(f"Market data for {symbol} not found in client.markets after fetch_ticker.")
        # --- End of new logging block ---

        base_currency = symbol.split('/')[0]

        params = {
            'category': 'spot' # Explicitly set for V5 API
        }

        if side == 'buy':
            # For market BUY, use amount_usdt directly as cost.
            # ccxt's create_market_buy_order_with_cost handles cost precision internally.
            logger.info(f"Placing BUY market order with cost: {amount_usdt:.2f} USDT for {symbol}...")
            order = await client.create_market_buy_order_with_cost(symbol, amount_usdt, params=params)
        
        elif side == 'sell':
            # For market SELL, calculate amount in base currency and ensure precision
            amount_base_currency_raw = amount_usdt / last_price
            logger.info(f"Calculated SELL order size (raw): {amount_base_currency_raw:.8f} {base_currency}")
            
            amount_base_currency_adjusted = float(client.amount_to_precision(symbol, amount_base_currency_raw))
            logger.info(f"Adjusted SELL order size (to precision): {amount_base_currency_adjusted:.8f} {base_currency}")

            # Optional: Check if adjusted amount is still above minimum (precision might round down)
            min_allowable_amount = client.markets[symbol]['limits']['amount']['min']
            if min_allowable_amount is not None and amount_base_currency_adjusted < min_allowable_amount:
                logger.warning(f"Adjusted SELL amount {amount_base_currency_adjusted:.8f} {base_currency} is below minimum {min_allowable_amount:.8f} {base_currency} after precision. Order might fail.")
            
            logger.info(f"Placing SELL market order for {amount_base_currency_adjusted:.8f} {base_currency}...")
            order = await client.create_market_order(symbol, 'sell', amount_base_currency_adjusted, params=params)
        
        else:
            logger.error(f"Invalid order side: {side}")
            return None

        if order and notifier:
            # Robustly construct Telegram message parts
            order_id_str = str(order.get('id', 'N/A'))
            order_symbol_str = str(order.get('symbol', symbol))
            # Initial side is what we sent, 'side' in order response might be None initially
            order_side_str = side.upper() 
            order_type_str = str(order.get('type', 'market')).capitalize()
            # Initial status from response might be None or an interim one. 'Submitted' is a safe assumption.
            order_status_str = str(order.get('status', 'Submitted')).capitalize()

            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1]

            details_str = ""
            if side == 'buy':
                # For market buy with cost, the cost is known (amount_usdt)
                details_str += f"**Order Cost:** {amount_usdt:.2f} {quote_currency}\n"
                filled_amount = order.get('filled')
                if isinstance(filled_amount, (float, int)):
                    details_str += f"**Amount Filled:** {filled_amount:.6f} {base_currency}\n"
            elif side == 'sell':
                # For market sell, the intended amount is known (amount_base_currency_adjusted)
                details_str += f"**Amount Intended:** {amount_base_currency_adjusted:.6f} {base_currency}\n"
                filled_amount = order.get('filled')
                if isinstance(filled_amount, (float, int)):
                    details_str += f"**Amount Filled:** {filled_amount:.6f} {base_currency}\n"
            
            avg_price_val = order.get('average')
            if isinstance(avg_price_val, (float, int)):
                details_str += f"**Avg. Price:** {avg_price_val:.2f} {quote_currency}\n"
            
            final_cost_val = order.get('cost')
            if isinstance(final_cost_val, (float, int)):
                details_str += f"**Total Value:** {final_cost_val:.2f} {quote_currency}\n"

            message = (
                f"✅ **Trade Alert on {client.id.capitalize()}** ✅\n\n"
                f"**Symbol:** {order_symbol_str}\n"
                f"**Side:** {order_side_str}\n"
                f"**Type:** {order_type_str}\n"
                f"{details_str}"
                f"**Order ID:** {order_id_str}\n"
                f"**Status:** {order_status_str}"
            )
            try:
                await notifier(message)
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")
        
        logger.info(f"Order placement attempt for {symbol} {side} completed.")
        logger.info(f"Order response from exchange: {order}")
        return order

    except ccxt.AuthenticationError as e:
        logger.error(f"Authentication Error placing order: {e}", exc_info=True)
        # Potentially trigger re-authentication or alert
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange Error placing order for {symbol}: {e}", exc_info=True)
        # Handle specific exchange errors (e.g., insufficient balance, invalid symbol)
        return None
    except ccxt.NetworkError as e:
        logger.error(f"Network Error placing order: {e}", exc_info=True)
        # Handle network issues, maybe retry
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred placing order: {e}", exc_info=True)
        return None

# --- Example Usage (Placeholder - Requires async environment) ---
async def test_order_placement():
    """Tests placing a market order (requires async execution)."""
    logger.info("--- Testing Order Manager Module ---")
    
    # Need to re-import and initialize a client specifically for testing
    from src.connectors.bybit_connector import get_bybit_client # Use the existing function
    
    logger.info("Initializing Bybit test client...")
    test_client = get_bybit_client(testnet=True) # Ensure testnet is used!
    
    if not test_client:
        logger.error("Failed to initialize test client. Aborting test.")
        return
        
    # --- Test Placing a BUY order ---
    logger.info("\n--- Testing BUY Order --- (TESTNET)")
    test_symbol = "BTC/USDT"
    test_amount_usdt = 50.0  # Increased from 15.0 to 50.0
    buy_order_result = await place_market_order(
        client=test_client, 
        symbol=test_symbol, 
        side='buy', 
        amount_usdt=test_amount_usdt,
        notifier=send_telegram_message_async
    )
    if buy_order_result:
        logger.info(f"BUY order test processing. Status from exchange: {buy_order_result.get('status', 'N/A')}")
    else:
        logger.error("BUY order test failed (place_market_order returned None or error).")

    # --- Test Placing a SELL order ---
    # Note: You might need an open position to successfully place a closing SELL order
    # or configure hedge mode if applicable.
    # For simplicity, we just attempt it. 
    logger.info("\n--- Testing SELL Order --- (TESTNET)")
    sell_order_result = await place_market_order(
        client=test_client, 
        symbol=test_symbol, 
        side='sell', 
        amount_usdt=test_amount_usdt,
        notifier=send_telegram_message_async
    )
    if sell_order_result:
        logger.info(f"SELL order test processing. Status from exchange: {sell_order_result.get('status', 'N/A')}")
    else:
        logger.error("SELL order test failed (place_market_order returned None or error).")

    # --- Cleanup ---
    logger.info("\nClosing test client connection...")
    await test_client.close()
    logger.info("Test client closed.")
    logger.info("\n--- Order Manager Module Test Complete ---")

if __name__ == '__main__':
    # This basicConfig is for when running the module directly for testing.
    # When imported, the root logger's config (from main.py) should take precedence.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    import asyncio
    try:
        asyncio.run(test_order_placement())
    except Exception as e:
        logger.error(f"Error running test: {e}", exc_info=True)
