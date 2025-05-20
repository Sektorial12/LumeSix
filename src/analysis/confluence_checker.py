import pandas as pd
from scipy.stats import linregress

def check_timegpt_alignment(forecast_df: pd.DataFrame, signal_direction: str, current_price: float, num_points: int = 3) -> bool:
    """
    Checks if the TimeGPT forecast aligns with the given signal direction.
    Alignment is determined by the slope of a linear regression on the next 'num_points' of the forecast.

    Args:
        forecast_df (pd.DataFrame): DataFrame with 'TimeGPT' forecast values.
        signal_direction (str): 'BUY' or 'SELL'.
        current_price (float): The current market price for context (not directly used in slope yet, but good to have).
        num_points (int): Number of future forecast points to consider for the trend.

    Returns:
        bool: True if aligned, False otherwise.
    """
    if forecast_df is None or forecast_df.empty or len(forecast_df) < num_points:
        print("Confluence: Not enough forecast data for TimeGPT alignment check.")
        return False

    try:
        # Take the next 'num_points' from the forecast
        relevant_forecast = forecast_df['TimeGPT'].iloc[:num_points].values
        x = list(range(len(relevant_forecast))) # Time steps 0, 1, 2, ...

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, relevant_forecast)

        print(f"Confluence: TimeGPT forecast slope for next {num_points} points: {slope:.4f}")

        if signal_direction == 'BUY':
            # For BUY, we want an upward trend (positive slope)
            # We can add a threshold later if a slightly positive slope isn't strong enough
            return slope > 0 
        elif signal_direction == 'SELL':
            # For SELL, we want a downward trend (negative slope)
            return slope < 0
        else:
            return False # Unknown signal direction
            
    except Exception as e:
        print(f"Confluence: Error during TimeGPT alignment check: {e}")
        return False

def check_ta_confirmation(latest_indicators: pd.Series, 
                          signal_direction: str, 
                          confirmation_rules_config: dict, 
                          indicator_params_config: dict) -> bool:
    """
    Checks if Technical Analysis indicators confirm the given signal direction based on configuration.

    Args:
        latest_indicators (pd.Series): Latest calculated TA values.
        signal_direction (str): 'BUY' or 'SELL'.
        confirmation_rules_config (dict): Config for TA confirmation rules 
                                          (e.g., from config['strategy']['ta_confirmation_rules']).
        indicator_params_config (dict): Config for TA indicator parameters 
                                        (e.g., from config['technical_analysis']).

    Returns:
        bool: True if TA confirms, False otherwise.
    """
    if latest_indicators is None or latest_indicators.empty:
        print("Confluence: Not enough indicator data for TA confirmation.")
        return False

    try:
        # --- RSI Check --- 
        rsi_confirmed = True # Default to true if not enabled
        if confirmation_rules_config.get('rsi_enabled', False):
            rsi_len = indicator_params_config['rsi']['length']
            rsi_col = f'RSI_{rsi_len}'
            rsi = latest_indicators.get(rsi_col)
            if rsi is None:
                print(f"Confluence: {rsi_col} not found in latest indicators. Skipping RSI check.")
                return False # Critical indicator missing
            
            if signal_direction == 'BUY':
                rsi_threshold = confirmation_rules_config.get('rsi_buy_threshold', 30)
                rsi_confirmed = rsi < rsi_threshold
                print(f"Confluence (BUY TA - RSI): Rule: <{rsi_threshold}. Actual:{rsi:.2f}. Confirmed: {rsi_confirmed}")
            elif signal_direction == 'SELL':
                rsi_threshold = confirmation_rules_config.get('rsi_sell_threshold', 70)
                rsi_confirmed = rsi > rsi_threshold
                print(f"Confluence (SELL TA - RSI): Rule: >{rsi_threshold}. Actual:{rsi:.2f}. Confirmed: {rsi_confirmed}")
        else:
            print("Confluence (TA - RSI): RSI check disabled by config.")

        # --- MACD Check --- 
        macd_confirmed = True # Default to true if not enabled
        if confirmation_rules_config.get('macd_enabled', False):
            macd_fast = indicator_params_config['macd']['fast_period']
            macd_slow = indicator_params_config['macd']['slow_period']
            macd_signal_p = indicator_params_config['macd']['signal_period']
            macd_col = f'MACD_{macd_fast}_{macd_slow}_{macd_signal_p}'
            macds_col = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal_p}'
            
            macd = latest_indicators.get(macd_col)
            macd_signal = latest_indicators.get(macds_col)

            if macd is None or macd_signal is None:
                print(f"Confluence: {macd_col} or {macds_col} not found. Skipping MACD check.")
                return False # Critical indicator missing

            if signal_direction == 'BUY':
                macd_confirmed = macd > macd_signal
                print(f"Confluence (BUY TA - MACD): Rule: MACD > Signal. Actual M:{macd:.2f},S:{macd_signal:.2f}. Confirmed: {macd_confirmed}")
            elif signal_direction == 'SELL':
                macd_confirmed = macd < macd_signal
                print(f"Confluence (SELL TA - MACD): Rule: MACD < Signal. Actual M:{macd:.2f},S:{macd_signal:.2f}. Confirmed: {macd_confirmed}")
        else:
            print("Confluence (TA - MACD): MACD check disabled by config.")

        # --- EMA Check --- 
        ema_confirmed = True # Default to true if not enabled
        if confirmation_rules_config.get('ema_enabled', False):
            close_price = latest_indicators.get('close')
            if close_price is None:
                print("Confluence: Close price not found for EMA check.")
                return False # Critical data missing

            ema_period_key = ''
            if signal_direction == 'BUY':
                ema_period_key = confirmation_rules_config.get('ema_to_check_buy', 'short_period_ema')
            elif signal_direction == 'SELL':
                ema_period_key = confirmation_rules_config.get('ema_to_check_sell', 'short_period_ema')
            
            # Resolve the key (e.g., 'short_period_ema') to the actual period number (e.g., 20)
            # The key from config ('short_period_ema') should match a key in indicator_params_config['emas'] that holds the period value
            # e.g. indicator_params_config['emas']['short_period'] gives 20
            actual_ema_period_name = ema_period_key.replace('_ema', '') # from 'short_period_ema' to 'short_period'
            ema_period_val = indicator_params_config['emas'].get(actual_ema_period_name)

            if ema_period_val is None:
                print(f"Confluence: EMA period for key '{ema_period_key}' (resolved to '{actual_ema_period_name}') not found in indicator_params_config['emas']. Skipping EMA check.")
                return False

            ema_col = f'EMA_{ema_period_val}'
            ema_value = latest_indicators.get(ema_col)

            if ema_value is None:
                print(f"Confluence: {ema_col} not found. Skipping EMA check.")
                return False # Critical indicator missing

            if signal_direction == 'BUY':
                ema_confirmed = close_price > ema_value
                print(f"Confluence (BUY TA - EMA): Rule: Close > {ema_col}. Actual C:{close_price:.2f},E:{ema_value:.2f}. Confirmed: {ema_confirmed}")
            elif signal_direction == 'SELL':
                ema_confirmed = close_price < ema_value
                print(f"Confluence (SELL TA - EMA): Rule: Close < {ema_col}. Actual C:{close_price:.2f},E:{ema_value:.2f}. Confirmed: {ema_confirmed}")
        else:
            print("Confluence (TA - EMA): EMA check disabled by config.")
        
        # Overall TA Confirmation: All enabled checks must pass
        overall_confirmed = rsi_confirmed and macd_confirmed and ema_confirmed
        print(f"Confluence (Overall TA for {signal_direction}): RSI Conf: {rsi_confirmed}, MACD Conf: {macd_confirmed}, EMA Conf: {ema_confirmed}. Overall: {overall_confirmed}")
        return overall_confirmed

    except Exception as e:
        print(f"Confluence: Error during TA confirmation check: {e}")
        return False

# Example usage (for testing):
if __name__ == '__main__':
    # Mock TimeGPT forecast data
    mock_forecast_buy = pd.DataFrame({'TimeGPT': [100, 101, 102, 103, 104]})
    mock_forecast_sell = pd.DataFrame({'TimeGPT': [100, 99, 98, 97, 96]})
    mock_forecast_flat = pd.DataFrame({'TimeGPT': [100, 100.1, 100, 99.9, 100]})

    print("--- TimeGPT Alignment Tests ---")
    print(f"BUY signal, rising forecast: {check_timegpt_alignment(mock_forecast_buy, 'BUY', 99.0)}") # Expected: True
    print(f"SELL signal, falling forecast: {check_timegpt_alignment(mock_forecast_sell, 'SELL', 101.0)}") # Expected: True
    print(f"BUY signal, falling forecast: {check_timegpt_alignment(mock_forecast_sell, 'BUY', 101.0)}") # Expected: False
    print(f"SELL signal, rising forecast: {check_timegpt_alignment(mock_forecast_buy, 'SELL', 99.0)}") # Expected: False
    print(f"BUY signal, flat forecast: {check_timegpt_alignment(mock_forecast_flat, 'BUY', 100.0)}") # Expected: False (or slightly pos/neg)

    # --- Mock TA Data & Configs for testing check_ta_confirmation ---
    mock_latest_indicators_buy_confirm = pd.Series({
        'RSI_14': 25, 'MACD_12_26_9': 10, 'MACDs_12_26_9': 5, 'close': 105, 'EMA_20': 100, 'EMA_50': 95
    })
    mock_latest_indicators_buy_fail_rsi = pd.Series({
        'RSI_14': 35, 'MACD_12_26_9': 10, 'MACDs_12_26_9': 5, 'close': 105, 'EMA_20': 100, 'EMA_50': 95
    })
    mock_latest_indicators_sell_confirm = pd.Series({
        'RSI_14': 75, 'MACD_12_26_9': -10, 'MACDs_12_26_9': -5, 'close': 95, 'EMA_20': 100, 'EMA_50': 105
    })

    # Default indicator parameters as per our config.yaml structure
    default_indicator_params = {
        'rsi': {'length': 14},
        'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'emas': {'short_period': 20, 'long_period': 50}
    }

    # Test case 1: BUY signal, all rules confirm
    rules_buy_confirm_all_enabled = {
        'rsi_enabled': True, 'rsi_buy_threshold': 30, 'rsi_sell_threshold': 70,
        'macd_enabled': True,
        'ema_enabled': True, 'ema_to_check_buy': 'short_period_ema', 'ema_to_check_sell': 'short_period_ema'
    }
    print("\n--- TA Confirmation Tests ---")
    print("Test 1: BUY signal, all rules confirm (Expected: True)")
    result1 = check_ta_confirmation(mock_latest_indicators_buy_confirm, 'BUY', rules_buy_confirm_all_enabled, default_indicator_params)
    print(f"Result 1: {result1}\n")

    # Test case 2: BUY signal, RSI fails (RSI=35, threshold=30)
    print("Test 2: BUY signal, RSI fails (Expected: False)")
    result2 = check_ta_confirmation(mock_latest_indicators_buy_fail_rsi, 'BUY', rules_buy_confirm_all_enabled, default_indicator_params)
    print(f"Result 2: {result2}\n")

    # Test case 3: SELL signal, all rules confirm
    print("Test 3: SELL signal, all rules confirm (Expected: True)")
    result3 = check_ta_confirmation(mock_latest_indicators_sell_confirm, 'SELL', rules_buy_confirm_all_enabled, default_indicator_params)
    print(f"Result 3: {result3}\n")

    # Test case 4: BUY signal, MACD disabled, others confirm
    rules_macd_disabled = rules_buy_confirm_all_enabled.copy()
    rules_macd_disabled['macd_enabled'] = False
    print("Test 4: BUY signal, MACD disabled, others confirm (Expected: True)")
    # For this test, let's assume RSI and EMA would pass. mock_latest_indicators_buy_confirm still works.
    result4 = check_ta_confirmation(mock_latest_indicators_buy_confirm, 'BUY', rules_macd_disabled, default_indicator_params)
    print(f"Result 4: {result4}\n")

    # Test case 5: SELL signal, EMA uses 'long_period_ema' and confirms
    rules_sell_long_ema = rules_buy_confirm_all_enabled.copy()
    rules_sell_long_ema['ema_to_check_sell'] = 'long_period_ema'
    # For sell, close < EMA_long (95 < 105) should be true for mock_latest_indicators_sell_confirm
    print("Test 5: SELL signal, EMA uses long_period_ema and confirms (Expected: True)")
    result5 = check_ta_confirmation(mock_latest_indicators_sell_confirm, 'SELL', rules_sell_long_ema, default_indicator_params)
    print(f"Result 5: {result5}")
