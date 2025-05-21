import os
import pandas as pd
from openai import OpenAI, OpenAIError
import logging

# --- Setup Logger ---
logger = logging.getLogger(__name__)
# --- End Logger Setup ---

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# --- OpenAI Client Initialization ---
def get_openai_client():
    """Initializes and returns the OpenAI client."""
    if not OPENAI_API_KEY:
        logger.warning("Error: OPENAI_API_KEY not found in environment variables.")
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Test connection - Attempt a simple listing models call
        # client.models.list() 
        logger.info("OpenAI client initialized successfully.")
        return client
    except OpenAIError as e:
        logger.error(f"Error initializing OpenAI client: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during OpenAI client initialization: {e}")
        return None

# --- Market Evaluation Function ---
def evaluate_market_conditions(client, historical_data, forecast_data, symbol="BTC/USDT"):
    """
    Uses GPT-4 to evaluate market conditions based on historical data (with indicators)
    and a forecast.

    Args:
        client (OpenAI): The initialized OpenAI client.
        historical_data (pd.DataFrame): DataFrame with recent OHLCV and technical indicators.
        forecast_data (pd.DataFrame): DataFrame with the price forecast (from Nixtla/TimeGPT).
        symbol (str): The trading symbol being evaluated.

    Returns:
        tuple(str, int): A tuple containing the evaluation (e.g., "BUY", "SELL", "HOLD") 
                         and the confidence score (0-100), or (None, None) if an error occurs.
    """
    if client is None:
        logger.error("Error: OpenAI client is not initialized.")
        return None, None
    if historical_data is None or historical_data.empty:
        logger.error("Error: Historical data is missing for evaluation.")
        return None, None

    logger.info("\n--- Preparing data for GPT-4 evaluation ---")
    # Select relevant recent data (e.g., last 10 candles)
    recent_data = historical_data.tail(10).to_string()
    
    # Format forecast data
    forecast_str = forecast_data.to_string() if forecast_data is not None else "No forecast available."

    # --- Construct the Prompt ---
    prompt = f"""
You are an expert crypto trading analyst specializing in short-term {symbol} futures trading. 
Your goal is to provide a concise trading recommendation (BUY, SELL, or HOLD) and a confidence score (0-100) based on the provided data.

Consider the recent market trend, technical indicators, and the price forecast.

Recent Market Data & Indicators (last 10 periods):
{recent_data}

Price Forecast (next {len(forecast_data) if forecast_data is not None else 'N/A'} periods):
{forecast_str}

Analysis Task:
Based *only* on the data provided above, what is your trading recommendation for {symbol} for the next 1-2 hours? 
Also, provide a confidence score (integer from 0 to 100) indicating your certainty in this recommendation.

Please respond in the following format ONLY:
DECISION: [BUY|SELL|HOLD]
CONFIDENCE: [0-100]

Example Response:
DECISION: BUY
CONFIDENCE: 85
"""

    logger.info("--- Sending request to GPT-4 ---")
    # logger.debug("Prompt:\n", prompt) # Uncomment for debugging

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview", # Reverted based on user feedback
            messages=[
                {"role": "system", "content": "You are an expert crypto trading analyst providing decisions and confidence scores in the specified format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50, # Increased slightly for two lines
            temperature=0.2 # Low temperature for more deterministic output
        )
        
        raw_response = response.choices[0].message.content.strip()
        logger.info(f"GPT-4 Raw Response:\n{raw_response}")
        
        # --- Parse the response --- 
        decision = None
        confidence = None
        lines = raw_response.split('\n')
        if len(lines) == 2:
            try:
                decision_part = lines[0].split(':')
                confidence_part = lines[1].split(':')
                
                if len(decision_part) == 2 and decision_part[0].strip().upper() == 'DECISION':
                    decision_val = decision_part[1].strip().upper()
                    if decision_val in ["BUY", "SELL", "HOLD"]:
                        decision = decision_val
                    else:
                         logger.warning(f"Invalid decision value '{decision_val}' received.")

                if len(confidence_part) == 2 and confidence_part[0].strip().upper() == 'CONFIDENCE':
                    confidence_val = confidence_part[1].strip()
                    try:
                        confidence = int(confidence_val)
                        if not (0 <= confidence <= 100):
                            logger.warning(f"Confidence value {confidence} out of range (0-100).")
                            confidence = None # Invalidate if out of range
                    except ValueError:
                        logger.warning(f"Could not parse confidence value '{confidence_val}' as integer.")
                        
            except Exception as parse_err:
                logger.error(f"Error parsing GPT-4 response format: {parse_err}")
        else:
            logger.warning("GPT-4 response did not contain the expected two lines (DECISION, CONFIDENCE).")

        if decision and confidence is not None:
            logger.info(f"Parsed Evaluation: Decision={decision}, Confidence={confidence}")
            return decision, confidence
        else:
            logger.error("Failed to parse valid decision and confidence from response.")
            return None, None # Indicate failure to parse

    except OpenAIError as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during GPT-4 evaluation: {e}")
        return None, None

# --- Example Usage (Placeholder) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Testing GPT Evaluator Module ---")
    
    # 1. Initialize Client
    openai_client = get_openai_client()

    if openai_client:
        logger.info("\n--- Simulating Evaluation Request ---")
        # Create sample DataFrames (replace with real data structure)
        sample_hist_data = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:05', '2023-01-01 10:10']), 
            'close': [100, 101, 102],
            'RSI_14': [50, 60, 70],
            'MACD_12_26_9': [0.5, 0.6, 0.7]
            # Add other relevant indicator columns based on technical_analyzer.py
        })
        sample_forecast_data = pd.DataFrame({
            'ds': pd.to_datetime(['2023-01-01 10:15', '2023-01-01 10:20']), 
            'TimeGPT': [103, 104]
        })

        logger.info("Sample Historical Data:")
        logger.info(sample_hist_data)
        logger.info("\nSample Forecast Data:")
        logger.info(sample_forecast_data)

        # 2. Get Evaluation
        decision_result, confidence_result = evaluate_market_conditions(
            client=openai_client,
            historical_data=sample_hist_data,
            forecast_data=sample_forecast_data
        )

        if decision_result is not None and confidence_result is not None:
            logger.info(f"\nEvaluation Result: Decision={decision_result}, Confidence={confidence_result}")
        else:
            logger.error("\nFailed to get evaluation or parse result.")
    else:
        logger.error("\nSkipping evaluation test due to client initialization failure.")

    logger.info("\n--- GPT Evaluator Module Test Complete ---")
