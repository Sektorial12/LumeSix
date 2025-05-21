import os
import logging
logger = logging.getLogger(__name__)

# Bybit API Credentials
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
# Convert string 'True' to boolean True, otherwise False
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "False").lower() == "true"

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Nixtla (TimeGPT) API Key
NIXTLA_API_KEY = os.getenv("NIXTLA_API_KEY")

# Basic validation (optional, but good practice)
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    logger.warning("Warning: BYBIT_API_KEY or BYBIT_API_SECRET not found. Bybit functionality will be limited.")

if not OPENAI_API_KEY:
    logger.warning("Warning: OPENAI_API_KEY not found. OpenAI functionality will be limited.")

if not NIXTLA_API_KEY:
    logger.warning("Warning: NIXTLA_API_KEY not found. TimeGPT functionality will be limited.")

# You can add other configurations here as needed
# For example, default trading pairs, timeframes, etc.
DEFAULT_TRADING_PAIR = "BTCUSDT"
DEFAULT_TIMEFRAME = "5m"
