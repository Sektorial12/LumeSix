import asyncio
import logging
import os
import telegram
from telegram.constants import ParseMode

# Configure logger for this module
logger = logging.getLogger(__name__)

# Load credentials from environment variables at the module level
# These will be populated by dotenv in main.py or by the test block below
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

def _escape_markdown_v2(text: str) -> str:
    # Characters to escape for MarkdownV2
    # Reordered to avoid self-escaping backslashes first
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    # Escape backslash first, then other characters
    escaped_text = text.replace('\\', '\\\\') # Escape backslashes themselves
    for char in escape_chars:
        escaped_text = escaped_text.replace(char, f'\\{char}')
    return escaped_text

async def _send_message_async_internal(message: str, bot_token: str, chat_id: str):
    """
    Asynchronously sends a message using a new Bot instance for each call.
    Manages bot lifecycle with async with.
    """
    if not bot_token or not chat_id:
        logger.error("Bot token or chat ID is missing for sending Telegram message.")
        return

    bot = telegram.Bot(token=bot_token)
    escaped_message = _escape_markdown_v2(message)

    try:
        async with bot:  # Handles bot.initialize() and bot.shutdown()
            await bot.send_message(
                chat_id=chat_id,
                text=escaped_message,
                parse_mode=ParseMode.MARKDOWN_V2
            )
        logger.info(f"Telegram message sent to chat ID {chat_id[:4] if chat_id and len(chat_id) >=4 else chat_id}...: \"{message[:60].replace(chr(10), ' ')}...\"")
    except telegram.error.TelegramError as e:
        logger.error(f"Failed to send Telegram message due to Telegram API error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending Telegram message: {e}")

async def send_telegram_message_async(message: str):
    """
    Asynchronously sends a Telegram message.
    This function is intended to be called from an existing asyncio event loop.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram Bot Token or Chat ID is not set in environment variables. Notification will be skipped.")
        return

    # Directly await the internal async function
    await _send_message_async_internal(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)


if __name__ == '__main__':
    # This block allows for direct testing of the notifier module.
    # Example: python -m src.notifications.telegram_notifier
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()] # Ensure logs go to console for this test
    )
    logger.info("Testing Telegram Notifier (telegram_notifier.py direct execution)...")
    
    from dotenv import load_dotenv
    load_dotenv() # Load environment variables from .env file

    # Re-populate the global module-level variables for the test execution context
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    # No need to manage _bot_instance anymore here

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("CRITICAL: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in environment variables for testing after attempting to load .env.")
        logger.error("Please ensure they are in your .env file.")
    else:
        logger.info(f"Test token loaded (preview): {TELEGRAM_BOT_TOKEN[:5]}..." )
        logger.info(f"Test chat ID loaded (preview): {TELEGRAM_CHAT_ID[:4] if TELEGRAM_CHAT_ID and len(TELEGRAM_CHAT_ID) >=4 else TELEGRAM_CHAT_ID}...")

        # Since send_telegram_message_async is now async, we need asyncio.run for each call in this test block,
        # or gather them. For simplicity, run one by one.
        async def main_test():
            await send_telegram_message_async("LumeSix Bot Test (1/3): Hello! This is a direct test of `telegram_notifier.py`.")
            await send_telegram_message_async("LumeSix Bot Test (2/3): Message with *special characters* like _underscore_ and [brackets]. They should be escaped for MarkdownV2.")
            await send_telegram_message_async(
                "LumeSix Bot Test (3/3):\nThis is a multi-line message.\nSecond line."
            )
        
        asyncio.run(main_test())

        logger.info("Test messages have been dispatched. Check your Telegram chat.")
        logger.info("If messages were not received, check logs above for errors, ensure bot token and chat ID are correct, and the bot has permission to send messages to the chat.")
