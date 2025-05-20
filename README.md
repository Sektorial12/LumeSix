# AI Futures Trading Bot (Bybit - BTC/SOL)

This project aims to develop a high-precision AI-powered trading bot that executes or signals futures trades on BTC or SOL using GPT-4 for trade evaluation and TimeGPT for short-term market forecasting.

## Project Goals

- Develop an AI trading bot for Bybit (BTC/USDT or SOL/USDT).
- Utilize GPT-4 (OpenAI API) for trade decision logic.
- Employ TimeGPT (Nixtla API) for market forecasting.
- Achieve >80% win rate or prediction accuracy in paper trading before live deployment.

## Setup Instructions

1.  **Prerequisites:**
    *   Ensure Python 3.10 or newer is installed on your system.
    *   Git (for version control, optional for initial local setup).

2.  **Create a Virtual Environment:**
    It is highly recommended to use a virtual environment to manage project dependencies.

    ```bash
    # Navigate to the project directory (e.g., lumesix)
    cd path/to/your/lumesix

    # Create a virtual environment (e.g., named .venv)
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment:**

    *   **Windows (Git Bash or similar):**
        ```bash
        source .venv/Scripts/activate
        ```
    *   **Windows (Command Prompt or PowerShell):**
        ```bash
        .\.venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```
    You should see the virtual environment name (e.g., `(.venv)`) in your terminal prompt.

4.  **Install Dependencies:**
    With the virtual environment activated, install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

    **Important Note on TA-Lib:**
    The `TA-Lib` library has C dependencies that need to be installed on your system before the Python wrapper can be installed via pip.
    *   **Windows:** You might need to download a precompiled `.whl` file for `TA-Lib` that matches your Python version (e.g., 3.10, 3.11) and system architecture (32-bit or 64-bit) from a source like [Unnofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib). Then install it using `pip install TA_Lib‑XYZ.whl`.
    *   **Linux:** Usually, you can install the C library using your system's package manager, e.g., `sudo apt-get install libta-lib0 libta-lib-dev` or `sudo yum install ta-lib ta-lib-devel`.
    *   **macOS:** You can use Homebrew: `brew install ta-lib`.
    After installing the C library, `pip install -r requirements.txt` should be able to install the Python wrapper for TA-Lib.
    If you encounter issues, please consult TA-Lib's official installation guides or search for OS-specific instructions.

5.  **Environment Variables:**
    Create a `.env` file in the root of the project directory to store your API keys and other sensitive configuration:
    ```
    BYBIT_API_KEY="YOUR_BYBIT_API_KEY"
    BYBIT_API_SECRET="YOUR_BYBIT_API_SECRET"
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    NIXTLA_API_KEY="YOUR_NIXTLA_API_KEY"
    # For Bybit Testnet (optional, if you want to specify explicitly)
    # BYBIT_TESTNET=True 
    ```
    **DO NOT commit the `.env` file to version control.** Add `.env` to your `.gitignore` file.

## Running the Bot

(Instructions to be added as development progresses)

## Roadmap Summary

- **Week 1:** Foundation, Bybit API integration, TA Engine, TimeGPT setup.
- **Week 2:** GPT-4 Signal System, prompt engineering.
- **Week 3:** Backtesting, Paper Trading Engine (Bybit Testnet).
- **Week 4:** Refinement, Automation, Performance Tuning.

Refer to `project-overview.txt` and `roadmap.txt` for more details.
