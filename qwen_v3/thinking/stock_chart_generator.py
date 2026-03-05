#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stock_chart_generator.py

Stock Performance Chart Generator using yfinance and matplotlib.
---------------------------------------------------------------------------------

### Description:
This script downloads historical adjusted closing price data for a list of stock
ticker symbols using the `yfinance` library. It then normalizes the data to a
starting value of 100% (or 1.0) on the first day, allowing for easy,
percentage-based performance comparison across different stocks, regardless of
their absolute price. Finally, it generates and saves a line chart of the
normalized price movements using `matplotlib`.

### Author & Copyright:
* **Author**: [Your Full Name or Company Name]
* **Email**: [Your Email Address]
* **Created**: 2025-11-11
* **License**: MIT License (or appropriate license, e.g., Apache 2.0, GPL)
* **Copyright**: (c) [Year] [Your Name or Company Name]

### Dependencies:
* `yfinance`: For downloading stock data.
* `matplotlib`: For plotting the chart.
* `pandas`: For data manipulation (DataFrame structure).
* `datetime`, `dateutil.relativedelta`: For date calculation.

### Main Function:
`generate_stock_chart(tickers: list, period_months: int = 6, output_file: str = "stock_price_chart.png")`
* Downloads data for the specified tickers and time period.
* Normalizes the 'Adj Close' prices for comparative charting.
* Saves the resulting plot as a PNG file.

### Usage Example:
"""
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


def generate_stock_chart(
    tickers: list, period_months: int = 6, output_file: str = "stock_price_chart.png"
):
    """
    Downloads stock data for the given tickers and generates a line chart
    of their adjusted closing prices.

    Args:
        tickers (list): A list of stock ticker symbols (e.g., ['NVDA', 'INTC', 'SONY']).
        period_months (int): The number of months back to retrieve data from.
        output_file (str): The filename for the generated chart image.
    """
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=period_months)

    ticker_list_str = ", ".join(tickers)
    print(
        f"Downloading data for: {ticker_list_str} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}..."
    )

    try:
        # Download the data
        # auto_adjust=True sets the prices to the adjusted close automatically
        data = yf.download(
            tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        # --- ROBUSTNESS CHECK 1: Is the DataFrame empty? ---
        if data.empty:
            print(
                "\n❌ Error: yfinance returned an empty dataset. This usually indicates a problem with your internet connection or the Yahoo Finance API."
            )
            return

        # For multiple tickers, yfinance returns a multi-index DataFrame,
        # where the first level is the metric (e.g., 'Adj Close').
        # If auto_adjust=True is used (as in this script), we extract 'Close' instead of 'Adj Close'
        # as the 'Adj Close' column is dropped/merged into 'Close'.
        # However, for consistency, we try 'Adj Close' and fall back to 'Close' if it fails.
        if "Adj Close" in data.columns:
            close_prices = data["Adj Close"]
        elif "Close" in data.columns.get_level_values(0):
            close_prices = data["Close"]
        elif "Close" in data.columns:
            # This handles the case where only one ticker is downloaded (no multi-index)
            close_prices = data["Close"]
        else:
            print(
                "\n❌ Error: Could not find 'Adj Close' or 'Close' price columns in the downloaded data structure."
            )
            return

    except Exception as e:
        # This catches errors like the KeyError for 'Adj Close' if the structure is unexpected
        print(f"FATAL ERROR during data processing: {e}")
        print("Please check your internet connection or try again later.")
        return

    # --- 1. Filter out failed downloads ---
    # We identify the tickers that successfully returned data by checking for all-NaN columns.
    successful_tickers = close_prices.columns[close_prices.notna().any()].tolist()

    if not successful_tickers:
        print(
            "\n❌ Error: Although data was downloaded, no valid price entries were found for any stock. Cannot generate chart."
        )
        return

    print(f"✅ Successfully retrieved data for: {', '.join(successful_tickers)}")

    # --- 2. Data Normalization ---
    # Filter the close_prices to only include successful tickers
    filtered_prices = close_prices[successful_tickers].copy()

    # Drop any leading rows with NaN values (important for the initial normalization point)
    filtered_prices = filtered_prices.dropna(how="all")

    # Normalize the prices so that the first price in the series is 100 for all stocks.
    normalized_prices = filtered_prices.div(filtered_prices.iloc[0]) * 100

    # --- 3. Plotting ---
    plt.figure(figsize=(12, 6))

    # Plot each stock's normalized performance
    for ticker in successful_tickers:
        plt.plot(
            normalized_prices.index,
            normalized_prices[ticker],
            label=ticker,
            linewidth=2,
        )

    # Add chart features
    plt.title(f"Normalized Stock Price Movement ({period_months} Months)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Price (%) - Base = 100%", fontsize=12)
    plt.legend(title="Ticker", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Use tight_layout to ensure labels are not cut off
    plt.tight_layout()

    # Save the chart
    plt.savefig(output_file)
    print(f"\n✅ Chart successfully saved to {output_file}")
    print(
        "The y-axis is normalized to 100% at the start date for performance comparison."
    )


if __name__ == "__main__":
    # Tickers: NVIDIA (NVDA), Intel (INTC), Sony (SONY)
    STOCK_TICKERS = ["NVDA", "INTC", "SONY"]

    # Run the chart generation for the last 6 months
    generate_stock_chart(STOCK_TICKERS, period_months=6)
