
import yfinance as yf
import pandas as pd
import os

# Prompt the user to input the two ticker symbols, time period, and file save location
ticker_symbol1 = input("Enter the first ticker symbol (e.g., AAPL for Apple): ").upper()
ticker_symbol2 = input("Enter the second ticker symbol (e.g., MSFT for Microsoft): ").upper()
time_period = input("Enter the period (e.g., 1d, 5d, 1mo, 1y, 5y, max): ").lower()
save_location = input("Enter the full path where you want to save the CSV file (e.g., /Users/username/Documents/): ")

# Ensure the directory exists
if not os.path.exists(save_location):
    print(f"The directory {save_location} does not exist.")
else:
    # Download historical data for the specified stocks
    ticker_data1 = yf.Ticker(ticker_symbol1)
    ticker_data2 = yf.Ticker(ticker_symbol2)

    # Fetch the data for the specified period
    try:
        data1 = ticker_data1.history(period=time_period)
        data2 = ticker_data2.history(period=time_period)

        # Check if data was successfully retrieved
        if data1.empty or data2.empty:
            print(f"No data found for one or both tickers ({ticker_symbol1}, {ticker_symbol2}) over the period {time_period}.")
        else:
            # Extract only the Close columns and rename them for clarity
            close1 = data1[['Close']].rename(columns={'Close': f'{ticker_symbol1}_Close'})
            close2 = data2[['Close']].rename(columns={'Close': f'{ticker_symbol2}_Close'})

            # Merge the two DataFrames on the Date index
            merged_data = pd.merge(close1, close2, left_index=True, right_index=True, how='outer')

            # Display the first few rows of the merged data
            print(merged_data.head())

            # Create the full file path
            csv_file_path = os.path.join(save_location, f'{ticker_symbol1}_{ticker_symbol2}_close_data.csv')

            # Save the merged data to a CSV file
            merged_data.to_csv(csv_file_path)

            print(f"Data for {ticker_symbol1} and {ticker_symbol2} has been saved to {csv_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
