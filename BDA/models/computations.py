import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
path1 = "/Users/kayttaja/Desktop/BDA/data/raw/Shiller_data.xls"
shiller_df = pd.read_excel(path1, skiprows=7)

print(shiller_df.head())
print(shiller_df.columns)

# Keep only CAPE, CPI, GS10, Date and 10 Year Annualized Stock Real Return
shiller_df = shiller_df[
    ["Date", "CAPE", "CPI", "GS10", "10 Year Annualized Stock Real Return"]
]
# Rename columns for easier access
shiller_df.columns = ["Date", "CAPE", "CPI", "GS10", "Real_Return_10Y"]
# Convert Date to datetime format
shiller_df["Date"] = shiller_df["Date"] * 100
shiller_df["Date"] = pd.to_datetime(shiller_df["Date"], format="%Y%m")
# Sort by Date
shiller_df = shiller_df.sort_values(by="Date").reset_index(drop=True)
print(shiller_df.tail())

# Compute year over year inflation
shiller_df["Inflation"] = shiller_df["CPI"].pct_change(periods=12) * 100
# Compute Real_Return_10Y to percent
shiller_df["Real_Return_10Y"] = shiller_df["Real_Return_10Y"] * 100
print(shiller_df.tail())

# Estimate the mean and std of inflation and GS10
print("Inflation Mean:", shiller_df["Inflation"].mean())
print("Inflation Std:", shiller_df["Inflation"].std())
print("GS10 Mean:", shiller_df["GS10"].mean())
print("GS10 Std:", shiller_df["GS10"].std())

# Let's plot time series of the Real_Return_10Y
plt.figure(figsize=(10, 5))
plt.plot(shiller_df["Date"], shiller_df["Real_Return_10Y"], color="purple")
plt.title("10-Year Annualized Real Stock Return Over Time")
plt.xlabel("Date")
plt.ylabel("Real Return (%)")
plt.grid()
plt.show()

# Let's plot the distributions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(shiller_df["Inflation"].dropna(), bins=50, color="blue", alpha=0.7)
plt.title("Inflation Distribution")
plt.xlabel("Inflation (%)")
plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
plt.hist(shiller_df["GS10"].dropna(), bins=50, color="green", alpha=0.7)
plt.title("10-Year Treasury Yield Distribution")
plt.xlabel("GS10 (%)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot time series of Inflation and GS10
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(shiller_df["Date"], shiller_df["Inflation"], color="blue")
plt.title("Year-over-Year Inflation Over Time")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(shiller_df["Date"], shiller_df["GS10"], color="green")
plt.title("10-Year Treasury Yield Over Time")
plt.xlabel("Date")
plt.ylabel("GS10 (%)")
plt.tight_layout()
plt.grid()
plt.show()


# Save cleaned data
path = "/Users/kayttaja/Desktop/BDA/data/processed/Shiller_cleaned.csv"
shiller_df.to_csv(path, index=False)
