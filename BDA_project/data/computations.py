import pandas as pd

# Load data
path1 = "/Users/kayttaja/Desktop/BDA/data/Shiller_data.xls"
shiller_df = pd.read_excel(path1, skiprows=7)

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
shiller_df = shiller_df.set_index("Date")

# Compute year over year inflation
shiller_df["Inflation"] = shiller_df["CPI"].pct_change(periods=12) * 100
# Compute Real_Return_10Y to percent
shiller_df["Real_Return_10Y"] = shiller_df["Real_Return_10Y"] * 100

# Save cleaned data
path = "/Users/kayttaja/Desktop/BDA/data/Shiller_cleaned.csv"
shiller_df.to_csv(path, index=True)
