import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Load the data
file_path = "/Users/kayttaja/Desktop/BDA_project/data/Shiller_cleaned.csv"
df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# Plot time series of Inflation, GS10 and CAPE
plt.figure(figsize=(12, 8))
title_fs = 16
label_fs = 14
tick_fs = 12

plt.subplot(3, 1, 1)
plt.plot(df.index, df["Inflation"], color="blue")
plt.title("Year-over-Year Inflation Over Time", fontsize=title_fs)
plt.xlabel("Date", fontsize=label_fs)
plt.ylabel("Inflation (%)", fontsize=label_fs)
plt.grid()
plt.tick_params(axis="both", which="major", labelsize=tick_fs)

plt.subplot(3, 1, 2)
plt.plot(df.index, df["GS10"], color="green")
plt.title("10-Year Treasury Yield Over Time", fontsize=title_fs)
plt.xlabel("Date", fontsize=label_fs)
plt.ylabel("GS10 (%)", fontsize=label_fs)
plt.grid()
plt.tick_params(axis="both", which="major", labelsize=tick_fs)

plt.subplot(3, 1, 3)
plt.plot(df.index, df["CAPE"], color="orange")
plt.title("CAPE Over Time", fontsize=title_fs)
plt.xlabel("Date", fontsize=label_fs)
plt.ylabel("CAPE", fontsize=label_fs)
plt.grid()
plt.tick_params(axis="both", which="major", labelsize=tick_fs)

plt.tight_layout()
plt.show()


# Make a dataframe for regression
reg_df = df.dropna(subset=["CAPE", "Real_Return_10Y"])
# Let's plot a time series of CAPE and its trend line
plt.figure(figsize=(12, 6))
plt.plot(reg_df.index, reg_df["CAPE"], label="CAPE", color="blue")
# Trend line
z = np.polyfit(np.arange(len(reg_df)), reg_df["CAPE"], 1)
p = np.poly1d(z)
plt.plot(reg_df.index, p(np.arange(len(reg_df))), "r--", label="Trend Line")
plt.xlabel("Date")
plt.ylabel("CAPE")
plt.legend()
plt.grid()
plt.show()

# Let's plot a time series of 10-year real return and its trend line
plt.figure(figsize=(12, 6))
plt.plot(
    reg_df.index, reg_df["Real_Return_10Y"], label="10-Year Real Return", color="green"
)
# Trend line
z = np.polyfit(np.arange(len(reg_df)), reg_df["Real_Return_10Y"], 1)
p = np.poly1d(z)
plt.plot(reg_df.index, p(np.arange(len(reg_df))), "r--", label="Trend Line")
plt.xlabel("Date")
plt.ylabel("10-Year Real Return (%)")
plt.legend()
plt.grid()
plt.show()


# Rolling regression coefficients over time
window_years = 30
start_plot_date = pd.Timestamp("1920-01-01")

dates_rolling = []
slopes_rolling = []
intercepts_rolling = []

for end_date in reg_df.index:
    if end_date < start_plot_date:
        continue

    start_date = end_date - pd.DateOffset(years=window_years)
    sub = reg_df[(reg_df.index >= start_date) & (reg_df.index < end_date)].dropna(
        subset=["CAPE", "Real_Return_10Y"]
    )

    # skip if too few observations in the window
    if len(sub) < 10:
        continue

    X = sm.add_constant(sub["CAPE"])
    y = sub["Real_Return_10Y"]
    m_temp = sm.OLS(y, X).fit()

    dates_rolling.append(end_date)
    slopes_rolling.append(m_temp.params["CAPE"])
    intercepts_rolling.append(m_temp.params["const"])

# Plot as subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Intercepts
axes[0].plot(dates_rolling, intercepts_rolling, color="brown")
axes[0].set_ylabel(f"Rolling {window_years}-Year Intercept")
axes[0].grid(True)

# Slopes
axes[1].plot(dates_rolling, slopes_rolling, color="orange")
axes[1].set_ylabel(f"Rolling {window_years}-Year Slope")
axes[1].set_xlabel("Date")
axes[1].grid(True)

plt.tight_layout()
plt.show()
