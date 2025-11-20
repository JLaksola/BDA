import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statistics import median
import numpy as np

# Load the data
file_path = "/Users/kayttaja/Desktop/BDA/data/processed/Shiller_cleaned.csv"
df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"])
print(df.head())
print(df.tail())


# Let's plot the 10-year inflation rolling average
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Inflation_10Y_Rolling"], color="orange")
plt.title("10-Year Rolling Average Inflation Over Time")
plt.xlabel("Date")
plt.ylabel("10-Year Rolling Average Inflation (%)")
plt.grid()
plt.show()

# Let's make a linear regression line for returns trend over time
x = df.index.values.reshape(-1, 1)  # Dates as numerical values
y = df["Real_Return_10Y"]
linreg = LinearRegression()
linreg.fit(x, y)
predictions = linreg.predict(x)
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, y, label="Actual 10-Year Real Returns", color="blue")
plt.plot(
    df.index, predictions, label="Linear Regression Trend", color="red", linestyle="--"
)
plt.xlabel("Date")
plt.ylabel("10-Year Real Returns")
plt.title("10-Year Real Returns Trend Over Time")
plt.grid(True)
plt.legend()
plt.show()

# Plot time series of Inflation, GS10 and CAPE
plt.figure(figsize=(12, 5))
plt.subplot(1, 1, 1)
plt.plot(df["Date"], df["Inflation"], color="blue")
plt.title("Year-over-Year Inflation Over Time")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")
plt.grid()
plt.show()


# Let's make a linear regression line for CAPE trend over time
x_cape = df.index.values.reshape(-1, 1)  # Dates as numerical values
y_cape = df["CAPE"]
linreg_cape = LinearRegression()
linreg_cape.fit(x_cape, y_cape)
predictions_cape = linreg_cape.predict(x_cape)
# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df.index, y_cape, label="Actual CAPE", color="blue")
plt.plot(
    df.index, predictions_cape, label="Linear Regression Trend", color="red", linestyle="--"
)
plt.xlabel("Date")
plt.ylabel("CAPE")
plt.title("CAPE Trend Over Time")
plt.grid(True)
plt.legend()
plt.show()

# Create inflation categories based on past 10-year median
def inflation_cats(past_values, current_value):
    if len(past_values) == 0:
        return "negative"

    pos_values = [v for v in past_values if v >= 0]
    pos_median = 0 if len(pos_values) == 0 else median(pos_values)

    if current_value < 0:
        return "negative"
    elif current_value < pos_median:
        return "low"
    else:
        return "high"

inflation_values = df["Inflation"].astype(float).tolist()
cats = []

window = 120  # montako AIEMPAA havaintoa huomioidaan

for i, cur in enumerate(inflation_values):
    # otetaan vain viimeiset 120 ennen nykyistä indeksiä
    start = max(0, i - window)
    past = inflation_values[start:i]   # ei sisällä nykyistä
    cats.append(inflation_cats(past, cur))

df["Inflation_Category"] = cats


# Let's plot inflation values over time without categories
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Inflation"], label="Inflation", color="blue")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")
plt.title("Inflation Over Time")
plt.grid(True)
plt.legend()
plt.show()


