import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
shiller_df = shiller_df.set_index("Date")
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

# Let's make a 10-year rolling average inflation value
shiller_df["Inflation_10Y_Rolling"] = shiller_df["Inflation"].rolling(window=120).mean()
print(shiller_df.tail())


# Create inflation categories based on past 10-year median
def inflation_cats(past_values, current_value):
    # Jos ei ole yhtään historiaa -> laitetaan negatiiviseen (tai voit valita jonkin muun defaultin)
    if len(past_values) == 0:
        return "negative"

    # Menneet positiiviset inflaatiot
    pos_values = [v for v in past_values if v >= 0]

    # Jos ei ole menneitä positiivisia havaintoja
    if len(pos_values) == 0:
        return "negative" if current_value < 0 else "low"

    # Ensin hoidetaan negatiiviset arvot
    if current_value < 0:
        return "negative"

    # Lasketaan 1/3- ja 2/3-quantile positiivisille
    q1, q2 = np.quantile(pos_values, [1 / 3, 2 / 3])

    # Jaotellaan kolmeen yhtä suureen osaan positiiviset
    if current_value < q1:
        return "low"  # matala
    elif current_value < q2:
        return "medium"  # keskitaso
    else:
        return "high"  # korkea


inflation_values = shiller_df["Inflation_10Y_Rolling"].astype(float).tolist()

start_date = pd.Timestamp("1990-01-01")

# Inflaatio sarjana
inflation = shiller_df["Inflation_10Y_Rolling"].astype(float)

# Maskit indeksin perusteella
pre_mask = shiller_df.index < start_date
post_mask = ~pre_mask

cats = pd.Series(index=shiller_df.index, dtype="object")

# --- 3.1 Pre-1980: kategoriat koko pre-1980 jakson jakauman mukaan ---

pre_pos_values = inflation[pre_mask & (inflation >= 0)].tolist()

if len(pre_pos_values) > 0:
    q1_pre, q2_pre = np.quantile(pre_pos_values, [1 / 3, 2 / 3])
else:
    q1_pre, q2_pre = None, None

for idx in shiller_df[pre_mask].index:
    val = inflation.loc[idx]

    if q1_pre is None:
        # fallback: jos ei ole positiivista dataa ennen 1980
        cats.loc[idx] = "negative" if val < 0 else "low"
    else:
        if val < 0:
            cats.loc[idx] = "negative"
        elif val < q1_pre:
            cats.loc[idx] = "low"
        elif val < q2_pre:
            cats.loc[idx] = "medium"
        else:
            cats.loc[idx] = "high"

# --- 3.2 Vuodesta 1980 alkaen: laajeneva ikkuna, joka sisältää myös pre-1980 datan ---

# Historiaksi kaikki pre-1980 inflaatiot
past_values = inflation[pre_mask].tolist()

for idx in shiller_df[post_mask].index:
    val = inflation.loc[idx]

    # Luokitus käyttäen KAIKKIA menneitä havaintoja (pre-1980 + 1980–t-1)
    cats.loc[idx] = inflation_cats(past_values, val)

    # Päivitä historia (laajeneva ikkuna)
    past_values.append(val)

# Lopuksi talteen
shiller_df["Inflation_Category"] = cats

print(shiller_df.tail(125))

# Let's plot time series of the Real_Return_10Y
plt.figure(figsize=(10, 5))
plt.plot(shiller_df.index, shiller_df["Real_Return_10Y"], color="purple")
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

# Plot time series of Inflation, GS10 and CAPE
plt.figure(figsize=(12, 5))
plt.subplot(3, 1, 1)
plt.plot(shiller_df["Date"], shiller_df["Inflation"], color="blue")
plt.title("Year-over-Year Inflation Over Time")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")
plt.grid()
plt.subplot(3, 1, 2)
plt.plot(shiller_df["Date"], shiller_df["GS10"], color="green")
plt.title("10-Year Treasury Yield Over Time")
plt.xlabel("Date")
plt.ylabel("GS10 (%)")
plt.tight_layout()
plt.grid()
plt.subplot(3, 1, 3)
plt.plot(shiller_df["Date"], shiller_df["CAPE"], color="orange")
plt.title("CAPE Over Time")
plt.xlabel("Date")
plt.ylabel("CAPE")
plt.grid()
plt.tight_layout()
plt.show()


# Save cleaned data
path = "/Users/kayttaja/Desktop/BDA/data/processed/Shiller_cleaned.csv"
shiller_df.to_csv(path, index=False)
