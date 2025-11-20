import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the data
file_path = "/Users/kayttaja/Desktop/BDA/data/processed/Shiller_cleaned.csv"
df = pd.read_csv(file_path)
# set the Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])
# set the index to Date
df = df.set_index("Date")
print(df.head())


print(df[["Inflation_Category", "Inflation_10Y_Rolling", "Inflation"]].tail(125))

# Plot inflation values over time with categories
plt.figure(figsize=(12, 6))
colors = {"negative": "red", "low": "green", "medium": "orange", "high": "purple"}
for cat, color in colors.items():
    sub = df[df["Inflation_Category"] == cat]
    plt.scatter(
        sub.index, sub["Inflation_10Y_Rolling"], label=cat, color=color, alpha=0.6
    )

plt.xlabel("Date")
plt.ylabel("Inflation")
plt.title("Inflation over Time by Category")
plt.legend()
plt.grid()
plt.show()

# Create a column for 10-year ahead real return
reg_df = df.dropna().copy()

# Malli: 10v tuotto ~ CAPE + inflaatiokategoriat
model = smf.ols(
    formula="Real_Return_10Y ~ CAPE + C(Inflation_Category)", data=reg_df
).fit()

print(model.summary())
# Get HAC robust standard errors
model_hac = model.get_robustcov_results(cov_type="HAC", maxlags=12)
print(model_hac.summary())

# Malli: 10v tuotto ~ CAPE + inflaatiokategoriat
model_CAPE = smf.ols(formula="Real_Return_10Y ~ CAPE", data=reg_df).fit()

print(model_CAPE.summary())

# Plotting the results
plt.figure(figsize=(12, 6))
plt.scatter(reg_df["CAPE"], reg_df["Real_Return_10Y"], label="Data Points", alpha=0.5)
# Create prediction line
cape_range = np.linspace(reg_df["CAPE"].min(), reg_df["CAPE"].max(), 100)
pred_df = pd.DataFrame({"CAPE": cape_range})
pred_df["Inflation_Category"] = "low"  # or any category for prediction
pred_df["Predicted_Return"] = model_CAPE.predict(pred_df)
plt.plot(
    pred_df["CAPE"], pred_df["Predicted_Return"], color="red", label="Regression Line"
)
plt.xlabel("CAPE")
plt.ylabel("10-Year Real Return (%)")
plt.title("10-Year Real Return vs CAPE with Regression Line")
plt.legend()
plt.grid()
plt.show()

categories = reg_df["Inflation_Category"].unique()

for cat in categories:
    sub = reg_df[reg_df["Inflation_Category"] == cat]

    X = sm.add_constant(sub["CAPE"])
    y = sub["Real_Return_10Y"]
    m_cat = sm.OLS(y, X).fit()

    cape_grid = np.linspace(sub["CAPE"].min(), sub["CAPE"].max(), 100)
    X_grid = sm.add_constant(cape_grid)
    y_hat = m_cat.predict(X_grid)

    plt.figure(figsize=(7, 4))
    plt.scatter(sub["CAPE"], sub["Real_Return_10Y"], alpha=0.5, label="Havainnot")
    plt.plot(cape_grid, y_hat, linewidth=2, label="OLS-regressioviiva")
    plt.title(f"10v-tuotto vs CAPE – oma OLS-malli, kategoria: {cat}")
    plt.xlabel("CAPE")
    plt.ylabel("10 vuoden eteenpäin tuotto")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"=== Kategoria: {cat} ===")
    print(m_cat.summary())
