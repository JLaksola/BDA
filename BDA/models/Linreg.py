import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
file_path = "/Users/kayttaja/Desktop/BDA/data/processed/Shiller_cleaned.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
df.dropna(inplace=True)
print(df.describe())
print(df.head())
# Define the inverse CAPE variable
df["inv_CAPE"] = 1 / df["CAPE"]

# Define the target variable
target = "Real_Return_10Y"

# Let's initiate the inverse CAPE based linear regression model
train_start = "1881-01-01"
test_start = "1990-01-01"
test_end = "2015-02-01"

# Initialize empty RMSE list
rmse_list = []
predictions = []  # Store predicted values
actuals = []  # Store actual values
dates = []  # Store corresponding dates

# Rolling Forecast Loop
for date in pd.date_range(test_start, test_end, freq="MS"):  # Monthly steps
    train_end = (pd.Timestamp(date) - pd.DateOffset(years=10, months=1)).strftime(
        "%Y-%m-%d"
    )
    train = df.loc[train_start:train_end]  # Only use past data (10 years before)
    test_sample = df.loc[date:date]  # Predict one step ahead

    # Train model
    X_train = train[["CAPE"]]
    y_train = train[target]
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prepare test sample
    X_test = test_sample[["CAPE"]]
    y_test = test_sample[target]

    # Predict & Calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error([y_test], y_pred))
    rmse_list.append(rmse)
    predictions.append(float(y_pred[0]))
    actuals.append(y_test.values[0])
    dates.append(date)

# Convert results to DataFrame
results_df = pd.DataFrame(
    {
        "Date": dates,
        "Predicted": predictions,
        "Actual": actuals,
        "RMSE": rmse_list,
    }
)
results_df.set_index("Date", inplace=True)

# Print RMSE and R-squared
rmse = np.sqrt(mean_squared_error(results_df["Actual"], results_df["Predicted"]))
r_squared = np.corrcoef(results_df["Actual"], results_df["Predicted"])[0, 1] ** 2
print(f"Overall RMSE: {rmse}")
print(f"Overall R-squared: {r_squared}")


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Predicted"], label="Predicted", color="blue")
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="red")
plt.title("Predicted vs Actual Returns")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.legend()
plt.grid()
plt.show()


# To CSV
results_df.to_csv(
    "/Users/kayttaja/Desktop/BDA/reports/linear_CAPE_results.csv",
    index=True,
)
